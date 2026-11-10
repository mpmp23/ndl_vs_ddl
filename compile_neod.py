"""Compile restricted Neo-Davidsonian XML rules and queries into Z3.

One shared compiler instance handles the rules and the query so predicates,
case constants, and deontic action schemas remain consistent.

Supported syntax: ∀, ∃, ¬, ∧, ∨, →, parentheses, square-bracket scopes,
                  n-ary predicates, [O], [P], and [F].

The deontic fragment is deliberately restricted: [F] A means [O] ¬A,
while permissions and obligations remain distinct, uninterpreted modal facts.
No defeasible priorities or weak permissions are inferred.
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from z3 import (
    And, BoolRef, BoolSort, Const, DeclareSort, Exists, ForAll,
    Function, Implies, Not, Or,
)


@dataclass
class CompiledNeoD:
    rules: list[BoolRef]
    situation: list[BoolRef]
    hypothesis: BoolRef
    vocabulary: set[str]


# Tokenize the exact NeoD notation used in the prompt.
_TOKEN = re.compile(
    r"\s*(\[O\]|\[P\]|\[F\]|∀|∃|¬|∧|∨|→|=>|[(),\[\]]|"
    r"[A-Za-z_][A-Za-z0-9_]*|[0-9][A-Za-z0-9_]*)"
)
_NAME = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|[0-9][A-Za-z0-9_]*)\Z")


def _tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        if text[pos:].isspace():
            break
        match = _TOKEN.match(text, pos)
        if not match:
            raise ValueError(f"Unsupported NeoD syntax near: {text[pos:pos + 35]!r}")
        tokens.append(match.group(1))
        pos = match.end()
    return tokens


class _Parser:
    """Small parser; produces syntax trees instead of running Python code."""

    def __init__(self, text):
        self.tokens = _tokenize(text)
        self.position = 0

    def peek(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def take(self):
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of NeoD expression")
        self.position += 1
        return token

    def accept(self, *options):
        if self.peek() in options:
            return self.take()
        return None

    def expect(self, expected):
        actual = self.take()
        if actual != expected:
            raise ValueError(f"Expected {expected!r}, got {actual!r}")

    def name(self):
        token = self.take()
        if not _NAME.fullmatch(token):
            raise ValueError(f"Expected a name, got {token!r}")
        return token

    def parse(self):
        expression = self.implication()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token: {self.peek()!r}")
        return expression

    def implication(self):
        left = self.disjunction()
        if self.accept("→", "=>"):
            return ("implies", left, self.implication())
        return left

    def disjunction(self):
        left = self.conjunction()
        while self.accept("∨"):
            left = ("or", left, self.conjunction())
        return left

    def conjunction(self):
        left = self.unary()
        while self.accept("∧"):
            left = ("and", left, self.unary())
        return left

    def unary(self):
        if self.accept("¬"):
            return ("not", self.unary())

        modal = self.accept("[O]", "[P]", "[F]")
        if modal:
            return ("modal", modal[1], self.unary())

        quantifier = self.accept("∀", "∃")
        if quantifier:
            names = [self.name()]
            while self.accept(","):
                names.append(self.name())
            if len(set(names)) != len(names):
                raise ValueError("Repeated quantifier variable")

            # Both ∃e1,e2 [P] and ∃e1 ∃e2 [P] are valid.
            # Nested quantifiers can have different kinds (∀s ∃e [P]).
            if self.accept("["):
                body = self.implication()
                self.expect("]")
            elif self.peek() in ("∀", "∃"):
                body = self.unary()
            else:
                raise ValueError(
                    f"Expected '[' or another quantifier, got {self.peek()!r}"
                )

            # Adjacent same-kind quantifiers are logically equivalent to a
            # grouped declaration. Normalize their ASTs (not the source rules)
            # so equal obligation descriptions receive the same modal schema.
            if (body[0] == "quant" and body[1] == quantifier
                    and not (set(names) & set(body[2]))):
                return ("quant", quantifier, tuple(names) + body[2], body[3])
            return ("quant", quantifier, tuple(names), body)

        opening = self.accept("(", "[")
        if opening:
            body = self.implication()
            self.expect(")" if opening == "(" else "]")
            return body

        predicate = self.name()
        self.expect("(")
        args = [self.name()]
        while self.accept(","):
            args.append(self.name())
        self.expect(")")
        return ("atom", predicate, tuple(args))


def _unbound_rule_variables(node, bound=frozenset()):
    """Find unbound lowercase symbols in a generated *rule*.

    By the NeoD prompt convention, lowercase terms are variables; uppercase
    or digit-initial terms are fixed symbols (e.g., Email, 2WorkingDays).
    Query case referents are deliberately exempt: this validator is applied
    only to rules, never to a query's shared sCase / eReceived constants.
    """
    tag = node[0]
    if tag == "atom":
        return {term for term in node[2]
                if term[0].islower() and term not in bound}
    if tag in ("and", "or", "implies"):
        return (_unbound_rule_variables(node[1], bound)
                | _unbound_rule_variables(node[2], bound))
    if tag == "not":
        return _unbound_rule_variables(node[1], bound)
    if tag == "modal":
        return _unbound_rule_variables(node[2], bound)
    if tag == "quant":
        return _unbound_rule_variables(node[3], bound | set(node[2]))
    raise ValueError(f"Unsupported expression node: {tag}")


def _flatten(node, connective):
    if node[0] == connective:
        return _flatten(node[1], connective) + _flatten(node[2], connective)
    return [node]


def _sort_key(node):
    """Stable traversal for common conjunct/disjunct reordering.

    Variable spellings are intentionally ignored in this sorting key.
    """
    tag = node[0]
    if tag == "atom":
        args = tuple(
            ("constant", a) if (a[0].isupper() or a[0].isdigit())
            else ("referent",)
            for a in node[2]
        )
        return (tag, node[1], args)
    if tag in ("and", "or"):
        return (tag, tuple(sorted((_sort_key(n) for n in _flatten(node, tag)))))
    if tag == "quant":
        return (tag, node[1], len(node[2]), _sort_key(node[3]))
    if tag == "not":
        return (tag, _sort_key(node[1]))
    if tag == "implies":
        return (tag, _sort_key(node[1]), _sort_key(node[2]))
    raise ValueError(f"Unsupported deontic action structure: {tag}")


class NeoDCompiler:
    def __init__(self):
        # A single uninterpreted universe for entities, events, and fixed terms.
        self.individual = DeclareSort("NeoDIndividual")
        self.predicate_arities = {}
        self.predicates = {}
        self.constants = {}
        self.modal_predicates = {}

    @property
    def vocabulary(self):
        return set(self.predicate_arities)

    def _validate_predicates(self, node, from_rule):
        tag = node[0]
        if tag == "atom":
            name, args = node[1], node[2]
            arity = len(args)
            if from_rule and name not in self.predicate_arities:
                self.predicate_arities[name] = arity
            if self.predicate_arities.get(name) != arity:
                raise ValueError(f"Unknown predicate or mismatched arity: {name}/{arity}")
        elif tag in ("and", "or", "implies"):
            self._validate_predicates(node[1], from_rule)
            self._validate_predicates(node[2], from_rule)
        elif tag == "not":
            self._validate_predicates(node[1], from_rule)
        elif tag == "modal":
            self._validate_predicates(node[2], from_rule)
        elif tag == "quant":
            self._validate_predicates(node[3], from_rule)
        else:
            raise ValueError(f"Unsupported expression node: {tag}")

    def _term(self, name, bound):
        if name in bound:
            return bound[name]
        if name not in self.constants:
            self.constants[name] = Const(name, self.individual)
        return self.constants[name]

    def _modal_signature(self, node):
        """Reify a quantified action description into a stable modal schema.

        Existential event names (e1 vs e2) and case variable names (s vs sCase)
        do not alter the schema, but predicate/role/temporal names do.
        """
        free_names = []
        bound_counter = [0]

        def shape(current, scoped):
            tag = current[0]
            if tag == "atom":
                args = []
                for term in current[2]:
                    if term in scoped:
                        args.append(("bound", scoped[term]))
                    elif term[0].isupper() or term[0].isdigit():
                        args.append(("fixed", term))
                    else:
                        if term not in free_names:
                            free_names.append(term)
                        args.append(("free", free_names.index(term)))
                return ("atom", current[1], tuple(args))

            if tag in ("and", "or"):
                children = sorted(_flatten(current, tag), key=_sort_key)
                return (tag, tuple(shape(child, scoped) for child in children))

            if tag == "quant":
                new_scope = dict(scoped)
                for name in current[2]:
                    new_scope[name] = bound_counter[0]
                    bound_counter[0] += 1
                return ("quant", current[1], len(current[2]), shape(current[3], new_scope))

            if tag == "not":
                return ("not", shape(current[1], scoped))

            if tag == "implies":
                return ("implies", shape(current[1], scoped), shape(current[2], scoped))

            # Not a full nested deontic calculus.
            raise ValueError("Nested deontic operators are not supported")

        return shape(node, {}), free_names

    def _modal(self, kind, action, bound):
        # The same restricted convention as compile_ddl.py:
        # prohibition [F] A is an obligation [O] ¬A.
        if kind == "F":
            kind = "O"
            action = ("not", action)

        signature, free_names = self._modal_signature(action)
        encoded = json.dumps(signature, ensure_ascii=False, separators=(",", ":"))
        key = (kind, encoded)

        if key not in self.modal_predicates:
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
            name = f"modal_{kind}_{digest}"
            self.modal_predicates[key] = Function(
                name, *([self.individual] * len(free_names)), BoolSort()
            )

        arguments = [self._term(name, bound) for name in free_names]
        return self.modal_predicates[key](*arguments)

    def _compile(self, node, bound=None):
        if bound is None:
            bound = {}
        tag = node[0]

        if tag == "atom":
            name, args = node[1], node[2]
            if name not in self.predicates:
                self.predicates[name] = Function(
                    name, *([self.individual] * len(args)), BoolSort()
                )
            return self.predicates[name](*[self._term(a, bound) for a in args])

        if tag == "not":
            return Not(self._compile(node[1], bound))
        if tag == "and":
            return And(self._compile(node[1], bound), self._compile(node[2], bound))
        if tag == "or":
            return Or(self._compile(node[1], bound), self._compile(node[2], bound))
        if tag == "implies":
            return Implies(self._compile(node[1], bound), self._compile(node[2], bound))

        if tag == "quant":
            quantifier, names, body = node[1], node[2], node[3]
            bindings = {name: Const(name, self.individual) for name in names}
            scoped = {**bound, **bindings}
            compiled_body = self._compile(body, scoped)
            if quantifier == "∀":
                return ForAll(list(bindings.values()), compiled_body)
            return Exists(list(bindings.values()), compiled_body)

        if tag == "modal":
            return self._modal(node[1], node[2], bound)

        raise ValueError(f"Unsupported NeoD structure: {tag}")

    def compile_rules(self, xml_text):
        root = ET.fromstring(xml_text)
        elements = list(root.iter("Rule"))
        if not elements:
            raise ValueError("No <Rule> elements found")

        parsed = []
        for element in elements:
            label = element.get("ruleLabel", "unknown")
            content = element.findtext("NeoDavidsonian")
            if not content or not content.strip():
                raise ValueError(f"Rule {label}: missing <NeoDavidsonian> expression")
            try:
                tree = _Parser(content).parse()
                unbound = sorted(_unbound_rule_variables(tree))
                if unbound:
                    raise ValueError(
                        "Unbound rule variable(s): " + ", ".join(unbound)
                        + "; refusing to treat them as fixed constants"
                    )
                self._validate_predicates(tree, from_rule=True)
                parsed.append((label, tree))
            except ValueError as error:
                raise ValueError(f"Rule {label}: {error}") from error

        return [self._compile(tree) for _label, tree in parsed]

    def compile_query(self, xml_text):
        root = ET.fromstring(xml_text)
        if root.tag != "Query":
            raise ValueError("Expected a <Query> root element")
        if root.get("status") != "ok":
            raise ValueError("Query translation was not successful")

        situation_text = root.findtext("Situation/NeoDavidsonian")
        hypothesis_text = root.findtext("Hypothesis/NeoDavidsonian")
        if not situation_text or not situation_text.strip():
            raise ValueError("Missing <Situation><NeoDavidsonian> expression")
        if not hypothesis_text or not hypothesis_text.strip():
            raise ValueError("Missing <Hypothesis><NeoDavidsonian> expression")

        situation_ast = _Parser(situation_text).parse()
        hypothesis_ast = _Parser(hypothesis_text).parse()
        self._validate_predicates(situation_ast, from_rule=False)
        self._validate_predicates(hypothesis_ast, from_rule=False)

        return [self._compile(situation_ast)], self._compile(hypothesis_ast)


def compile_neod(rules_xml, query_xml):
    """Compile both XML inputs together, using one shared symbol vocabulary."""
    compiler = NeoDCompiler()
    rules = compiler.compile_rules(rules_xml)
    situation, hypothesis = compiler.compile_query(query_xml)
    return CompiledNeoD(
        rules=rules,
        situation=situation,
        hypothesis=hypothesis,
        vocabulary=compiler.vocabulary,
    )
