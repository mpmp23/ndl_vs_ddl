
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from z3 import Bool, BoolRef, And, Not, Implies


@dataclass
class CompiledDDL:
    rules: list[BoolRef]
    situation: list[BoolRef]
    hypothesis: BoolRef
    vocabulary: set[str]


class DDLCompiler:

    def __init__(self):
        self.vocabulary = set()
        self.facts = {}
        self.modals = {}

    # -----------------------------------
    # 1. PARSE DDL LITERALS
    # -----------------------------------

    def literal(self, text, from_rule=False):

        text = text.strip()

        pattern = (
            r"(?P<outer>-)?\s*"
            r"(?:\[(?P<modal>O|P|F)\]\s*)?"
            r"(?P<inner>-)?\s*"
            r"(?P<name>[a-z][A-Za-z0-9]*)\(X\)"
        )

        match = re.fullmatch(pattern, text)

        if not match:
            raise ValueError(f"Unsupported DDL literal: {text}")

        name = match.group("name")
        modal = match.group("modal")
        outer_neg = bool(match.group("outer"))
        inner_neg = bool(match.group("inner"))

        # Rules establish the permitted vocabulary.
        if from_rule:
            self.vocabulary.add(name)

        elif name not in self.vocabulary:
            raise ValueError(f"Unknown predicate: {name}")

        # -----------------------------------
        # Factual literal
        # -----------------------------------

        if modal is None:

            if inner_neg:
                raise ValueError(f"Invalid factual negation: {text}")

            if name not in self.facts:
                self.facts[name] = Bool(f"fact_{name}")

            atom = self.facts[name]

            return Not(atom) if outer_neg else atom

        # -----------------------------------
        # Deontic literal
        # -----------------------------------

        # Prohibition: F(A) = O(not A)
        if modal == "F":
            modal = "O"
            inner_neg = not inner_neg

        polarity = "negative" if inner_neg else "positive"
        key = (modal, polarity, name)

        if key not in self.modals:
            symbol = f"{modal}_{polarity}_{name}"
            self.modals[key] = Bool(symbol)

        expression = self.modals[key]

        # Negating the operator is different
        # from negating its action.
        return Not(expression) if outer_neg else expression

    # -----------------------------------
    # 2. COMPILE ONE DDL RULE
    # -----------------------------------

    def compile_rule(self, text):

        parts = text.strip().split("=>")

        if len(parts) != 2:
            raise ValueError(f"Invalid DDL rule: {text}")

        antecedent_text, consequent_text = parts

        conditions = [
            self.literal(part, from_rule=True)
            for part in antecedent_text.split(",")
        ]

        consequence = self.literal(
            consequent_text,
            from_rule=True
        )

        return Implies(
            And(*conditions),
            consequence
        )

    # -----------------------------------
    # 3. COMPILE CONTRACT XML
    # -----------------------------------

    def compile_rules(self, xml_text):

        root = ET.fromstring(xml_text)
        compiled = []

        rule_elements = list(root.iter("Rule"))

        if not rule_elements:
            raise ValueError("No DDL rules found in XML")

        for rule in rule_elements:

            # Accept <Simple> or direct rule text.
            text = rule.findtext("Simple")

            if text is None:
                text = rule.text

            if not text or not text.strip():
                raise ValueError(
                    f"Missing rule expression: {rule.get('ruleLabel')}"
                )

            try:
                expression = self.compile_rule(text)
            except ValueError as error:
                label = rule.get("ruleLabel", "unknown")
                raise ValueError(
                    f"Rule {label}: {error}"
                ) from error

            compiled.append(expression)

        return compiled

    # -----------------------------------
    # 4. COMPILE QUERY XML
    # -----------------------------------

    def compile_query(self, xml_text):

        root = ET.fromstring(xml_text)

        if root.tag != "Query":
            raise ValueError("Expected <Query> root")

        if root.get("status") != "ok":
            raise ValueError(
                "Query translation was not successful"
            )

        situation_element = root.find("Situation")
        hypothesis_element = root.find("Hypothesis")

        if situation_element is None:
            raise ValueError("Missing Situation")

        if hypothesis_element is None:
            raise ValueError("Missing Hypothesis")

        facts = situation_element.findall("Fact")

        if not facts:
            raise ValueError("No situation facts found")

        situation = []

        for fact in facts:
            if not fact.text or not fact.text.strip():
                raise ValueError("Empty situation fact")

            situation.append(self.literal(fact.text))

        if not hypothesis_element.text:
            raise ValueError("Empty hypothesis")

        hypothesis = self.literal(
            hypothesis_element.text
        )

        return situation, hypothesis


# -----------------------------------
# 5. PUBLIC COMPILER FUNCTION
# -----------------------------------

def compile_ddl(rules_xml, query_xml):

    compiler = DDLCompiler()

    rules = compiler.compile_rules(rules_xml)

    situation, hypothesis = compiler.compile_query(
        query_xml
    )

    return CompiledDDL(
        rules=rules,
        situation=situation,
        hypothesis=hypothesis,
        vocabulary=compiler.vocabulary.copy()
    )