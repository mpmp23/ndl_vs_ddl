"""Shared Z3 verifier for the LexFlow DDL / Neo-Davidsonian pipelines.

Both compilers are expected to return:
    compiled.rules      -> list of Z3 Boolean expressions
    compiled.situation  -> list of Z3 Boolean expressions
    compiled.hypothesis -> one Z3 Boolean expression

This module does NOT parse XML, translate natural language, or redefine deontic
operators. Those representations have already been compiled into Z3 formulas.

Three labels:
  Entailment    R and S imply H
  Contradiction R and S imply not H
  NotMentioned  neither is forced by R and S

An inconsistent rule+scenario base is flagged separately as 'Inconsistent'.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from z3 import BoolRef, Not, Solver, sat, unsat


@dataclass
class VerificationResult:
    prediction: str
    checks: dict[str, str]
    check_seconds: dict[str, float]
    z3_seconds: float
    solver_inputs: dict[str, str]
    models: dict[str, str]


def verify_detailed(rules, situation, hypothesis, timeout_ms=None):
    """Run the consistency and two hypothesis checks.

    Parameters
    ----------
    rules, situation : list[z3.BoolRef]
        Precompiled Z3 formulas (all are asserted).
    hypothesis : z3.BoolRef
        Precompiled proposition to test.
    timeout_ms : int | None
        Optional per-check timeout. No timeout by default.

    Returns
    -------
    VerificationResult
        Includes predicted label, individual check results/times,
        SMT-LIB text for every executed check, and satisfying models.

    Raises
    ------
    ValueError
        A supplied formula is not a Z3 Boolean expression.
    RuntimeError
        Z3 returns 'unknown'; no E/C/N label is guessed.
    """
    rules = list(rules)
    situation = list(situation)

    for formula in [*rules, *situation, hypothesis]:
        if not isinstance(formula, BoolRef):
            raise ValueError(f"Expected a Z3 Boolean expression, got: {formula!r}")

    checks = {}
    check_seconds = {}
    solver_inputs = {}
    models = {}
    total_start = time.perf_counter()

    def run_check(name, extra=None):
        solver = Solver()
        if timeout_ms is not None:
            solver.set(timeout=timeout_ms)
        solver.add(*rules, *situation)
        if extra is not None:
            solver.add(extra)

        # Save the actual SMT-LIB input before calling the solver.
        solver_inputs[name] = solver.to_smt2()
        start = time.perf_counter()
        outcome = solver.check()
        check_seconds[name] = time.perf_counter() - start
        checks[name] = str(outcome)

        if outcome == sat:
            models[name] = str(solver.model())
        elif outcome != unsat:
            raise RuntimeError(
                f"Z3 could not decide the {name} check: "
                f"{outcome} ({solver.reason_unknown()})"
            )
        return outcome

    # 1. An inconsistent premise set cannot be assigned an E/C/N label.
    if run_check("consistency") == unsat:
        prediction = "Inconsistent"
    else:
        # 2. If R & S & not(H) is impossible, R & S entail H.
        entailment = run_check("entailment", Not(hypothesis))

        # 3. If R & S & H is impossible, R & S contradict H.
        contradiction = run_check("contradiction", hypothesis)

        if entailment == unsat and contradiction == sat:
            prediction = "Entailment"
        elif contradiction == unsat and entailment == sat:
            prediction = "Contradiction"
        elif entailment == sat and contradiction == sat:
            prediction = "NotMentioned"
        else:
            raise RuntimeError("Unexpected solver results for a consistent premise set")

    return VerificationResult(
        prediction=prediction,
        checks=checks,
        check_seconds=check_seconds,
        z3_seconds=time.perf_counter() - total_start,
        solver_inputs=solver_inputs,
        models=models,
    )


def verify(rules, situation, hypothesis):
    """Simple interface: return just Entailment, Contradiction, etc."""
    return verify_detailed(rules, situation, hypothesis).prediction


def verify_compiled(compiled):
    """Accept output from compile_ddl() or compile_neod() directly."""
    return verify_detailed(compiled.rules, compiled.situation, compiled.hypothesis)


def save_verification(result, path, *, run_id=None, query_id=None, framework=None):
    """Save the solver inputs, results, model(s), and timings as JSON."""
    data = asdict(result)
    if run_id is not None:
        data["run_id"] = run_id
    if query_id is not None:
        data["query_id"] = query_id
    if framework is not None:
        data["framework"] = framework

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
