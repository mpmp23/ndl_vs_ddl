
"""Evaluate the frozen LexFlow DDL vs NeoD verification benchmark.

Reads individual result.json files, checks saved queries against the
current benchmark, selects the latest matching attempt per case,
and counts missing cases. Does not modify verification artifacts.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


LABELS = ("Entailment", "Contradiction", "NotMentioned")
FRAMEWORKS = ("ddl", "neod")
MODELS = (
    "claude_sonnet_45", "deepseek_v32", "gemini_pro_3_preview",
    "gpt_oss_120b", "gpt52", "kimi_k2_thinking",
    "gpt6_astra", "claude_sonnet_55", "gemini4_argon",
)
TIMING_FIELDS = (
    "llm_seconds", "compile_seconds", "z3_seconds", "total_seconds"
)
TOKEN_FIELDS = (
    "prompt_tokens", "completion_tokens",
    "reasoning_tokens", "total_tokens"
)


def normalize(s):
    return " ".join(str(s or "").split())


def sections(value):
    if isinstance(value, list):
        return tuple(str(s).strip() for s in value)
    return tuple(
        s.strip()
        for s in str(value or "").split(";")
        if s.strip()
    )


def read_queries(path):
    root = ET.parse(path).getroot()
    if root.tag not in {"Queries", "Query"}:
        raise ValueError("Expected <Queries> or <Query> root")

    elements = (
        [root] if root.tag == "Query"
        else root.findall("Query")
    )
    result = {}

    for e in elements:
        qid = e.get("queryId")
        if not qid or qid in result:
            raise ValueError(f"Missing or repeated ID: {qid}")

        result[qid] = {
            "scenario_id": e.get("scenarioId"),
            "situation": normalize(e.findtext("Situation")),
            "hypothesis": normalize(e.findtext("Hypothesis")),
            "sections": tuple(
                normalize(x.text)
                for x in e.findall("RelevantSections/Section")
            ),
        }

    return result


def read_gold(path, queries):
    gold = {}

    with path.open(newline="", encoding="utf-8-sig") as f:
        for line in csv.DictReader(f):
            qid = line["query_id"]
            label = line["gold_label"]

            if (
                qid in gold
                or qid not in queries
                or label not in LABELS
            ):
                raise ValueError(f"Invalid gold row: {line}")

            if line["scenario_id"] != queries[qid]["scenario_id"]:
                raise ValueError(f"Scenario mismatch: {qid}")

            if (
                sections(line["relevant_sections"])
                != queries[qid]["sections"]
            ):
                raise ValueError(f"Section mismatch: {qid}")

            gold[qid] = label

    if set(gold) != set(queries):
        raise ValueError(
            f"XML/CSV IDs differ: {set(gold) ^ set(queries)}"
        )

    return gold


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return round(a / b, 4) if b else 0.0


def average(rows, field):
    values = [
        float(r[field])
        for r in rows
        if r.get(field) not in (None, "")
    ]
    return round(sum(values) / len(values), 3) if values else ""


def summarize(rows, name, value):
    total = len(rows)
    classified = sum(r["classified"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    compiled = sum(r["compiled"] for r in rows)
    verification_saved = sum(
        r["verification_saved"] for r in rows
    )

    result = {
        name: value,
        "cases": total,
        "correct": correct,
        "classified": classified,
        "end_to_end_accuracy": ratio(correct, total),
        "supported_accuracy": (
            ratio(correct, classified) if classified else ""
        ),
        "classification_coverage": ratio(classified, total),
        "compilation_success_rate": ratio(compiled, total),
        "verification_recorded_rate": ratio(
            verification_saved, total
        ),
    }

    counts = Counter(r["status"] for r in rows)

    for status in (
        "ok", "unsupported", "compile_error",
        "translation_error", "inconsistent", "missing",
        "solver_error", "other_error", "invalid_prediction"
    ):
        result["status_" + status] = counts[status]

    for field in TIMING_FIELDS + TOKEN_FIELDS:
        result["mean_" + field] = average(rows, field)
        result["n_" + field] = sum(
            r.get(field) not in (None, "")
            for r in rows
        )

    for label in LABELS:
        tp = sum(
            r["gold_label"] == label
            and r["predicted"] == label
            for r in rows
        )
        fp = sum(
            r["gold_label"] != label
            and r["predicted"] == label
            for r in rows
        )
        fn = sum(
            r["gold_label"] == label
            and r["predicted"] != label
            for r in rows
        )

        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        f1 = ratio(
            2 * precision * recall,
            precision + recall
        )

        result["tp_" + label] = tp
        result["fp_" + label] = fp
        result["fn_" + label] = fn
        result["precision_" + label] = precision
        result["recall_" + label] = recall
        result["f1_" + label] = f1

    result["macro_f1_end_to_end"] = round(
        sum(result["f1_" + label] for label in LABELS) / 3,
        4
    )

    # Calculate a second Macro-F1 using classified cases only.
    filtered = [r for r in rows if r["classified"]]
    f1_scores = []

    for label in LABELS:
        tp = sum(
            r["gold_label"] == label
            and r["predicted"] == label
            for r in filtered
        )
        fp = sum(
            r["gold_label"] != label
            and r["predicted"] == label
            for r in filtered
        )
        fn = sum(
            r["gold_label"] == label
            and r["predicted"] != label
            for r in filtered
        )

        p = ratio(tp, tp + fp)
        rec = ratio(tp, tp + fn)
        f1_scores.append(ratio(2 * p * rec, p + rec))

    result["macro_f1_supported"] = (
        round(sum(f1_scores) / 3, 4)
        if filtered else ""
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--runs", type=Path,
        default=Path("runs/verification")
    )
    parser.add_argument(
        "--queries", type=Path,
        default=Path("queries.xml")
    )
    parser.add_argument(
        "--gold", type=Path,
        default=Path("gold_labels.csv")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("runs/evaluation")
    )
    args = parser.parse_args()

    queries = read_queries(args.queries)
    gold = read_gold(args.gold, queries)
    args.out.mkdir(parents=True, exist_ok=True)

    latest = {}
    audit = []
    counts = Counter()

    # Read saved runs, not the append-only results.csv.
    for path in sorted(args.runs.rglob("result.json")):
        counts["result_files"] += 1

        try:
            record = json.loads(
                path.read_text(encoding="utf-8")
            )
            qid = record["query_id"]
            model = record["encoding_model"]
            framework = record["framework"]
            run_id = record["run_id"]
            key = (qid, model, framework)

            if (
                qid not in queries
                or model not in MODELS
                or framework not in FRAMEWORKS
            ):
                reason = "unknown_query_model_or_framework"
            else:
                input_path = path.with_name("query_input.json")

                if not input_path.is_file():
                    reason = "missing_query_input"
                else:
                    input_data = json.loads(
                        input_path.read_text(encoding="utf-8")
                    )
                    expected = queries[qid]

                    same = (
                        input_data.get("query_id") == qid
                        and normalize(
                            input_data.get("situation")
                        ) == expected["situation"]
                        and normalize(
                            input_data.get("hypothesis")
                        ) == expected["hypothesis"]
                        and sections(
                            input_data.get("sections")
                        ) == expected["sections"]
                    )

                    reason = (
                        "candidate" if same
                        else "outdated_query_text"
                    )

            if reason == "candidate":
                timestamp = record.get("started_at") or ""

                # Latest matching attempt, including failures.
                if (
                    key not in latest
                    or (timestamp, run_id) > latest[key][0]
                ):
                    latest[key] = (
                        (timestamp, run_id), record, path
                    )

            audit.append({
                "run_id": run_id,
                "query_id": qid,
                "encoding_model": model,
                "framework": framework,
                "status": record.get("status"),
                "audit_status": reason,
                "result_path": str(path),
            })
            counts[reason] += 1

        except (
            OSError, ValueError, KeyError,
            json.JSONDecodeError
        ) as exc:
            audit.append({
                "run_id": "",
                "query_id": "",
                "encoding_model": "",
                "framework": "",
                "status": "",
                "audit_status": "invalid_file",
                "result_path": str(path),
                "detail": str(exc),
            })
            counts["invalid_file"] += 1

    cases = []

    # Generate all expected combinations, including missing runs.
    for model in MODELS:
        for framework in FRAMEWORKS:
            for qid in sorted(gold):
                item = latest.get(
                    (qid, model, framework)
                )
                record, path = (
                    (item[1], item[2])
                    if item else ({}, None)
                )

                status = str(
                    record.get("status") or "missing"
                ).lower()

                prediction = record.get("prediction")
                classified = int(
                    status == "ok"
                    and prediction in LABELS
                )

                if status == "ok" and not classified:
                    status = "invalid_prediction"

                row = {
                    "query_id": qid,
                    "scenario_id": queries[qid]["scenario_id"],
                    "encoding_model": model,
                    "framework": framework,
                    "gold_label": gold[qid],
                    "predicted": (
                        prediction if classified else ""
                    ),
                    "status": status,
                    "classified": classified,
                    "correct": int(
                        classified
                        and prediction == gold[qid]
                    ),
                    "compiled": int(bool(
                        path and path.with_name(
                            "compiled_z3.json"
                        ).is_file()
                    )),
                    "verification_saved": int(bool(
                        path and path.with_name(
                            "verification.json"
                        ).is_file()
                    )),
                    "run_id": record.get("run_id", ""),
                    "encoding_source": record.get(
                        "encoding_source", ""
                    ),
                    "query_model": record.get(
                        "query_model", ""
                    ),
                    "sections": ";".join(
                        queries[qid]["sections"]
                    ),
                    "error": record.get("error") or "",
                    "result_path": (
                        str(path) if path else ""
                    ),
                    "rules_sha256": record.get(
                        "rules_sha256", ""
                    ),
                }

                for field in TIMING_FIELDS + TOKEN_FIELDS:
                    row[field] = record.get(field, "")

                cases.append(row)

    # Record which attempts were excluded.
    for row in audit:
        if row["audit_status"] == "candidate":
            key = (
                row["query_id"],
                row["encoding_model"],
                row["framework"]
            )
            chosen = latest.get(key)

            row["audit_status"] = (
                "selected"
                if chosen
                and row["result_path"] == str(chosen[2])
                else "older_attempt"
            )

    write_csv(
        args.out / "case_results.csv",
        cases, list(cases[0])
    )
    write_csv(
        args.out / "run_audit.csv",
        audit,
        [
            "run_id", "query_id", "encoding_model",
            "framework", "status", "audit_status",
            "result_path", "detail"
        ]
    )

    by_model = [
        summarize(
            [
                r for r in cases
                if r["encoding_model"] == model
                and r["framework"] == fw
            ],
            "model_framework",
            f"{model}/{fw}"
        )
        for model in MODELS
        for fw in FRAMEWORKS
    ]

    by_framework = [
        summarize(
            [r for r in cases if r["framework"] == fw],
            "framework", fw
        )
        for fw in FRAMEWORKS
    ]

    write_csv(
        args.out / "model_summary.csv",
        by_model, list(by_model[0])
    )
    write_csv(
        args.out / "framework_summary.csv",
        by_framework, list(by_framework[0])
    )

    # Confusion matrix with failures kept separate.
    matrix = []

    for fw in FRAMEWORKS:
        for label in LABELS:
            group = [
                r for r in cases
                if r["framework"] == fw
                and r["gold_label"] == label
            ]

            matrix.append({
                "framework": fw,
                "gold_label": label,
                **{
                    f"pred_{p}": sum(
                        r["predicted"] == p
                        for r in group
                    )
                    for p in LABELS
                },
                "no_classification": sum(
                    not r["classified"] for r in group
                ),
                "total": len(group),
            })

    write_csv(
        args.out / "confusion_matrix.csv",
        matrix, list(matrix[0])
    )

    # Per-label metrics and computation costs.
    label_rows = []

    for fw in FRAMEWORKS:
        for model in ("ALL_MODELS",) + MODELS:
            group = [
                r for r in cases
                if r["framework"] == fw
                and (
                    model == "ALL_MODELS"
                    or r["encoding_model"] == model
                )
            ]
            score = summarize(
                group, "scope", f"{model}/{fw}"
            )

            for label in LABELS:
                subset = [
                    r for r in group
                    if r["gold_label"] == label
                ]

                row = {
                    "framework": fw,
                    "encoding_model": model,
                    "gold_label": label,
                    "gold_cases": len(subset),
                    "classified_gold_cases": sum(
                        r["classified"] for r in subset
                    ),
                    "tp": score["tp_" + label],
                    "fp": score["fp_" + label],
                    "fn_including_failures": score[
                        "fn_" + label
                    ],
                    "precision": score[
                        "precision_" + label
                    ],
                    "recall": score["recall_" + label],
                    "f1": score["f1_" + label],
                }

                for field in (
                    TIMING_FIELDS + TOKEN_FIELDS
                ):
                    row["mean_" + field] = average(
                        subset, field
                    )
                    row["n_" + field] = sum(
                        r.get(field) not in (None, "")
                        for r in subset
                    )

                label_rows.append(row)

    write_csv(
        args.out / "label_summary.csv",
        label_rows, list(label_rows[0])
    )

    questions = []

    for qid in sorted(gold):
        for fw in FRAMEWORKS:
            group = [
                r for r in cases
                if r["query_id"] == qid
                and r["framework"] == fw
            ]

            questions.append({
                "query_id": qid,
                "framework": fw,
                "gold_label": gold[qid],
                "correct": sum(
                    r["correct"] for r in group
                ),
                "classified": sum(
                    r["classified"] for r in group
                ),
                "missing": sum(
                    r["status"] == "missing"
                    for r in group
                ),
                "unsupported": sum(
                    r["status"] == "unsupported"
                    for r in group
                ),
                "compile_error": sum(
                    r["status"] == "compile_error"
                    for r in group
                ),
                "other_failures": sum(
                    r["status"] not in {
                        "ok", "missing",
                        "unsupported", "compile_error"
                    }
                    for r in group
                ),
            })

    write_csv(
        args.out / "question_summary.csv",
        questions, list(questions[0])
    )

    manifest = {
        "queries": str(args.queries),
        "gold": str(args.gold),
        "runs": str(args.runs),
        "n_questions": len(gold),
        "models": MODELS,
        "frameworks": FRAMEWORKS,
        "source_file_counts": dict(counts),
        "expected_cases": len(cases),
        "selected_runs": sum(
            bool(r["run_id"]) for r in cases
        ),
        "missing_cases": sum(
            r["status"] == "missing"
            for r in cases
        ),
        "selection_policy": (
            "latest attempt whose query_input.json "
            "matches the current benchmark"
        ),
        "macro_f1_end_to_end": (
            "unclassified cases count as false negatives "
            "for their gold class"
        ),
    }

    with (
        args.out / "evaluation_manifest.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Read {counts['result_files']} result.json files"
    )
    print(
        f"Benchmark: {len(gold)} questions x "
        f"{len(MODELS)} models x 2 frameworks "
        f"= {len(cases)} cases"
    )
    print(
        "Selected:",
        sum(bool(r["run_id"]) for r in cases)
    )
    print(
        "Missing:",
        sum(r["status"] == "missing" for r in cases)
    )
    print(
        "Outdated query attempts excluded:",
        counts["outdated_query_text"]
    )

    print(
        "\nFramework | Correct/All | Classified/All "
        "| End-to-end Acc | Supported Acc | Macro-F1"
    )

    for row in by_framework:
        print(
            f"{row['framework']:9} | "
            f"{row['correct']:3}/{row['cases']:<3} | "
            f"{row['classified']:3}/{row['cases']:<3} | "
            f"{row['end_to_end_accuracy']:.3f} | "
            f"{str(row['supported_accuracy']):>13} | "
            f"{row['macro_f1_end_to_end']:.3f}"
        )

    print(f"\nSaved evaluation to: {args.out}")


if __name__ == "__main__":
    main()
