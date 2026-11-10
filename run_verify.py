"""Verify natural-language legal queries against saved DDL and NeoD encodings.

Sources:
  baselines/horner_ddl/<model>/*.xml
      Horner XML: <Paragraphs><Paragraph><Generated><Paragraph>...<Rules>...
  outputs/encodings/<batch>/<framework>/<model>/<section>/<run_id>/rules.xml
      Newly generated XML: <Paragraph>...<Rules>...

Each query in queries.xml names its RelevantSections, Situation, and Hypothesis.
Every case uses the SAME query translator (GPT-6 Astra), but a different
saved encoding. Original predicate names and rule expressions are not repaired.

Examples:
  python run_verify.py --list-encodings
  python run_verify.py --query-id Q001 --model gpt6_astra --framework neod
  python run_verify.py --query-id Q001 --models gpt6_astra claude_sonnet_55 gemini4_argon
  python run_verify.py --all                     # 3 newer models; all queries
  python run_verify.py --all-models --all        # 9 models; all queries
  python run_verify.py --query-id Q001 --model gpt6_astra --dry-run

Requires: z3-solver, python-dotenv, the existing compilers and translate_query.py.
Never sends an API request with --dry-run or --list-encodings.
"""

import argparse
import csv
import hashlib
import json
import re
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from compile_ddl import compile_ddl
from compile_neod import compile_neod
from translate_query import MODEL as QUERY_MODEL, translate_query
from verify import save_verification, verify_compiled


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs" / "verification"

HISTORICAL_MODELS = (
    "claude_sonnet_45", "deepseek_v32", "gemini_pro_3_preview",
    "gpt_oss_120b", "gpt52", "kimi_k2_thinking",
)
NEW_MODELS = ("gpt6_astra", "claude_sonnet_55", "gemini4_argon")
ALL_MODELS = HISTORICAL_MODELS + NEW_MODELS
FRAMEWORKS = ("ddl", "neod")
EXPECTED_QUERY_MODEL = "openai/gpt-6-astra"

CSV_COLUMNS = [
    "run_id", "pair_id", "query_id", "encoding_model", "encoding_source",
    "framework", "query_model", "sections", "status", "prediction", "error",
    "started_at", "finished_at", "llm_seconds", "compile_seconds",
    "z3_seconds", "total_seconds", "prompt_tokens", "completion_tokens",
    "total_tokens", "reasoning_tokens", "response_model", "response_id",
    "rules_sha256", "sources_json", "run_dir",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not name:
        raise ValueError(f"Unsafe empty identifier: {value!r}")
    return name


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(record):
    path = RUNS_DIR / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if header:
            writer.writeheader()
        writer.writerow(record)


def load_queries(path):
    root = ET.parse(path).getroot()
    if root.tag == "Query":
        elements = [root]
    elif root.tag == "Queries":
        elements = root.findall("Query")
    else:
        raise ValueError("queries.xml must have <Queries> or <Query> as its root")
    if not elements:
        raise ValueError("No queries found")

    queries = []
    seen = set()
    for element in elements:
        qid = (element.get("queryId") or "").strip()
        situation = (element.findtext("Situation") or "").strip()
        hypothesis = (element.findtext("Hypothesis") or "").strip()
        sections = [
            (section.text or "").strip()
            for section in element.findall("RelevantSections/Section")
        ]
        if not qid or not situation or not hypothesis:
            raise ValueError("Every query requires queryId, Situation and Hypothesis")
        if not sections or any(not section for section in sections):
            raise ValueError(f"Query {qid}: missing <RelevantSections><Section> entries")
        if len(sections) != len(set(sections)):
            raise ValueError(f"Query {qid}: duplicate relevant section")
        if qid in seen:
            raise ValueError(f"Duplicate query ID: {qid}")
        seen.add(qid)
        queries.append({"query_id": qid, "situation": situation,
                        "hypothesis": hypothesis, "sections": sections})
    return queries


def extract_horner_paragraph(path, section_id):
    """Select generated rules from a Horner evaluation XML, NOT source/gold text."""
    root = ET.parse(path).getroot()
    if root.tag != "Paragraphs":
        raise ValueError(f"Baseline root must be <Paragraphs>: {path}")
    matches = [p for p in root.findall("Paragraph")
               if p.get("paragraphLabel") == section_id]
    if len(matches) != 1:
        raise ValueError(f"Section {section_id}: expected one baseline paragraph, got {len(matches)}")
    generated = matches[0].find("Generated/Paragraph")
    if generated is None:
        raise ValueError(f"No generated model rules for {section_id} in {path}")
    actual_label = generated.get("paragraphLabel")

    known_mismatch = (
        path.parent.name == "gpt_oss_120b"
        and section_id in {
            "8.2.1.a.ii",
            "8.2.1.a.iii",
            "8.2.1.a.iv.D",
        }
        and actual_label == "8.2.1"
    )

    if actual_label != section_id and not known_mismatch:
        raise ValueError(
            f"Unexpected generated paragraph label {actual_label!r} "
            f"for {section_id} in {path}"
        )
    return generated


def extract_new_paragraph(path, section_id):
    root = ET.parse(path).getroot()
    if root.tag != "Paragraph" or root.get("paragraphLabel") != section_id:
        raise ValueError(f"Unexpected generated paragraph label/root: {path}")
    return root


def check_rules(paragraph, framework, section_id):
    rules_container = paragraph.find("Rules")
    rules = rules_container.findall("Rule") if rules_container is not None else []
    if not rules:
        raise ValueError(f"{section_id}: no generated <Rule> elements")
    for rule in rules:
        label = rule.get("ruleLabel", "(unlabeled)")
        expr = (rule.findtext("NeoDavidsonian") if framework == "neod"
                else rule.findtext("Simple") or rule.text)
        if not expr or not expr.strip():
            raise ValueError(f"{section_id}: missing {framework} expression in {label}")
    return len(rules)


def get_baseline_file(baselines_dir, model):
    folder = baselines_dir / model
    files = sorted(folder.glob("*.xml"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one baseline XML in {folder}; found {len(files)}")
    return files[0]


def get_new_file(batch_dir, framework, model, section_id):
    # Horner's Gemini 3 Pro is retired. The NeoD replacement may be stored
    # under a new key; this is a CROSS-VERSION comparison, not a matched model.
    keys = [model]
    if framework == "neod" and model == "gemini_pro_3_preview":
        keys.append("gemini_pro_3.1_preview")

    candidates = []
    for key in keys:
        folder = batch_dir / framework / key / safe_name(section_id)
        for meta_path in sorted(folder.glob("*/metadata.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("status") != "ok":
                continue
            if (meta.get("section_id") != section_id or
                    meta.get("framework") != framework or
                    meta.get("model_key") != key):
                continue
            rules_path = meta_path.parent / "rules.xml"
            if rules_path.is_file():
                candidates.append((meta.get("finished_at") or "", rules_path))
    if not candidates:
        folders = ", ".join(str(batch_dir / framework / key / section_id) for key in keys)
        raise FileNotFoundError(f"No successful encoding for {section_id}: {folders}")
    # Deterministic, reported choice when --overwrite produced multiple runs.
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    if len(candidates) > 1:
        print(f"    NOTE: {len(candidates)} successful encodings of {section_id}; using newest")
    return candidates[-1][1]


def load_selected_rules(model, framework, sections, baselines_dir, batch_dir):
    """Return a <Paragraphs> rules document and a complete provenance manifest.

    Every requested section must be present. No predicate names or expressions
    are harmonized or repaired. XML is reserialized only to combine fragments.
    """
    source = "horner_ddl" if framework == "ddl" and model in HISTORICAL_MODELS else batch_dir.name
    combined = ET.Element("Paragraphs")
    manifest = []
    for section_id in sections:
        if framework == "ddl" and model in HISTORICAL_MODELS:
            path = get_baseline_file(baselines_dir, model)
            paragraph = extract_horner_paragraph(path, section_id)
        else:
            path = get_new_file(batch_dir, framework, model, section_id)
            paragraph = extract_new_paragraph(path, section_id)
        rule_count = check_rules(paragraph, framework, section_id)
        # ET.tostring preserves the text/meaning of rules, though it may change
        # whitespace, XML escaping, or CDATA representation.
        paragraph_copy = ET.fromstring(
            ET.tostring(paragraph, encoding="utf-8")
        )

        original_label = paragraph_copy.get("paragraphLabel")

        if source == "horner_ddl" and original_label != section_id:
            print(
                f"WARNING: Generated label {original_label!r} "
                f"mapped to source section {section_id!r}"
            )
            paragraph_copy.set("paragraphLabel", section_id)

        combined.append(paragraph_copy)
        entry = {
            "section_id": section_id,
            "source": source,
            "file": str(path.resolve()),
            "file_sha256": digest(path.read_bytes()),
            "rule_count": rule_count,
        }
        if source == "horner_ddl":
            entry["original_generated_label"] = original_label
            entry["selected_section_label"] = section_id
            entry["label_mismatch"] = original_label != section_id
        if source != "horner_ddl":
            metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
            entry["generated_model_key"] = metadata.get("model_key")
            entry["generated_model_id"] = metadata.get("requested_model")
            entry["generation_run_id"] = metadata.get("run_id")
            if (model == "gemini_pro_3_preview" and framework == "neod" and
                    "gemini-3.1" in str(entry["generated_model_id"])):
                entry["comparison_warning"] = (
                    "Horner DDL uses Gemini 3 Pro; this NeoD encoding uses Gemini 3.1 Pro. "
                    "These are not same-model paired results."
                )
                print(f"    WARNING: {entry['comparison_warning']}")
        manifest.append(entry)
    return ET.tostring(combined, encoding="unicode"), source, manifest


def run_case(query, model, framework, baselines_dir, batch_dir, pair_id):
    """One query + one encoding model + one framework; never retries an LLM call."""
    run_id = f"{pair_id}_{framework}"
    run_dir = RUNS_DIR / run_id / framework
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "run_id": run_id, "pair_id": pair_id, "query_id": query["query_id"],
        "encoding_model": model, "framework": framework,
        "encoding_source": "horner_ddl" if framework == "ddl" and model in HISTORICAL_MODELS else batch_dir.name,
        "query_model": QUERY_MODEL, "sections": ";".join(query["sections"]),
        "status": "error", "prediction": None, "error": None,
        "started_at": now(), "run_dir": str(run_dir),
    }
    save_json(run_dir / "query_input.json", query)
    start = time.perf_counter()
    stage = "load_error"

    try:
        rules_xml, source, manifest = load_selected_rules(
            model, framework, query["sections"], baselines_dir, batch_dir
        )
        record["encoding_source"] = source
        record["rules_sha256"] = digest(rules_xml.encode("utf-8"))
        record["sources_json"] = json.dumps(manifest, ensure_ascii=False)
        save_json(run_dir / "sources.json", manifest)
        (run_dir / "selected_rules.xml").write_text(rules_xml, encoding="utf-8")

        stage = "translation_error"
        translation = translate_query(
            framework=framework,
            query_id=query["query_id"],
            rules_xml=rules_xml,
            situation=query["situation"],
            hypothesis=query["hypothesis"],
            run_id=run_id,
            output_dir=run_dir,
        )
        for key in ("llm_seconds", "prompt_tokens", "completion_tokens",
                    "total_tokens", "reasoning_tokens", "response_model", "response_id"):
            record[key] = translation.get(key)
        if translation["status"] == "unsupported":
            record["status"] = "unsupported"
            record["error"] = translation.get("reason")
        else:
            stage = "compile_error"
            compile_start = time.perf_counter()
            try:
                compiled = (compile_ddl(rules_xml, translation["xml"])
                            if framework == "ddl"
                            else compile_neod(rules_xml, translation["xml"]))
            finally:
                record["compile_seconds"] = round(time.perf_counter() - compile_start, 6)
            save_json(run_dir / "compiled_z3.json", {
                "rules": [str(rule) for rule in compiled.rules],
                "situation": [str(fact) for fact in compiled.situation],
                "hypothesis": str(compiled.hypothesis),
                "vocabulary": sorted(compiled.vocabulary),
            })

            stage = "z3_error"
            verification = verify_compiled(compiled)
            record["prediction"] = verification.prediction
            record["z3_seconds"] = verification.z3_seconds
            record["status"] = ("inconsistent" if verification.prediction == "Inconsistent"
                                else "ok")
            save_verification(verification, run_dir / "verification.json",
                              run_id=run_id, query_id=query["query_id"], framework=framework)
            for check_name, smt2 in verification.solver_inputs.items():
                (run_dir / f"{check_name}.smt2").write_text(smt2, encoding="utf-8")
    except Exception as exc:
        record["status"] = stage
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["total_seconds"] = round(time.perf_counter() - start, 6)
        record["finished_at"] = now()
        save_json(run_dir / "result.json", record)
        append_csv(record)

    print(f"{record['status'].upper():17} {query['query_id']} / {framework} / {model}"
          f" -> {record['prediction'] or '-'}")
    if record["error"]:
        print(f"  {record['error'][:300]}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Verify LexFlow queries using stored model encodings")
    parser.add_argument("--queries", type=Path, default=ROOT / "queries.xml")
    parser.add_argument("--query-id", help="Run one query ID (default: all queries)")
    parser.add_argument("--model", action="append", help="Encoding model key; repeatable")
    parser.add_argument("--models", nargs="+", help="Encoding model keys")
    parser.add_argument("--framework", action="append", choices=FRAMEWORKS,
                        help="ddl or neod; repeatable (default: both)")
    parser.add_argument("--all", action="store_true", help="Run every selected query/model/framework")
    parser.add_argument("--all-models", action="store_true", help="Use all nine encoding models")
    parser.add_argument("--batch-id", default="batch_001")
    parser.add_argument("--encodings", type=Path, default=ROOT / "outputs" / "encodings")
    parser.add_argument("--baselines", type=Path, default=ROOT / "baselines" / "horner_ddl")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source availability without API calls")
    parser.add_argument("--list-encodings", action="store_true", help="List saved encoding coverage (no queries needed)")
    args = parser.parse_args()

    batch_dir = args.encodings / safe_name(args.batch_id)
    models = list(dict.fromkeys((args.model or []) + (args.models or [])))
    if not models:
        models = list(ALL_MODELS if args.all_models else NEW_MODELS)
    invalid = [key for key in models if key not in ALL_MODELS]
    if invalid:
        parser.error(f"Unknown model keys: {invalid}")
    frameworks = list(dict.fromkeys(args.framework or FRAMEWORKS))

    if args.list_encodings:
        snippets = ROOT / "snippets.xml"
        if not snippets.is_file():
            parser.error(f"Required for listing: {snippets}")
        root = ET.parse(snippets).getroot()
        section_ids = [p.get("paragraphLabel") for p in root.findall("Paragraph")]
        for model in models:
            for framework in frameworks:
                available = 0
                for section_id in section_ids:
                    try:
                        load_selected_rules(model, framework, [section_id], args.baselines, batch_dir)
                        available += 1
                    except (OSError, ValueError, ET.ParseError):
                        pass
                print(f"{model:24} {framework:4} {available}/{len(section_ids)} snippets")
        return

    if QUERY_MODEL != EXPECTED_QUERY_MODEL and not args.dry_run:
        parser.error(
            f"translate_query.py still uses {QUERY_MODEL}. Set MODEL = {EXPECTED_QUERY_MODEL!r} "
            "to use the agreed fixed query translator."
        )

    queries = load_queries(args.queries)
    if args.query_id:
        queries = [q for q in queries if q["query_id"] == args.query_id]
        if not queries:
            parser.error(f"Query not found: {args.query_id}")

    print(f"Verification: {len(queries)} queries × {len(models)} encoding models × "
          f"{len(frameworks)} frameworks = {len(queries) * len(models) * len(frameworks)} cases")
    print(f"Query translator: {EXPECTED_QUERY_MODEL}")

    for query in queries:
        for model in models:
            if args.dry_run:
                for framework in frameworks:
                    try:
                        rules_xml, source, manifest = load_selected_rules(
                            model, framework, query["sections"], args.baselines, batch_dir
                        )
                        print(f"READY   {query['query_id']} / {framework} / {model} "
                              f"({len(manifest)} snippets; {source}; {len(rules_xml)} chars)")
                    except Exception as exc:
                        print(f"MISSING {query['query_id']} / {framework} / {model}: {exc}")
                continue
            pair_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                       + "_" + uuid.uuid4().hex[:8] + "_" + safe_name(model)
                       + "_" + safe_name(query["query_id"]))
            for framework in frameworks:
                run_case(query, model, framework, args.baselines, batch_dir, pair_id)

    if not args.dry_run:
        print(f"\nResults saved to {RUNS_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()