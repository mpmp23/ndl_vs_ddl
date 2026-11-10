import argparse
import csv
import hashlib
import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
API_URL = "https://openrouter.ai/api/v1/chat/completions"
CSV_COLUMNS = [
    "batch_id", "run_id", "section_id", "framework", "model_key",
    "requested_model", "returned_model", "status", "started_at", "finished_at",
    "llm_seconds", "total_seconds", "prompt_tokens", "completion_tokens",
    "total_tokens", "reasoning_tokens", "cost", "prompt_sha256",
    "source_sha256", "request_sha256", "output_dir", "error",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_path_part(text):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    if not value:
        raise ValueError(f"Invalid path component: {text!r}")
    return value


def load_sections(path):

    root = ET.parse(path).getroot()

    if root.tag != "Paragraphs":
        raise ValueError("Expected <Paragraphs> root")

    sections = []
    seen_ids = set()

    for paragraph in root.findall("Paragraph"):

        section_id = paragraph.get("paragraphLabel")
        relevant_text = paragraph.find("RelevantText")

        if not section_id or relevant_text is None:
            raise ValueError("Missing paragraph label or RelevantText")

        if section_id in seen_ids:
            raise ValueError(f"Duplicate section: {section_id}")

        # Ensure gold-standard rules are not accidentally supplied
        if paragraph.find("Rules") is not None:
            raise ValueError(f"Gold rules present in {section_id}")

        if paragraph.find("SuperiorityRelations") is not None:
            raise ValueError(f"Gold priorities present in {section_id}")

        snippet = "".join(relevant_text.itertext()).strip()

        if not snippet:
            raise ValueError(f"Empty snippet: {section_id}")

        seen_ids.add(section_id)
        sections.append((section_id, snippet))

    if not sections:
        raise ValueError("No contract snippets found")

    return sections


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("models"), dict):
        raise ValueError("models.yaml must contain a models mapping")
    models = config["models"]
    for framework in ("ddl", "neod"):
        roster = config.get(f"{framework}_models", [])
        if not isinstance(roster, list) or any(m not in models for m in roster):
            raise ValueError(f"Invalid {framework}_models list in models.yaml")
        if len(roster) != len(set(roster)):
            raise ValueError(f"Duplicate models in {framework}_models")
    options = config.get("request_options", {})
    if not isinstance(options, dict):
        raise ValueError("request_options must be a mapping")
    return config


def build_prompt(framework, section_id, snippet):
    path = ROOT / "prompts" / "encode" / f"{framework}.txt"
    template = path.read_text(encoding="utf-8")
    for placeholder in ("{{SECTION_ID}}", "{{SNIPPET}}"):
        if template.count(placeholder) != 1:
            raise ValueError(f"Expected exactly one {placeholder} in {path}")
    return (template.replace("{{SECTION_ID}}", section_id)
                    .replace("{{SNIPPET}}", snippet))


def call_openrouter(payload, api_key):
    request = Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=240) as response:
            data = json.load(response)
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter connection error: {exc}") from exc
    return data, time.perf_counter() - start


def get_text_content(response):
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices")
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter returned no text content")
    return content


def validate_xml(text, framework, section_id):
    # done to fix issue of added ``` xml` output by models
    cleaned = text.strip()

    if cleaned.startswith("```xml"):
        cleaned = cleaned[len("```xml"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    root = ET.fromstring(cleaned)  # raises ParseError; never silently repairs output
    if root.tag != "Paragraph":
        raise ValueError(f"Expected Paragraph root, got {root.tag}")
    if root.get("paragraphLabel") != section_id:
        raise ValueError(
            f"Paragraph label {root.get('paragraphLabel')!r} != {section_id!r}"
        )
    rules = root.find("Rules")
    if rules is None or not rules.findall("Rule"):
        raise ValueError("No <Rules><Rule> entries found")
    for rule in rules.findall("Rule"):
        if not rule.get("ruleLabel"):
            raise ValueError("Rule missing ruleLabel")
        if framework == "neod":
            expression = rule.findtext("NeoDavidsonian")
        else:
            expression = rule.findtext("Simple") or rule.text
        if not expression or not expression.strip():
            raise ValueError(f"Empty rule: {rule.get('ruleLabel')}")
    return len(rules.findall("Rule"))


def existing_success(task_dir, request_hash):
    if not task_dir.exists():
        return False
    for path in task_dir.glob("*/metadata.json"):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if info.get("status") == "ok" and info.get("request_sha256") == request_hash:
            return True
    return False


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if header_needed:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CSV_COLUMNS})


def execute_task(batch_dir, framework, model_key, model_id, options,
                 section_id, snippet, api_key, overwrite=False, dry_run=False):
    prompt = build_prompt(framework, section_id, snippet)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        **options,
    }
    if payload["model"] != model_id or payload["messages"] != [{"role": "user", "content": prompt}]:
        raise ValueError("request_options must not override model or messages")
    request_hash = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    task_dir = (batch_dir / framework / safe_path_part(model_key)
                / safe_path_part(section_id))
    task_name = f"{section_id} / {framework} / {model_key}"
    if not overwrite and existing_success(task_dir, request_hash):
        print(f"SKIP {task_name} (matching completed encoding)")
        return
    if dry_run:
        print(f"DRY  {task_name} -> {model_id}")
        return

    run_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
              + "_" + uuid.uuid4().hex[:8])
    run_dir = task_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "source.txt").write_text(snippet, encoding="utf-8")

    started_at = utc_now()
    total_start = time.perf_counter()
    metadata = {
        "batch_id": batch_dir.name,
        "run_id": run_id,
        "section_id": section_id,
        "framework": framework,
        "model_key": model_key,
        "requested_model": model_id,
        "returned_model": None,
        "status": "api_error",
        "started_at": started_at,
        "finished_at": None,
        "llm_seconds": None,
        "total_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
        "prompt_sha256": sha256(prompt),
        "source_sha256": sha256(snippet),
        "request_sha256": request_hash,
        "request_options": options,
        "output_dir": str(run_dir),
        "error": None,
    }

    try:
        response, llm_seconds = call_openrouter(payload, api_key)
        metadata["llm_seconds"] = round(llm_seconds, 6)
        (run_dir / "raw_response.json").write_text(
            json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        metadata["returned_model"] = response.get("model")
        metadata["response_id"] = response.get("id")
        metadata["provider"] = response.get("provider")
        usage = response.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        metadata.update({
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens", usage.get("reasoning_tokens")),
            "cost": usage.get("cost"),
        })
        content = get_text_content(response)
        (run_dir / "raw_output.txt").write_text(content, encoding="utf-8")
        try:
            metadata["rule_count"] = validate_xml(content, framework, section_id)
            (run_dir / "rules.xml").write_text(content, encoding="utf-8")
            metadata["status"] = "ok"
        except (ET.ParseError, ValueError) as exc:
            metadata["status"] = "invalid_xml"
            metadata["error"] = str(exc)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        # No retry and no automatic output refinement.
        metadata["error"] = str(exc)
        (run_dir / "error.txt").write_text(str(exc), encoding="utf-8")
    finally:
        metadata["finished_at"] = utc_now()
        metadata["total_seconds"] = round(time.perf_counter() - total_start, 6)
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        append_csv(batch_dir / "results.csv", metadata)
        print(f"{metadata['status'].upper():11} {task_name} "
              f"({metadata['llm_seconds']}s, {metadata['total_tokens']} tokens)")
        if metadata["error"]:
            print(f"  Error: {metadata['error'][:240]}")


def main():
    parser = argparse.ArgumentParser(description="Generate DDL/NeoD XML rule encodings")
    parser.add_argument("--snippets", type=Path, default=ROOT / "snippets.xml")
    parser.add_argument("--config", type=Path, default=ROOT / "models.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "encodings")
    parser.add_argument("--batch-id", default="batch_001")
    parser.add_argument("--section", action="append", help="Select a section ID; repeatable")
    parser.add_argument("--framework", choices=["ddl", "neod"], action="append")
    parser.add_argument("--model", action="append", help="Model key from models.yaml; repeatable")
    parser.add_argument("--all", action="store_true", help="Allow every selected task")
    parser.add_argument("--overwrite", action="store_true", help="Rerun successful tasks too")
    parser.add_argument("--dry-run", action="store_true", help="Show planned calls, no API use")
    args = parser.parse_args()

    config = load_config(args.config)
    sections = load_sections(args.snippets)
    if args.section:
        wanted = set(args.section)
        available = {section_id for section_id, _ in sections}
        if not wanted <= available:
            parser.error(f"Unknown sections: {sorted(wanted - available)}")
        sections = [(sid, text) for sid, text in sections if sid in wanted]
    elif len(sections) > 1 and not args.all:
        parser.error("Multiple sections found; pass --section ID or --all")

    if not args.all and not args.model:
        parser.error("Use --model KEY for a selected model, or --all for a batch")
    frameworks = args.framework or ["ddl", "neod"]
    selected = []
    for framework in frameworks:
        roster = config.get(f"{framework}_models", [])
        if framework == "ddl" and len(roster) != 3:
            parser.error("Select exactly 3 keys in ddl_models in models.yaml before running DDL")
        models = [key for key in roster if not args.model or key in args.model]
        for key in models:
            selected.append((framework, key, config["models"][key]))
    if not selected:
        parser.error("No models selected; check --framework, --model, and model rosters")
    if args.model:
        selected_keys = {key for _, key, _ in selected}
        if set(args.model) - selected_keys:
            parser.error(f"Selected models not eligible for chosen frameworks: {sorted(set(args.model)-selected_keys)}")

    batch_dir = args.out / safe_path_part(args.batch_id)
    print(f"Batch {args.batch_id}: {len(sections)} sections x configured model/framework tasks")
    print(f"Planned calls: {len(sections) * len(selected)}")

    api_key = None
    if not args.dry_run:
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            parser.error("OPENROUTER_API_KEY missing. Put it in .env (never commit that file).")

    for section_id, snippet in sections:
        for framework, model_key, model_id in selected:
            # One independent call per snippet/framework/model, no cross-snippet history.
            execute_task(
                batch_dir, framework, model_key, model_id,
                config.get("request_options", {}).get(model_key, {}),
                section_id, snippet, api_key,
                overwrite=args.overwrite, dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()