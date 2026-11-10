
import os
import re
import json
import time
import uuid
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from dotenv import load_dotenv


# -----------------------------------
# 1. CONFIGURATION
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

MODEL = "openai/gpt-6-astra"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT_FILES = {
    "ddl": "ddl_query.txt",
    "neod": "neod_query.txt"
}


# -----------------------------------
# 2. LOAD PROMPT
# -----------------------------------

def load_prompt(framework):

    filename = PROMPT_FILES[framework]

    path = BASE_DIR / "prompts" / "verify" / filename

    if not path.exists():
        path = BASE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {filename}"
        )

    return path.read_text(encoding="utf-8")


# -----------------------------------
# 3. INSERT QUERY INTO PROMPT
# -----------------------------------

def build_prompt(
    framework,
    query_id,
    rules_xml,
    situation,
    hypothesis
):

    template = load_prompt(framework)

    rule_placeholder = (
        "{{DDL_RULES_XML}}"
        if framework == "ddl"
        else "{{NEOD_RULES_XML}}"
    )

    values = {
        "{{QUERY_ID}}": query_id,
        rule_placeholder: rules_xml,
        "{{SITUATION}}": situation,
        "{{HYPOTHESIS}}": hypothesis
    }

    for placeholder, value in values.items():

        if placeholder not in template:
            raise ValueError(
                f"Missing prompt placeholder: {placeholder}"
            )

        template = template.replace(
            placeholder,
            str(value)
        )

    return template


# -----------------------------------
# 4. CALL GPT-6 ASTRA
# -----------------------------------

def call_llm(prompt):

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found in .env"
        )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 16384,
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    try:
        with urlopen(request, timeout=120) as response:
            data = json.load(response)

    except HTTPError as error:
        details = error.read().decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(
            f"OpenRouter HTTP {error.code}: {details}"
        ) from error

    llm_seconds = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc).isoformat()

    content = data["choices"][0]["message"]["content"]

    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty or invalid content")

    usage = data.get("usage") or {}

    token_details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )

    return {
        "xml": content.strip(),
        "started_at": started_at,
        "finished_at": finished_at,
        "llm_seconds": llm_seconds,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": token_details.get(
            "reasoning_tokens",
            usage.get("reasoning_tokens")
        ),
        "response_id": data.get("id"),
        "response_model": data.get("model")
    }


# -----------------------------------
# 5. VALIDATE XML STRUCTURE
# -----------------------------------

def validate_xml(xml_text, framework, query_id):

    root = ET.fromstring(xml_text)

    if root.tag != "Query":
        raise ValueError("Expected <Query> root")

    if root.get("queryId") != query_id:
        raise ValueError("Query ID mismatch")

    status = root.get("status")

    if status not in ("ok", "unsupported"):
        raise ValueError("Invalid query status")

    if status == "unsupported":

        reason = root.findtext("Reason")

        if not reason or not reason.strip():
            raise ValueError("Missing unsupported reason")

        return status, reason.strip()

    situation = root.find("Situation")
    hypothesis = root.find("Hypothesis")

    if situation is None or hypothesis is None:
        raise ValueError(
            "Missing Situation or Hypothesis"
        )

    if framework == "ddl":

        if not situation.findall("Fact"):
            raise ValueError("Missing DDL facts")

        if not hypothesis.text or not hypothesis.text.strip():
            raise ValueError("Missing DDL hypothesis")

    else:

        situation_expr = situation.findtext(
            "NeoDavidsonian"
        )

        hypothesis_expr = hypothesis.findtext(
            "NeoDavidsonian"
        )

        if not situation_expr or not situation_expr.strip():
            raise ValueError("Missing NeoD situation")

        if not hypothesis_expr or not hypothesis_expr.strip():
            raise ValueError("Missing NeoD hypothesis")

    return status, None


# -----------------------------------
# 6. TRANSLATE QUERY
# -----------------------------------

def translate_query(
    framework,
    query_id,
    rules_xml,
    situation,
    hypothesis,
    run_id=None,
    output_dir=None
):

    framework = framework.lower()

    if framework not in PROMPT_FILES:
        raise ValueError(
            "Framework must be 'ddl' or 'neod'"
        )

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    # Prepare the full prompt
    prompt = build_prompt(
        framework,
        query_id,
        rules_xml,
        situation,
        hypothesis
    )

    # Save artifacts in a unique run folder
    safe_run_id = re.sub(
        r"[^A-Za-z0-9_.-]", "_", run_id
    )

    safe_query_id = re.sub(
        r"[^A-Za-z0-9_.-]", "_", query_id
    )

    output_dir = (
        Path(output_dir) if output_dir is not None
        else BASE_DIR / "runs" / safe_run_id / framework
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save exact prompt
    (output_dir / f"{safe_query_id}_prompt.txt").write_text(
        prompt,
        encoding="utf-8"
    )

    # One LLM call, no retries
    response = call_llm(prompt)

    xml_text = response["xml"]

    # Save raw translation
    xml_path = (
        output_dir / f"{safe_query_id}_query.xml"
    )

    xml_path.write_text(
        xml_text,
        encoding="utf-8"
    )

    # Check XML format locally
    status, reason = validate_xml(
        xml_text,
        framework,
        query_id
    )

    # Prepare metadata
    metadata = {
        "run_id": run_id,
        "query_id": query_id,
        "framework": framework,
        "model": MODEL,
        "response_model": response["response_model"],
        "response_id": response["response_id"],
        "status": status,
        "reason": reason,
        "started_at": response["started_at"],
        "finished_at": response["finished_at"],
        "llm_seconds": response["llm_seconds"],
        "prompt_tokens": response["prompt_tokens"],
        "completion_tokens": response["completion_tokens"],
        "total_tokens": response["total_tokens"],
        "reasoning_tokens": response["reasoning_tokens"],
        "xml_path": str(xml_path)
    }

    metadata_path = (
        output_dir / f"{safe_query_id}_translation.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Return everything needed by run.py
    return {
        **metadata,
        "xml": xml_text
    }