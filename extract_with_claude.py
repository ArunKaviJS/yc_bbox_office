import os
import re
import json
import base64
import boto3
from collections import defaultdict
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
# ── AWS / Bedrock setup ──────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
REGION = os.getenv("REGION", "ap-south-1")

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)
CLAUDE_MODEL_ID = os.getenv("CLAUDE_MODEL_ID")   # change to whichever version is enabled in your region



# ── Bedrock OCR (replaces claude_ocr with direct anthropic client) ────────────
def claude_ocr_bedrock(file_path: str) -> str:
    """
    Send an image or PDF to Claude via AWS Bedrock and return the extracted text.
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    encoded_file = base64.b64encode(file_bytes).decode("utf-8")

    ext = file_path.lower()
    if ext.endswith(".pdf"):
        media_type = "application/pdf"
        content_block_type = "document"
    elif ext.endswith(".png"):
        media_type = "image/png"
        content_block_type = "image"
    elif ext.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
        content_block_type = "image"
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": content_block_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_file,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all text from this document exactly as written.",
                    },
                ],
            }
        ],
    }

    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# ── Extraction helper (replaces AzureLLMAgent call) ─────────────────────────
def call_claude_for_llm(prompt: str) -> str:
    """
    Send a plain text prompt to Claude via Bedrock and return the raw string reply.
    """
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = bedrock_client.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# ── Main extraction function ──────────────────────────────────────────────────
def extract_all_fields_and_tables(
    field_schema: List[Dict],
    content: str,
) -> dict:
    # ── Split schemas ──
    field_items = [f for f in field_schema if f["fieldType"] == "field"]
    table_items = [f for f in field_schema if f["fieldType"] == "table"]

    # ── Build field instructions ──
    field_text_lines = []
    for f in field_items:
        extra = (
            f"with {f['fieldName']} - carefully understand what is meant by "
            f"the description ({f['fieldDescription']}) for this field and return "
            f"the output in the type expected ({f['fieldDatatype']}) "
            "(e.g., string, number, date, etc.)."
        )
        field_text_lines.append(
            f"- **{f['fieldName']}** ({f['fieldDatatype']}): {f['fieldDescription']}\n  {extra}"
        )
    field_text = "\n".join(field_text_lines)

    # ── Build table instructions ──
    tables = defaultdict(list)
    for r in table_items:
        tables[r["tableName"]].append(r)

    table_text = ""
    for table_name, cols in tables.items():
        table_text += f"\n### Table: {table_name}\n"
        for c in cols:
            extra = (
                f"with {c['fieldName']} - carefully understand what is meant by "
                f"the description ({c['fieldDescription']}) for this column and return "
                f"the output in the type expected ({c['fieldDatatype']}) "
                "(e.g., string, number, date, etc.)."
            )
            table_text += (
                f"- **{c['fieldName']}** ({c['fieldDatatype']}): {c['fieldDescription']}\n  {extra}\n"
            )

    # ── Build prompt ──
    prompt = f"""
You are an extremely accurate information extraction model.

Extract ALL requested fields and ALL table rows across ALL pages.

========================================================
FIELDS TO EXTRACT (OUTPUT AS SIMPLE KEY: VALUE)
========================================================
{field_text}

========================================================
TABLES TO EXTRACT (OUTPUT AS ITEMS ARRAY)
========================================================
(Return format example:
"TableName": {{
    "fieldType": "table",
    "items": [
        {{ "col1": "...", "col2": "..." }},
        ...
    ]
}}
)

{table_text}

========================================================
STRICT RULES
========================================================
- Return value in EXACT datatype expected.
- If no value found → return null.
- Preserve field and table names EXACTLY.
- Do NOT rename or modify keys.
- Output ONLY valid JSON.
- No explanation. No markdown.

========================================================
CONTENT
========================================================
{content}

Return FINAL JSON ONLY:
""".strip()

    # ── Call Bedrock ──
    try:
        raw = call_claude_for_llm(prompt)
    except Exception as e:
        print("❌ Bedrock LLM failed:", e)
        return None

    # ── Parse JSON safely ──
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)
    except Exception as e:
        print("❌ JSON parsing error:", e)
        return {"raw": raw}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    file_path = "20260224063720449_1640.jpeg"          # change to your file

    print("🔍 Running OCR via Bedrock Claude...")
    ocr_text = claude_ocr_bedrock(file_path)
    print("OCR result:\n", ocr_text)

    print("\n📊 Extracting structured fields...")
    final_result = extract_all_fields_and_tables(fields, ocr_text)
    print("Final result:\n", json.dumps(final_result, indent=2))