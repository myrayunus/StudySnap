import base64
import json
import os

import boto3

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-5")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = """You are a helpful study assistant. A student has provided the following raw lecture notes.

Based on these notes, generate exactly 5 flashcards covering the key concepts. Format each flashcard like this:

\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
\U0001f0cf FLASHCARD 1 of 5
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
\U0001f4d6 HOW TO USE: Read the question, think of your answer, then scroll down to check!

\u2753 QUESTION:
Write the question here

\u2b07\ufe0f Think of your answer first, then reveal below...
\u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7 \u00b7
\U0001f4a1 ANSWER:
Write the answer here
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501

Repeat this format for all 5 flashcards, numbering them 1 of 5 through 5 of 5. Replace all placeholder text with real content from the lecture notes. Cover a broad range of key topics. Keep answers clear and concise."""


def _response(status, body):
    return {"statusCode": status, "headers": CORS_HEADERS, "body": json.dumps(body)}


IMAGE_MEDIA = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"}
TEXT_EXTS = {"txt", "md", "markdown", "csv", "tsv", "json", "html", "htm", "rtf", "log", "xml", "yaml", "yml"}


def _ext(file_name):
    if file_name and "." in file_name:
        return file_name.rsplit(".", 1)[1].lower()
    return ""


def file_content_block(file_data, file_mime, file_name):
    mime = (file_mime or "").lower()
    ext = _ext(file_name)

    if mime in IMAGE_MEDIA or mime.startswith("image/") or ext in {"png", "jpg", "jpeg", "gif", "webp"}:
        media = mime if mime.startswith("image/") else "image/" + ("jpeg" if ext in {"jpg", "jpeg"} else ext or "png")
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": file_data}}

    if mime == "application/pdf" or ext == "pdf":
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": file_data}}

    if ext in TEXT_EXTS or mime.startswith("text/") or mime in {"application/json", "application/xml"}:
        try:
            decoded = base64.b64decode(file_data).decode("utf-8", errors="replace")
            return {"type": "text", "text": "Uploaded file contents:\n\n" + decoded}
        except Exception:  # noqa: BLE001
            pass

    media = mime or "application/pdf"
    return {"type": "document", "source": {"type": "base64", "media_type": media, "data": file_data}}


def build_messages(body):
    notes_text = body.get("text") or body.get("lecture_notes") or ""
    content = []

    file_data = body.get("file_data")
    if file_data:
        content.append(file_content_block(file_data, body.get("file_mime"), body.get("file_name")))

    prompt_text = (
        "Here are the lecture notes:\n\n" + notes_text
        if notes_text
        else "Here are the lecture notes (see the attached file)."
    )
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def invoke(messages):
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0.7,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    response = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(request_body))
    payload = json.loads(response["body"].read())
    parts = [b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text"]
    return "".join(parts)


def handler(event, context):
    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or "POST"
    )
    if method == "OPTIONS":
        return _response(200, {})

    raw_body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw_body = base64.b64decode(raw_body).decode("utf-8")
    try:
        body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except json.JSONDecodeError:
        body = {}

    try:
        text = invoke(build_messages(body))
        return _response(200, {"text": text})
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": str(exc)})
