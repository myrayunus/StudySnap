import base64
import json
import os

import boto3

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-5")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,x-api-key",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT = """Generate 5 multiple-choice quiz questions based on the lecture notes provided by the student (supplied as text and/or an attached document on the first turn).

Do not reveal the correct answers yet. You are a quiz master. Present one question at a time. Start by presenting Question 1 with 4 options (A, B, C, D) and wait for the user's answer before proceeding. After the user answers, tell them if they are correct or not, briefly explain the right answer, then move on to the next question. After all 5 questions, show a final score out of 5. Then ask the user to rate their confidence level on this topic from 1 to 5, where 1 means very unsure and 5 means very confident. Based on their rating, provide personalised study tips and recommendations."""


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


def build_first_user_content(body):
    notes_text = body.get("text") or body.get("lecture_notes") or body.get("topic") or ""
    content = []

    file_data = body.get("file_data")
    if file_data:
        content.append(file_content_block(file_data, body.get("file_mime"), body.get("file_name")))

    parts = []
    if notes_text:
        parts.append("Lecture notes:\n\n" + notes_text)
    parts.append(body.get("message") or "start")
    content.append({"type": "text", "text": "\n\n".join(parts)})
    return content


def build_messages(body):
    """Assemble messages from history + the new user message.

    Body: { topic, level, history: [{role, content}], message, file_data?, file_mime? }
    The lecture notes (text and/or file) are attached to the FIRST user turn.
    """
    history = body.get("history") or []
    messages = []

    if not history:
        messages.append({"role": "user", "content": build_first_user_content(body)})
        return messages

    for turn in history:
        role = turn.get("role")
        text = turn.get("content", "")
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    new_message = body.get("message")
    if new_message:
        messages.append({"role": "user", "content": [{"type": "text", "text": new_message}]})

    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})

    return messages


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
