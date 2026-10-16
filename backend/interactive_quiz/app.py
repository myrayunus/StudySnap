import json
import os

import boto3
from flask import Flask, Response, request, stream_with_context

app = Flask(__name__)

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-1")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

SYSTEM_PROMPT = """Generate 5 multiple-choice quiz questions based on the lecture notes provided by the student (supplied as text and/or an attached document on the first turn).

Do not reveal the correct answers yet. You are a quiz master. Present one question at a time. Start by presenting Question 1 with 4 options (A, B, C, D) and wait for the user's answer before proceeding. After the user answers, tell them if they are correct or not, briefly explain the right answer, then move on to the next question. After all 5 questions, show a final score out of 5. Then ask the user to rate their confidence level on this topic from 1 to 5, where 1 means very unsure and 5 means very confident. Based on their rating, provide personalised study tips and recommendations."""


def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }


def media_type_for(file_mime):
    if file_mime:
        return file_mime
    return "application/pdf"


def build_first_user_content(body):
    """Content block for the first user turn: lecture notes text and/or file + the message."""
    notes_text = body.get("text") or body.get("lecture_notes") or body.get("topic") or ""
    content = []

    file_data = body.get("file_data")
    file_mime = body.get("file_mime")
    if file_data:
        if file_mime and file_mime.startswith("image/"):
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": file_mime,
                        "data": file_data,
                    },
                }
            )
        else:
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": media_type_for(file_mime),
                        "data": file_data,
                    },
                }
            )

    parts = []
    if notes_text:
        parts.append("Lecture notes:\n\n" + notes_text)
    message = body.get("message") or "start"
    parts.append(message)
    content.append({"type": "text", "text": "\n\n".join(parts)})
    return content


def build_messages(body):
    """Assemble messages from the conversation history plus the new user message.

    Expected body: { topic, level, history: [{role, content}], message, file_data?, file_mime? }
    The lecture notes (text and/or file) are attached to the FIRST user turn so the
    model keeps them in context for the whole quiz.
    """
    history = body.get("history") or []
    messages = []

    if not history:
        # First turn: fold the lecture notes + file into the opening user message.
        messages.append({"role": "user", "content": build_first_user_content(body)})
        return messages

    # Subsequent turns: replay history, then append the new user message.
    for turn in history:
        role = turn.get("role")
        text = turn.get("content", "")
        if role not in ("user", "assistant"):
            continue
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    new_message = body.get("message")
    if new_message:
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": new_message}]}
        )

    # Bedrock requires the conversation to end on a user turn.
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})

    return messages


@app.route("/", methods=["OPTIONS"])
def options():
    return Response(status=200, headers=cors_headers())


@app.route("/", methods=["POST"])
def generate_route():
    body = request.get_json(force=True, silent=True) or {}
    messages = build_messages(body)

    def generate():
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.7,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                body=json.dumps(request_body),
            )
            for event in response["body"]:
                chunk = event.get("chunk")
                if not chunk:
                    continue
                payload = json.loads(chunk["bytes"].decode("utf-8"))
                if payload.get("type") == "content_block_delta":
                    text = payload.get("delta", {}).get("text", "")
                    if text:
                        yield text
        except Exception as exc:  # noqa: BLE001
            yield f"\n[Error: {exc}]"

    return Response(
        stream_with_context(generate()),
        content_type="text/plain; charset=utf-8",
        headers=cors_headers(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
