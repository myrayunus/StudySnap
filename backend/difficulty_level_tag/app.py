import json
import os

import boto3
from flask import Flask, Response, request, stream_with_context

app = Flask(__name__)

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-5")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

SYSTEM_PROMPT = """You are a helpful study assistant. A student has uploaded the following lecture notes.

Analyse the key concepts in these notes. Tag each main concept with a difficulty level using this format:

\U0001f7e2 EASY \u2014 Concept name: One sentence explaining why it is straightforward.
\U0001f7e1 MEDIUM \u2014 Concept name: One sentence explaining why it needs some attention.
\U0001f534 HARD \u2014 Concept name: One sentence explaining why it is challenging.

Tag at least 5 concepts. End with a short tip on where the student should focus most of their study time."""


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


def build_messages(body):
    notes_text = body.get("text") or body.get("lecture_notes") or ""
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

    prompt_text = (
        "Here are the lecture notes:\n\n" + notes_text
        if notes_text
        else "Here are the lecture notes (see the attached file)."
    )
    content.append({"type": "text", "text": prompt_text})

    return [{"role": "user", "content": content}]


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
