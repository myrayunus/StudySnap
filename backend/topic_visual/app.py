import base64
import json
import os
import random

import boto3

MODEL_ID = "amazon.nova-canvas-v1:0"
REGION = os.environ.get("BEDROCK_REGION", "ap-southeast-1")

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
    "Content-Type": "application/json",
}

PROMPT_TEMPLATE = (
    "Create an adorable, cute illustration representing the main topic and key "
    "concepts from these lecture notes: {topic}. Use a kawaii or chibi art style "
    "with soft pastel colors, friendly characters, and charming details. Include "
    "cute mascot-like characters or animals related to the subject matter, with "
    "big eyes and cheerful expressions. The overall feel should be warm, playful, "
    "and delightful while still being educational and relevant to the topic."
)


def _response(status, body):
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


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

    topic = body.get("topic") or body.get("text") or body.get("lecture_notes") or "a study topic"
    topic = str(topic)[:800]

    prompt = PROMPT_TEMPLATE.format(topic=topic)

    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt[:1024]},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 720,
            "width": 1280,
            "cfgScale": 8.0,
            "seed": random.randint(0, 858993459),
        },
    }

    try:
        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
        )
        payload = json.loads(response["body"].read())
        images = payload.get("images", [])
        if not images:
            return _response(502, {"error": "No image returned from Bedrock."})
        return _response(200, {"image_base64": images[0], "mime": "image/png"})
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": str(exc)})
