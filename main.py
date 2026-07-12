import base64
import json
import re

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import config

app = FastAPI()

# Allow the grader to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}


@app.get("/")
async def root():
    return {"status": "ok", "email": config.EMAIL}


def detect_mime(image_bytes):
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"

    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"

    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"

    return "image/png"


def clean_answer(ans):
    ans = str(ans).strip()

    ans = ans.strip('"')
    ans = ans.strip("'")

    ans = re.sub(r"^[₹$€£]\s*", "", ans)

    ans = ans.replace(",", "")

    if re.fullmatch(r"-?\d+\.0+", ans):
        ans = str(int(float(ans)))

    return ans


@app.post("/answer-image")
async def answer_image(request: Request):

    body = await request.json()

    image_b64 = body["image_base64"]
    question = body["question"]

    image_bytes = base64.b64decode(image_b64)

    mime = detect_mime(image_bytes)

    prompt = f"""
Answer the user's question using ONLY the image.

Question:
{question}

Rules:

- Read every visible value exactly.
- Never estimate.
- If arithmetic is needed, calculate carefully.
- Numeric answers:
    - digits only
    - decimal point only if needed
    - no commas
    - no currency symbols
    - no units
- Text answers:
    - copy exactly

Return ONLY valid JSON.

{{"answer":"..."}}
"""

    payload = {
        "model": config.VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1000,
        "response_format": {
            "type": "json_object"
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            f"{config.AIPIPE_BASE}/chat/completions",
            headers=HEADERS,
            json=payload,
        )

        response.raise_for_status()

        text = response.json()["choices"][0]["message"]["content"]

    try:
        answer = json.loads(text)["answer"]
    except Exception:

        m = re.search(r'"answer"\s*:\s*"([^"]+)"', text)

        if m:
            answer = m.group(1)
        else:
            answer = text

    return {
        "answer": clean_answer(answer)
    }