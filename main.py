import os
import io
import json
import time
import base64
import asyncio
import traceback
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import aiohttp

# ======================
# FastAPI + CORS config
# ======================
app = FastAPI(title="Data Analyst Agent - Generic")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ======================
# LLM Credentials
# ======================
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "YOUR_API_TOKEN_HERE")  # set as env var in prod

# ======================
# Utilities
# ======================
def is_json_string(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False

# ======================
# AIPIPE API Wrapper
# ======================
async def aipipe_call(prompt: str, max_tokens: int = 1000) -> str:
    """
    Calls AIPIPE LLM API and returns the generated text.
    """
    url = "https://api.aipipe.com/v1/llm/generate"
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            resp_json = await resp.json()
            return resp_json.get("text", "")

# ======================
# LLM calls using AIPIPE
# ======================
async def llm_call(prompt: str, system: Optional[str] = None, timeout: int = 60) -> str:
    try:
        return await asyncio.wait_for(aipipe_call(prompt), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError("LLM call timed out")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

async def llm_extract_fast(context: str, question: str, timeout: int = 10) -> str:
    prompt = f"""
You have {timeout} seconds. Extract the answer to the following question from the given context.
Return only valid JSON (array or object) with no explanations.

Question:
{question}

Context:
{context[:4000]}
"""
    try:
        return await asyncio.wait_for(aipipe_call(prompt, max_tokens=300), timeout=timeout)
    except asyncio.TimeoutError:
        return ""
    except Exception:
        return ""

# ======================
# Scraping helpers
# ======================
def fetch_url(url: str, timeout: int = 15) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "data-analyst-agent/1.0"})
    r.raise_for_status()
    return r.text

def parse_html_table_to_df(html: str) -> List[pd.DataFrame]:
    soup = BeautifulSoup(html, "lxml")
    tables = [pd.read_html(str(t))[0] for t in soup.find_all("table")]
    return tables

# ======================
# Plot helper
# ======================
def fig_to_b64(fig, max_bytes: int = 100_000, fmt: str = "png") -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format=fmt)
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return f"data:image/{fmt};base64," + base64.b64encode(data).decode("ascii")

    # Try compress to webp
    img = Image.open(io.BytesIO(data)).convert("RGB")
    width, height = img.size
    quality = 80
    while True:
        buf2 = io.BytesIO()
        img.resize((width, height), Image.LANCZOS).save(buf2, format="WEBP", quality=quality)
        if len(buf2.getvalue()) <= max_bytes or quality <= 20:
            return "data:image/webp;base64," + base64.b64encode(buf2.getvalue()).decode("ascii")
        quality -= 15
        width = int(width * 0.9)
        height = int(height * 0.9)

# ======================
# Core pipeline functions
# ======================
async def interpret_instructions(raw_text: str) -> Dict[str, Any]:
    prompt = f"""
Parse the following instructions and return JSON with:
intent, url (if any), questions (list), response_type ("array" or "object"), plot_specs (if any).

Instructions:
{raw_text}
"""
    try:
        parsed = await llm_call(prompt)
        return json.loads(parsed) if is_json_string(parsed) else {"raw": raw_text}
    except:
        return {"raw": raw_text}

async def fetch_and_prepare(plan: Dict[str, Any], attachments: Dict[str, UploadFile]) -> Dict[str, Any]:
    context = {"attachments": {}}
    for name, up in attachments.items():
        content = await up.read()
        context["attachments"][name] = content
        if name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content))
                context.setdefault("dataframes", {})[name] = df
            except:
                pass
    if plan.get("url"):
        html = fetch_url(plan["url"])
        context["html"] = html
        try:
            context["dataframes_from_html"] = parse_html_table_to_df(html)
        except:
            pass
    return context

async def answer_with_llm(plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
    prompt = json.dumps({
        "plan": plan,
        "context_keys": list(context.keys()),
        "instructions": "Produce EXACT JSON output only."
    })
    result = await llm_call(prompt)
    if not is_json_string(result):
        raise ValueError("Invalid JSON from LLM")
    return json.loads(result)

# ======================
# API Endpoint
# ======================
app = FastAPI()

@app.post("/api/")
async def handle_request(request: Request):
    form = await request.form()
    print("Received form keys:", list(form.keys()))
    questions_text = None
    attachments = {}

    for key, file in form.multi_items():
        if isinstance(file, UploadFile):
            if key == "questions.txt":  # form field name to find questions.txt
                content = await file.read()
                questions_text = content.decode("utf-8", errors="ignore")
            else:
                attachments[file.filename] = file

    if not questions_text:
        raise HTTPException(400, "questions.txt required")

    # Process questions_text and attachments here
    try:
        plan = await interpret_instructions(questions_text)
        context = await fetch_and_prepare(plan, attachments)
        answer = await answer_with_llm(plan, context)
        return JSONResponse(answer)
    except Exception:
        context_text = context.get("html", "")[:4000]
        fast = await llm_extract_fast(context_text, questions_text)
        if fast and is_json_string(fast):
            return JSONResponse(json.loads(fast))
        return JSONResponse("Sorry I cannot find the answer")

