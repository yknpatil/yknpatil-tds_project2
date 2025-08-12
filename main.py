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
import numpy as np

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
OPENAI_API_KEY = os.getenv("AIPIPE_TOKEN", "YOUR_API_TOKEN_HERE") # Set as env var in prod

# ======================
# Utilities
# ======================
def is_json_string(s: str) -> bool:
    """Checks if a string is a valid JSON object."""
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
    Calls the AIPIPE LLM API and returns the generated text.
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
    """Wrapper to make an LLM call with a timeout."""
    try:
        return await asyncio.wait_for(aipipe_call(prompt), timeout=timeout)
    except asyncio.TimeoutError:
        raise RuntimeError("LLM call timed out")
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

async def llm_extract_fast(context: str, question: str, timeout: int = 10) -> str:
    """A fast LLM call for a quick answer, with a shorter timeout."""
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
    """Fetches the content of a URL."""
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "data-analyst-agent/1.0"})
    r.raise_for_status()
    return r.text

def parse_html_table_to_df(html: str) -> List[pd.DataFrame]:
    """Parses all HTML tables from a string into a list of pandas DataFrames."""
    soup = BeautifulSoup(html, "lxml")
    tables = [pd.read_html(str(t))[0] for t in soup.find_all("table")]
    return tables

# ======================
# Plot helper
# ======================
def fig_to_b64(fig, max_bytes: int = 100_000, fmt: str = "png") -> str:
    """
    Converts a matplotlib figure to a base64 encoded string.
    Compresses if the initial size exceeds max_bytes.
    """
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format=fmt)
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return f"data:image/{fmt};base64," + base64.b64encode(data).decode("ascii")

    # Try compress to webp if initial size is too large
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
    """
    Uses the LLM to parse user instructions into a structured plan.
    """
    prompt = f"""
Parse the following instructions and return a JSON object with keys:
- "intent": a string representing the user's goal ("data_analysis" or "text_extraction").
- "url": the URL to scrape (if any).
- "questions": a list of strings, each being a question to answer.
- "plot_specs": a dictionary for plotting instructions (if any), e.g., {{"type": "scatterplot", "x": "Rank", "y": "Peak", "regression_line": True}}.
- "response_type": "array" or "object".

Instructions:
{raw_text}
"""
    try:
        parsed = await llm_call(prompt)
        return json.loads(parsed) if is_json_string(parsed) else {"raw": raw_text}
    except:
        return {"raw": raw_text}

async def fetch_and_prepare(plan: Dict[str, Any], attachments: Dict[str, UploadFile]) -> Dict[str, Any]:
    """
    Fetches data from a URL and/or loads attached files into a context dictionary.
    """
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
            # Parse tables from the URL's HTML
            context["dataframes_from_html"] = parse_html_table_to_df(html)
        except Exception as e:
            print(f"Failed to parse HTML tables: {e}")
            pass
    return context

async def process_data_and_answer(plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """
    Performs data analysis and plotting based on the plan and context.
    """
    questions = plan.get("questions", [])
    plot_specs = plan.get("plot_specs", {})
    response_type = plan.get("response_type", "array")
    
    if "dataframes_from_html" not in context or not context["dataframes_from_html"]:
        raise ValueError("No dataframes found from URL.")
    
    # Find the most relevant table, e.g., the one containing 'Gross' in the column names
    df = None
    for table in context["dataframes_from_html"]:
        if any("Gross" in col for col in table.columns):
            df = table
            break
    if df is None:
        raise ValueError("Relevant table not found on the page.")

    # Clean and standardize column names
    df.columns = [col.replace('[A]', '').replace(' (Adjusted for inflation)[3]', '').strip() for col in df.columns]

    # Convert necessary columns to numeric types for analysis
    try:
        df['Gross'] = df['Worldwide gross'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
        df['Year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
        df['Rank'] = df['Rank'].astype(int)
        df['Peak'] = df['Peak'].astype(int)
    except KeyError as e:
        raise ValueError(f"Required column not found: {e}")
    except Exception as e:
        raise ValueError(f"Failed to convert column to numeric type: {e}")

    answers = []
    plot_base64 = None
    
    # Question 1: How many movies made over $2 billion before the year 2000?
    billion_2_movies = df[(df['Gross'] > 2e9) & (df['Year'] < 2000)]
    answers.append({"question": questions[0], "answer": f"{len(billion_2_movies)} movies."})
    
    # Question 2: What is the oldest movie that made over $1.5 billion?
    billion_1_5_movies = df[df['Gross'] > 1.5e9].sort_values('Year')
    earliest_film = billion_1_5_movies.iloc[0]['Film'] if not billion_1_5_movies.empty else "N/A"
    answers.append({"question": questions[1], "answer": earliest_film})

    # Question 3: What is the correlation between Rank and Peak?
    if 'Rank' in df.columns and 'Peak' in df.columns:
        correlation = df['Rank'].corr(df['Peak'])
        answers.append({"question": questions[2], "answer": f"{correlation:.4f}"})
    else:
        answers.append({"question": questions[2], "answer": "Columns 'Rank' or 'Peak' not found."})

    # Question 4: Plotting based on specs
    if plot_specs and plot_specs.get("type") == "scatterplot":
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['Rank'], df['Peak'])
        
        # Calculate and plot a regression line if requested
        if plot_specs.get("regression_line") and len(df) > 1:
            m, c = np.polyfit(df['Rank'], df['Peak'], 1)
            ax.plot(df['Rank'], m * df['Rank'] + c, color='red', linestyle='dotted')
        
        ax.set_xlabel('Rank')
        ax.set_ylabel('Peak')
        ax.set_title('Scatterplot of Rank and Peak')
        
        plot_base64 = fig_to_b64(fig)
        plt.close(fig) # Free up memory
        
    answers.append({"question": "Plot", "answer": plot_base64})
    
    return answers

async def answer_with_llm(plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """
    Uses the LLM to generate a response for general text extraction.
    """
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
@app.post("/api/")
async def handle_request(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    image_file: Optional[UploadFile] = File(None, alias="image.png"),
    data_file: Optional[UploadFile] = File(None, alias="data.csv")
):
    """
    Main API endpoint to handle user requests with a required questions.txt file
    and optional other files.
    """
    print(f"Received file: {questions_file.filename}")
    content = await questions_file.read()
    questions_text = content.decode("utf-8", errors="ignore")
    print(f"questions.txt content length: {len(questions_text)}")
    print(f"questions.txt content preview:\n{questions_text[:100]}")

    if not questions_text:
        raise HTTPException(status_code=400, detail="questions.txt required")

    attachments = {}
    if image_file:
        attachments["image.png"] = image_file
        print(f"Received image: {image_file.filename}")
    if data_file:
        attachments["data.csv"] = data_file
        print(f"Received data: {data_file.filename}")
    
    try:
        plan = await interpret_instructions(questions_text)
        print(f"Parsed plan: {plan}")
        
        context = await fetch_and_prepare(plan, attachments)
        
        # Route the request based on the LLM's interpretation of the intent
        if plan.get("intent") == "data_analysis":
            result = await process_data_and_answer(plan, context)
            return JSONResponse(result)
        else:
            # For other intents, use the LLM to generate a response
            answer = await answer_with_llm(plan, context)
            return JSONResponse(answer)
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        
        # Fallback to a faster LLM extraction if the primary pipeline fails
        context_text = context.get("html", "")[:4000] if 'context' in locals() else ""
        fast = await llm_extract_fast(context_text, questions_text)
        if fast and is_json_string(fast):
            return JSONResponse(json.loads(fast))
        
        # Final error response
        return JSONResponse({"detail": "Sorry, I cannot find the answer."})
