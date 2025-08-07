import os
import io
import json
import base64
import re
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Enable CORS for all origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

async def call_aipipe(prompt: str) -> str:
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 1000
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    detail = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"AIPipe API error: {detail}")
                resp_json = await resp.json()
                return resp_json.get("text", "").strip()
        except aiohttp.ClientError as e:
            raise HTTPException(status_code=500, detail=f"AIPipe connection error: {str(e)}")

def load_file_to_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        elif filename.endswith('.json'):
            data = json.loads(content.decode())
            return pd.json_normalize(data)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file extension: {filename}")
    except Exception as e:
        raise ValueError(f"Failed to load file {filename}: {e}")

def summarize_dataframe(df: pd.DataFrame) -> str:
    summary = f"Columns: {', '.join(df.columns)}\n"
    summary += f"Data Types:\n{df.dtypes.to_string()}\n"
    summary += f"Description:\n{df.describe(include='all').to_string()}"
    return summary

def plot_scatter_with_regression(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    df_clean = df[[x_col, y_col]].dropna()
    if df_clean.empty:
        return "No data to plot."

    slope, intercept, _, _, _ = linregress(df_clean[x_col], df_clean[y_col])

    plt.figure(figsize=(6, 4))
    plt.scatter(df_clean[x_col], df_clean[y_col], label='Data points')
    plt.plot(df_clean[x_col], intercept + slope * df_clean[x_col], 'r--', label='Regression line')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatterplot of {x_col} vs {y_col} with Regression Line')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)

    b64_img = base64.b64encode(buf.read()).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64_img}"
    if len(data_uri) > 100000:
        return "Image size exceeds limit."
    return data_uri

@app.post("/api")
async def analyze(
    questions: UploadFile = File(...),
    files: list[UploadFile] = File(default=[])
):
    question_text = (await questions.read()).decode()
    question_lines = [line.strip() for line in question_text.splitlines() if line.strip()]
    if not question_lines:
        raise HTTPException(status_code=400, detail="questions.txt is empty")

    dataframes = []
    for file in files:
        content = await file.read()
        try:
            df = load_file_to_dataframe(file.filename, content)
            dataframes.append(df)
        except Exception:
            # skip files that can't be loaded
            continue

    df = dataframes[0] if dataframes else pd.DataFrame()

    answers = []
    for question in question_lines:
        q_lower = question.lower()
        if "scatterplot" in q_lower and ("regression" in q_lower or "regression line" in q_lower):
            cols = re.findall(r'scatterplot.*of ([\w]+) and ([\w]+)', q_lower)
            if cols:
                x_col, y_col = cols[0]
                if x_col in df.columns and y_col in df.columns:
                    answer = plot_scatter_with_regression(df, x_col, y_col)
                else:
                    answer = f"Columns {x_col} or {y_col} not found in data."
            else:
                if 'Rank' in df.columns and 'Peak' in df.columns:
                    answer = plot_scatter_with_regression(df, 'Rank', 'Peak')
                else:
                    answer = "Could not parse columns for scatterplot."
        else:
            if df.empty:
                prompt = f"Answer this question without any data:\n{question}"
            else:
                summary = summarize_dataframe(df)
                prompt = f"Data summary:\n{summary}\n\nQuestion:\n{question}\nAnswer:"
            try:
                answer = await call_aipipe(prompt)
            except Exception as e:
                answer = f"Error getting answer from AIPipe: {str(e)}"
        answers.append(answer)

    return JSONResponse(content=answers)





