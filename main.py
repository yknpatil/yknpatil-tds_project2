import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from scipy.stats import linregress
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins and methods (GET, POST, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow requests from any origin
    allow_methods=["*"],       # allow all HTTP methods
    allow_headers=["*"],       # allow all headers
)

AIPIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDE3MjlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.nZjYM7jpIm_XJKggJW2A3m7b5JOU0_Dx00UyrigmFOE"  # Use environment var ideally
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"

async def aipipe_call(prompt: str):
    url = "https://api.aipipe.com/v1/llm/generate"
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "max_tokens": 1000
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp_json = await resp.json()
            return resp_json.get("text", "")

async def scrape_wikipedia_table(url: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    df = tables[0]
    return df

def clean_film_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda x: x.strip())
    df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'[\$,]', '', regex=True).astype(float)
    df['Year'] = df['Year'].astype(int)
    if 'Rank' in df.columns:
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    if 'Peak' in df.columns:
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
    return df

def answer_questions(df: pd.DataFrame):
    answers = []
    count_2bn_pre2000 = df[(df['Worldwide gross'] >= 2e9) & (df['Year'] < 2000)].shape[0]
    answers.append(count_2bn_pre2000)

    df_15bn = df[df['Worldwide gross'] > 1.5e9]
    if not df_15bn.empty:
        earliest_film = df_15bn.sort_values('Year').iloc[0]
        answers.append(earliest_film['Title'])
    else:
        answers.append("No films grossed over $1.5 billion.")

    if 'Rank' in df.columns and 'Peak' in df.columns:
        corr = df[['Rank', 'Peak']].dropna().corr().iloc[0,1]
        answers.append(round(corr, 6))
    else:
        answers.append(None)

    plot_uri = plot_rank_peak(df)
    answers.append(plot_uri)

    return answers

def plot_rank_peak(df: pd.DataFrame):
    if 'Rank' not in df.columns or 'Peak' not in df.columns:
        return "Insufficient data to plot."

    df_clean = df[['Rank', 'Peak']].dropna()
    x = df_clean['Rank']
    y = df_clean['Peak']
    slope, intercept, _, _, _ = linregress(x, y)

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, label='Data points')
    plt.plot(x, intercept + slope*x, 'r--', label='Regression line')
    plt.xlabel('Rank')
    plt.ylabel('Peak')
    plt.title('Scatterplot of Rank vs Peak with Regression Line')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()

    b64_str = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64_str}"

    if len(data_uri) > 100000:
        return "Image size exceeds limit."
    return data_uri

@app.post("/api")
async def api_handler():
    df_raw = await scrape_wikipedia_table(WIKI_URL)
    df = clean_film_data(df_raw)
    answers = answer_questions(df)
    return JSONResponse(content=answers)

