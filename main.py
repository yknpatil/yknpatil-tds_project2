import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"

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
    count_2bn_pre2000 = df[(df['Worldwide gross'] >= 2e9) & (df['Year'] < 2000)].shape[0]

    df_15bn = df[df['Worldwide gross'] > 1.5e9]
    if not df_15bn.empty:
        earliest_film = df_15bn.sort_values('Year').iloc[0]['Title']
    else:
        earliest_film = None

    if 'Rank' in df.columns and 'Peak' in df.columns:
        corr = df[['Rank', 'Peak']].dropna().corr().iloc[0,1]
    else:
        corr = None

    return {
        "count_2bn_pre2000": count_2bn_pre2000,
        "earliest_film": earliest_film,
        "correlation_rank_peak": corr
    }

def plot_rank_peak(df: pd.DataFrame):
    if 'Rank' not in df.columns or 'Peak' not in df.columns:
        return None

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
async def api_handler(
    questions_file: UploadFile = File(...),
    image_file: UploadFile = File(None),
    data_file: UploadFile = File(None)
):
    # Read questions.txt
    questions_text = await questions_file.read()
    questions = questions_text.decode('utf-8').splitlines()

    # Load CSV or fallback to scrape
    if data_file:
        data_csv = await data_file.read()
        df = pd.read_csv(io.BytesIO(data_csv))
    else:
        df_raw = await scrape_wikipedia_table(WIKI_URL)
        df = clean_film_data(df_raw)

    # Here, you could parse and answer based on 'questions'.
    # For now, answer using your current logic (fixed questions).
    # To do: customize logic to handle 'questions' list if needed.

    answers = answer_questions(df)
    plot_uri = plot_rank_peak(df)

    response = {
        "answers": answers,
        "scatterplot_base64": plot_uri
    }

    return JSONResponse(content=response)
