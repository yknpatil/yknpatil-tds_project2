import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from scipy.stats import linregress
import json

AIPIPE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDE3MjlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.nZjYM7jpIm_XJKggJW2A3m7b5JOU0_Dx00UyrigmFOE"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"

# Async function to call the aipipe LLM API (stub)
async def aipipe_call(prompt: str):
    url = "https://api.aipipe.com/v1/llm/generate"  # example endpoint, adjust accordingly
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
            return resp_json["text"]  # Adjust depending on actual response format

# Scrape the Wikipedia table into a pandas DataFrame
async def scrape_wikipedia_table(url: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    # Assume the first table contains the data we want
    df = tables[0]
    return df

# Clean and preprocess the film data
def clean_film_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda x: x.strip())
    # Remove $ and commas from 'Worldwide gross' and convert to float
    df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'[\$,]', '', regex=True).astype(float)
    # Convert Year to int
    df['Year'] = df['Year'].astype(int)
    # Convert Rank and Peak to numeric if they exist
    if 'Rank' in df.columns:
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
    if 'Peak' in df.columns:
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
    return df

# Answer the given questions using the cleaned DataFrame
def answer_questions(df: pd.DataFrame):
    answers = []

    # 1. How many $2bn+ movies were released before 2000?
    count_2bn_pre2000 = df[(df['Worldwide gross'] >= 2e9) & (df['Year'] < 2000)].shape[0]
    answers.append(f"There are {count_2bn_pre2000} movies grossing over $2 billion released before 2000.")

    # 2. Earliest film grossing over $1.5bn
    df_15bn = df[df['Worldwide gross'] > 1.5e9]
    if not df_15bn.empty:
        earliest_film = df_15bn.sort_values('Year').iloc[0]
        answers.append(f"The earliest film to gross over $1.5 billion is \"{earliest_film['Title']}\" released in {earliest_film['Year']}.")
    else:
        answers.append("No films grossed over $1.5 billion.")

    # 3. Correlation between Rank and Peak
    if 'Rank' in df.columns and 'Peak' in df.columns:
        corr = df[['Rank', 'Peak']].dropna().corr().iloc[0,1]
        answers.append(f"The correlation between Rank and Peak is {corr:.3f}.")
    else:
        answers.append("Rank or Peak data is missing.")

    return answers

# Generate scatterplot of Rank vs Peak with a dotted red regression line
def plot_rank_peak(df: pd.DataFrame):
    if 'Rank' not in df.columns or 'Peak' not in df.columns:
        return "Insufficient data to plot."

    df_clean = df[['Rank', 'Peak']].dropna()
    x = df_clean['Rank']
    y = df_clean['Peak']
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

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

    # Encode plot image as base64 data URI
    b64_str = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64_str}"
    # Ensure under 100,000 bytes limit (adjust compression if needed)
    if len(data_uri) > 100000:
        return "Image size exceeds limit."
    return data_uri

# Main orchestrator function
async def main():
    # 1. Scrape Wikipedia for film data
    df_raw = await scrape_wikipedia_table(WIKI_URL)
    df = clean_film_data(df_raw)

    # 2. Use aipipe LLM for question parsing / code gen (optional)
    # For demo, answering directly with local code:
    answers = answer_questions(df)

    # 3. Generate plot and get data URI
    plot_uri = plot_rank_peak(df)

    # 4. Return answers + plot as JSON array
    output = answers + [plot_uri]
    print(json.dumps(output, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
