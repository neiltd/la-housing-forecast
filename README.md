# Forecasting LA Home Values

**[Open the interactive neighborhood map →](https://neiltd.github.io/la-housing-forecast/)**

Final project for MGMTMSA 437 Forecasting and Time Series Analysis, UCLA Anderson MSBA (Fall 2026, Dr. William Yu).

**Team:** Yang Liu, Jinwoo Roh, Yiu-Hon Lee, Thanapol Doungsaeng

## The question

LA home values rose 5.6% a year from 2000 to 2026, then stalled after mortgage rates went from 2.7% to 7.6%. We asked which forecasting method predicts LA home values best, whether mortgage rates and local jobs improve the forecast, whether a chat LLM can forecast as well as a statistical model, and which neighborhoods the rate shock hit hardest.

## What we found

- **Simple models beat complex ones.** Across 14 models in a rolling test (18 forecast origins, 1 to 12 months ahead), ARIMAX with mortgage rates had the lowest error (2.67%), just ahead of ARIMA (2.77%). LSTM, GRU, XGBoost, LightGBM and Amazon's Chronos foundation model never beat them.
- **Chat LLMs only won when they could recognize the period.** Claude Opus 5.5, GPT-5.5 and Gemini 3.1 Pro averaged 1.4% to 2.1% error when they could tell the place and dates, better than any model. With place, dates and rates hidden, Claude matched ARIMAX (2.67%), GPT scored 2.74% and Gemini 3.00%. All test dates fall inside their training data, so the first numbers mostly measure memory.
- **One test window can crown the wrong model.** LightGBM won a single 80/20 split but ranked tenth across the 18 rolling windows.
- **Rates move prices slowly; local jobs add nothing.** In a VAR, a 1-point rise in the mortgage rate lowers LA home values about 4.5% after one year and 9.2% after two. ARIMAX with LA and Orange County payroll jobs placed third (2.81%), and the jobs signal disappears without the 2020–21 COVID swings.
- **Use the long history.** Training on all data since 2000 forecast better than starting in 2008, 2012 or 2016.
- **The market is split.** In LA County, pricier neighborhoods fell hardest in 2023 (a tested cross-section result). Over the last year coastal neighborhoods led again while central LA fell. 220 of 496 neighborhoods are rising, 216 flat and 60 falling.
- **Outlook:** about +4.8% for the LA metro by August 2027, with an 80% range of −6% to +17%.

## What's here

| Path | Contents |
| --- | --- |
| `index.html` | Interactive map of 496 LA and Orange County neighborhoods (the website) |
| `report/LA_home_values_report.pdf` | Written report (20 pages) |
| `slides/` | Presentation slides, PDF and editable PowerPoint |
| `analysis/la_home_value_forecasting.ipynb` | Full analysis notebook: data, models, both tests, LLM scoring, neighborhoods, outlook |
| `analysis/data/` | Zillow ZHVI (metro and LA-metro neighborhoods), FRED mortgage rate, BLS payroll jobs, neighborhood map positions |
| `analysis/llm_prompts.py`, `analysis/llm_run_api.py` | Build the 162 LLM prompts, send them to the GPT and Gemini APIs, score the answers |
| `analysis/llm/` | Every LLM prompt, every raw answer, and the scored forecasts |

## Run the analysis

```bash
cd analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt jupyter
.venv/bin/jupyter notebook la_home_value_forecasting.ipynb
```

The notebook runs top to bottom in about 10 minutes (the first run also downloads the Chronos model) and writes every chart and results table to `analysis/figures/`. It only scores the saved LLM answers, so it needs no API keys.

To collect the LLM forecasts again, put `GEMINI_API_KEY` and `OPENAI_API_KEY` in `analysis/.env` (ignored by git) and run:

```bash
python llm_prompts.py build
python llm_run_api.py run openai gpt-5.5
python llm_run_api.py run gemini gemini-3.1-pro-preview 5
# Claude, with its tools turned off so it cannot run code:
for f in llm/prompts/*.txt; do claude -p --model claude-opus-5-5 --tools "" < "$f" > "llm/answers/claude-opus-5.5_$(basename "$f")"; done
python llm_prompts.py score
```

## How we used AI tools

- **Claude Code (Claude Opus 5.5)** was our main coding assistant: it wrote and debugged the notebook, ran the models, drew the charts, and built the slides and the map. We reviewed the code, reran everything end to end, and checked every reported number against the notebook output.
- **Chronos-Bolt (Amazon)** is one of the 14 forecasting models, used zero-shot.
- **Claude, GPT-5.5 and Gemini 3.1 Pro** were asked directly for forecasts and scored like any other model.

## Data sources

- [Zillow Home Value Index](https://www.zillow.com/research/data/): all homes, mid tier, smoothed and seasonally adjusted; metro and neighborhood level, January 2000 to August 2026.
- [FRED 30-Year Fixed Rate Mortgage Average (MORTGAGE30US)](https://fred.stlouisfed.org/series/MORTGAGE30US).
- BLS payroll jobs, total nonfarm, via FRED: [LA County (SMU06310840000000001)](https://fred.stlouisfed.org/series/SMU06310840000000001) and [Orange County (SMU06112440000000001)](https://fred.stlouisfed.org/series/SMU06112440000000001).
- Neighborhood map positions geocoded from [OpenStreetMap](https://www.openstreetmap.org/copyright) (Nominatim). Map tiles by Esri.
