# Forecasting LA Home Values

**[Open the interactive neighborhood map →](https://neiltd.github.io/la-housing-forecast/)**

Final project for MGMTMSA 437 Forecasting and Time Series Analysis, UCLA Anderson MSBA (Fall 2026, Dr. William Yu).

**Team:** Yang Liu, Jinwoo Roh, Yiu-Hon Lee, Thanapol Doungsaeng

## The question

LA home values rose 5.6% a year from 2000 to 2026, then stalled after mortgage rates went from 2.7% to 7.6%. We asked which forecasting method predicts LA home values best, whether mortgage rates improve the forecast, and which neighborhoods the rate shock hit hardest.

## What we found

- **Simple models beat complex ones.** In a rolling test (18 forecast origins, 1 to 12 months ahead), ARIMAX with mortgage rates had the lowest error (2.67%), just ahead of ARIMA (2.77%). LSTM, GRU, XGBoost and LightGBM never beat them.
- **One test window can crown the wrong model.** LightGBM won a single 80/20 split but ranked eighth across the 18 rolling windows.
- **Rates matter slowly.** Their effect on LA home values shows up about two years later. A naive drift forecast is as good as anything a year out.
- **The market is split.** In LA County, pricier neighborhoods fell hardest in 2023. Over the last year coastal neighborhoods led again (Beverly Hills, Santa Monica, coastal Orange County) while central LA fell (MacArthur Park, Koreatown, Downtown, Westlake). 220 of 496 neighborhoods are rising, 216 flat and 60 falling.
- **Outlook:** about +4.8% for the LA metro by August 2027, with an 80% range of −6% to +17%.

## What's here

| Path | Contents |
| --- | --- |
| `index.html` | Interactive map of 496 LA and Orange County neighborhoods (the website) |
| `report/LA_home_values_report.pdf` | Written report (19 pages) |
| `slides/` | Presentation slides, PDF and editable PowerPoint |
| `analysis/la_home_value_forecasting.ipynb` | Full analysis notebook: data, models, both tests, neighborhoods, outlook |
| `analysis/data/` | Zillow ZHVI (metro and LA-metro neighborhoods), FRED mortgage rate, neighborhood map positions |

## Run the analysis

```bash
cd analysis
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt jupyter
.venv/bin/jupyter notebook la_home_value_forecasting.ipynb
```

The notebook runs top to bottom in about 4 minutes and writes every chart and results table to `analysis/figures/`.

## Data sources

- [Zillow Home Value Index](https://www.zillow.com/research/data/): all homes, mid tier, smoothed and seasonally adjusted; metro and neighborhood level, January 2000 to August 2026.
- [FRED 30-Year Fixed Rate Mortgage Average (MORTGAGE30US)](https://fred.stlouisfed.org/series/MORTGAGE30US).
- Neighborhood map positions geocoded from [OpenStreetMap](https://www.openstreetmap.org/copyright) (Nominatim). Map tiles by Esri.
