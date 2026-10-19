"""Build one LLM forecasting prompt per rolling-test origin, and score the answers.

The prompts give an LLM exactly what the statistical models see at each origin: the monthly home
value series and the 30-year mortgage rate up to that month, nothing after it.

Versions per origin:
  blind - no place, no dates, home values rebased so the first month = 100. The LLM cannot look up
          what actually happened, so this is the fair test.
  named - says "Los Angeles metro, Zillow ZHVI" with real dates and dollar values. Comparing it with
          the blind version shows how much an LLM leans on what it remembers from training.
  masked - only the last 10 years of values, rebased so the latest month = 100; no rate, no dates.
  maskedfull - the masked prompt with every month since January 2000 (memory check: does more
          history help, or give the dates away?).
  quiz - the masked 10-year data, but asks the LLM to identify the series instead of forecasting it
          ("name this series" memory check). Scored with `python llm_prompts.py quiz`.

    python llm_prompts.py build    -> llm/prompts/<variant>_<origin>.txt
    python llm_prompts.py score    -> reads llm/answers/<model>_<variant>_<origin>.txt, writes llm/llm_forecasts.csv
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "llm"

z = pd.read_csv(DATA / "Metro_zhvi.csv")
row = z[z.RegionName == "Los Angeles, CA"].iloc[0]
zhvi = row.iloc[5:].astype(float)
zhvi.index = pd.to_datetime(zhvi.index)
rate = (pd.read_csv(DATA / "MORTGAGE30US.csv", parse_dates=["observation_date"])
          .set_index("observation_date")["MORTGAGE30US"].resample("ME").mean())
df = pd.concat([zhvi.rename("zhvi"), rate.rename("rate")], axis=1).dropna()

N = len(df)
N_TRAIN = int(0.8 * N)
ORIGINS = list(range(N_TRAIN, N - 12 + 1, 3))      # same origins as the notebook's rolling test

QUIZ = ("Do not write or run code and do not use tools. Using only your own knowledge, identify this series as "
        "best you can. Reply with only a JSON object with keys: \"measure\" (what it most likely measures), "
        "\"country\", \"region\" (state/city/metro, or \"unknown\"), \"first_month\" and \"last_month\" (calendar "
        "YYYY-MM of the first and last rows, your best guess), \"next_12_months\" (what you expect happened over the "
        "12 months after the last row, in one sentence), \"confidence_0_100\", \"clues\" (one sentence on what gave "
        "it away).")
ASK = ("Forecast the home value for each of the next 12 months (months {a} to {b}). "
       "Do not write or run code and do not use tools; use your own judgment. "
       "Reply with only a JSON list of 12 numbers, nothing else.")


def prompt(o, variant):
    d = df.iloc[:o]
    if variant in ("masked", "maskedfull", "quiz"):
        w = d.zhvi if variant == "maskedfull" else d.zhvi.iloc[-120:]
        lines = [f"{i + 1},{100 * v / w.iloc[-1]:.3f}" for i, v in enumerate(w)]
        head = ("Below is a monthly index (latest month = 100). What it measures, the place "
                "and the dates are hidden.\n\nmonth,index\n")
        if variant == "quiz":
            return head + "\n".join(lines) + "\n\n" + QUIZ
        return head + "\n".join(lines) + "\n\n" + ASK.format(a=len(w) + 1, b=len(w) + 12).replace("home value", "index")
    if variant == "blind":
        base = d.zhvi.iloc[0]
        lines = [f"{i + 1},{100 * v / base:.3f},{r:.2f}" for i, (v, r) in enumerate(zip(d.zhvi, d.rate))]
        head = ("Below is a monthly regional home value index (first month = 100) and the average "
                "30-year mortgage rate (%) for the same month. Place and dates are hidden.\n\n"
                "month,home_value_index,mortgage_rate\n")
        return head + "\n".join(lines) + "\n\n" + ASK.format(a=o + 1, b=o + 12).replace("home value", "home value index")
    lines = [f"{t:%Y-%m},{v:.0f},{r:.2f}" for t, v, r in zip(d.index, d.zhvi, d.rate)]
    nxt = pd.date_range(d.index[-1], periods=13, freq="ME")[1:]
    head = ("Below is the Zillow Home Value Index (typical home value, US$) for the Los Angeles metro area "
            "and the average 30-year mortgage rate (%), monthly.\n\nmonth,zhvi_usd,mortgage_rate\n")
    return head + "\n".join(lines) + "\n\n" + ASK.format(a=f"{nxt[0]:%Y-%m}", b=f"{nxt[-1]:%Y-%m}").replace(
        "home value", "home value in US$")


def build():
    (OUT / "prompts").mkdir(parents=True, exist_ok=True)
    for o in ORIGINS:
        for v in ("blind", "named", "masked", "maskedfull", "quiz"):
            (OUT / "prompts" / f"{v}_{df.index[o - 1]:%Y-%m}.txt").write_text(prompt(o, v))
    print(f"{len(ORIGINS)} origins, {df.index[ORIGINS[0] - 1]:%Y-%m} to {df.index[ORIGINS[-1] - 1]:%Y-%m}")


def parse(text):
    m = re.findall(r"\[[^\[\]]*\]", text)
    vals = json.loads(m[-1]) if m else []
    if len(vals) != 12:
        raise ValueError(f"expected 12 numbers, got {len(vals)}")
    return np.array(vals, dtype=float)


def score():
    rows = []
    for f in sorted((OUT / "answers").glob("*.txt")):
        model, variant, origin = f.stem.rsplit("_", 2)
        if variant == "quiz":
            continue
        o = df.index.get_loc(pd.Timestamp(origin) + pd.offsets.MonthEnd(0)) + 1
        fc = parse(f.read_text())
        if variant == "blind":
            fc = fc * df.zhvi.iloc[0] / 100
        elif variant in ("masked", "maskedfull"):
            fc = fc * df.zhvi.iloc[o - 1] / 100
        actual = df.zhvi.iloc[o:o + 12].values
        ape = 100 * np.abs(fc - actual) / actual
        rows.append({"model": model, "variant": variant, "origin": origin,
                     **{f"h{k}": ape[k - 1] for k in (1, 3, 6, 12)}, "avg 1-12": ape.mean(),
                     **{f"f{k}": fc[k - 1] for k in range(1, 13)}})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "llm_forecasts.csv", index=False)
    print(res.groupby(["model", "variant"])[["h1", "h3", "h6", "h12", "avg 1-12"]].agg(["mean"]).round(2)
             .droplevel(1, axis=1).assign(n=res.groupby(["model", "variant"]).size()))


def quiz():
    """Score the "name this series" answers: right place? how many months off are the dates?"""
    rows = []
    for f in sorted((OUT / "answers").glob("*_quiz_*.txt")):
        model, _, origin = f.stem.rsplit("_", 2)
        o = df.index.get_loc(pd.Timestamp(origin) + pd.offsets.MonthEnd(0)) + 1
        text = f.read_text()
        m = re.findall(r"\{.*\}", text, re.S)
        a = json.loads(m[-1]) if m else {}
        where = " ".join(str(a.get(k, "")) for k in ("country", "region")).lower()

        def months_off(guess, true):
            try:
                g = pd.Period(str(guess)[:7], "M")
            except Exception:
                return np.nan
            return (g - true.to_period("M")).n

        rows.append({"model": model, "origin": origin, "measure": a.get("measure"), "country": a.get("country"),
                     "region": a.get("region"), "named_LA": ("los angeles" in where) or bool(re.search(r"\bla\b", where)),
                     "named_California": "california" in where or bool(re.search(r"\bca\b", where)),
                     "first_month_guess": a.get("first_month"), "last_month_guess": a.get("last_month"),
                     "last_month_true": f"{df.index[o - 1]:%Y-%m}",
                     "months_off_last": months_off(a.get("last_month"), df.index[o - 1]),
                     "confidence": a.get("confidence_0_100"), "next_12_months": a.get("next_12_months"),
                     "clues": a.get("clues")})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "llm_quiz.csv", index=False)
    print(res.groupby("model").agg(n=("origin", "size"), named_LA=("named_LA", "sum"),
                                   named_California=("named_California", "sum"),
                                   median_abs_months_off=("months_off_last", lambda x: x.abs().median()),
                                   within_12_months=("months_off_last", lambda x: (x.abs() <= 12).sum())))


if __name__ == "__main__":
    {"build": build, "score": score, "quiz": quiz}[sys.argv[1]]()
