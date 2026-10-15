"""Crime by type for the City of LA neighborhoods: violent vs property vs all crime.

Does crime explain which neighborhoods fell hardest in 2022-23, or which are rising now, once the
price level is controlled for? City of LA only (LAPD data), neighborhoods with a reliable map position.
Run after amenities.py:  python crime_city_la.py  (from the Final Project folder)
"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, FIG = Path("data/amenities"), Path("figures")
R_KM, EARTH = 1.5, 6371.0088
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"

# FBI "Part I" definitions, so 2021 (LAPD crime codes) and 2025 (NIBRS offense codes) count the same things
VIOLENT_21 = {110, 113, 121, 122, 815, 820, 821, 210, 220, 230, 231, 235, 236, 250, 251, 761, 926}
PROPERTY_21 = set(range(310, 523))          # burglary, theft, theft from / of vehicles
VIOLENT_25 = {"09A", "11A", "11B", "11C", "120", "13A"}   # murder, rape, robbery, aggravated assault
PROPERTY_25 = {"220", "240"} | {f"23{c}" for c in "ABCDEFGH"}   # burglary, vehicle theft, larceny


def count_within(nb, lat, lon, chunk=20000):
    la1, lo1 = np.radians(nb.lat.values)[:, None], np.radians(nb.lon.values)[:, None]
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    out = np.zeros(len(nb))
    for i in range(0, len(lat), chunk):
        la2, lo2 = lat[None, i:i + chunk], lon[None, i:i + chunk]
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        out += (2 * EARTH * np.arcsin(np.sqrt(a)) <= R_KM).sum(axis=1)
    return out


nb = pd.read_csv(A / "nbhd_amenities.csv", dtype={"RegionID": str})
nb = nb[(nb.City == "Los Angeles") & nb.geocode_ok & (nb.population > 0)].copy()

c21 = pd.read_csv(A / "crime_2021.csv")
c21 = c21[(c21.lat.abs() > 1) & (c21.lon.abs() > 1)]
c25 = pd.read_csv(A / "crime_2025.csv").rename(columns={"hndrdth_lat": "lat", "hndrdth_lon": "lon"})
c25 = c25[(c25.lat.abs() > 1) & (c25.lon.abs() > 1)]
c25["case"] = c25.uniquenibrno.str.split("_").str[0]
c25["off"] = c25.uniquenibrno.str.split("_").str[1]
inc25 = c25.groupby("case").agg(lat=("lat", "first"), lon=("lon", "first"),
                                violent=("off", lambda s: s.isin(VIOLENT_25).any()),
                                property=("off", lambda s: s.isin(PROPERTY_25).any()))

for yr, df_, v, p in [("21", c21, c21.crm_cd.isin(VIOLENT_21), c21.crm_cd.isin(PROPERTY_21)),
                      ("25", inc25, inc25.violent, inc25.property)]:
    for kind, mask in [("all", np.ones(len(df_), bool)), ("violent", v.values), ("property", p.values)]:
        nb[f"{kind}{yr}"] = 1000 * count_within(nb, df_.lat.values[mask], df_.lon.values[mask]) / nb.population
print(f"{len(nb)} City of LA neighborhoods")
print("share violent: 2021 %.0f%%, 2025 %.0f%%" % (100 * c21.crm_cd.isin(VIOLENT_21).mean(), 100 * inc25.violent.mean()))

shock = pd.read_csv(FIG / "nbhd_rate_shock.csv", dtype={"RegionID": str})[["RegionID", "value_dec2021", "drawdown_%"]]
now = pd.read_csv(FIG / "nbhd_trend_now.csv", dtype={"RegionID": str})[["RegionID", "value_aug2026", "chg_12m_%"]]
d = nb.merge(shock, on="RegionID").merge(now, on="RegionID")
d = d.rename(columns={"drawdown_%": "drop22", "chg_12m_%": "chg12"})
d["lp21"], d["lp26"] = np.log(d.value_dec2021), np.log(d.value_aug2026)
for c in ["all21", "violent21", "property21", "all25", "violent25", "property25"]:
    d["l_" + c] = np.log(d[c].clip(lower=0.1))

# correlations: crime with price level (the confound)
print("\ncorrelation of log crime per 1,000 with log home value:")
print({c: round(d["l_" + c].corr(d.lp21 if c.endswith("21") else d.lp26), 2)
       for c in ["all21", "violent21", "property21", "all25", "violent25", "property25"]})

rows = []
for y, lp, yr, label in [("drop22", "lp21", "21", "2022-23 drop (%)"), ("chg12", "lp26", "25", "Last 12 months (%)")]:
    for kind in ["all", "violent", "property"]:
        x = f"l_{kind}{yr}"
        for ctrl in [False, True]:
            f = f"{y} ~ {x}" + (f" + {lp}" if ctrl else "")
            m = smf.ols(f, data=d).fit(cov_type="HC1")
            rows.append({"outcome": label, "crime": kind, "price control": ctrl, "n": int(m.nobs),
                         "coef (pts per doubling)": m.params[x] * np.log(2), "p": m.pvalues[x], "R2": m.rsquared})
res = pd.DataFrame(rows)
print("\n" + res.round(3).to_string(index=False))
res.round(4).to_csv(FIG / "crime_city_la.csv", index=False)
d[["RegionID", "RegionName", "population", "value_aug2026", "drop22", "chg12",
   "all21", "violent21", "property21", "all25", "violent25", "property25"]].round(2).to_csv(
    A / "crime_city_la_nbhd.csv", index=False)

# ---------------------------------------------------------------- figure: raw link vs after price control
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.edgecolor": "#c3c2b7"})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axes[0]
ax.scatter(d.violent25, d.value_aug2026 / 1e3, s=18, color=BLUE, alpha=0.7, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_yticks([500, 750, 1000, 1500, 2000, 3000, 4000]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1e3:g}M" if v >= 1000 else f"${v:.0f}K")); ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("Violent crimes per 1,000 residents, 2025"); ax.set_ylabel("Typical home value, Aug 2026")
ax.set_title(f"Safer areas cost more\n(correlation {d.l_violent25.corr(d.lp26):.2f})")
for ax, y, lp, yr, ttl in [(axes[1], "drop22", "lp21", "21", "...but it didn't predict\nthe 2022-23 drop"),
                           (axes[2], "chg12", "lp26", "25", "...or who is\nrising now")]:
    # partial-residual view: both sides with the price level taken out
    ry = smf.ols(f"{y} ~ {lp}", data=d).fit().resid
    rx = smf.ols(f"l_violent{yr} ~ {lp}", data=d).fit().resid
    ax.scatter(rx, ry, s=18, color=GRAY, alpha=0.8, lw=0)
    b = np.polyfit(rx, ry, 1); xx = np.linspace(rx.min(), rx.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=ORANGE, lw=1.6)
    r = res[(res.crime == "violent") & res["price control"] & (res.outcome.str.startswith("2022" if y == "drop22" else "Last"))].iloc[0]
    ax.set_title(f"{ttl} (p = {r.p:.2f})")
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel(f"Violent crime 20{yr}, vs same-price areas (log)")
    ax.set_ylabel(("2022-23 change" if y == "drop22" else "Last-12-month change") + ", vs same-price (pts)")
fig.suptitle(f"City of LA, {len(d)} neighborhoods: crime within 1.5 km (LAPD data)", x=0.01, ha="left",
             fontweight="bold", color=INK, y=1.04)
fig.tight_layout(w_pad=3); fig.savefig(FIG / "21_crime_city_la.png")
print("saved figures/21_crime_city_la.png")
