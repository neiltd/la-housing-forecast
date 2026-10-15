"""Per-neighborhood population, schools, parks and crime within 1.5 km of each centroid,
plus cross-section regressions against the 2022 rate-shock drawdown and the last-12-month change."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

FP = Path(sys.argv[1])               # Final Project folder
A, FIG = FP / "data" / "amenities", FP / "figures"
R_KM = 1.5
EARTH = 6371.0088


def hav(lat1, lon1, lat2, lon2):
    """km distance matrix between arrays of points (1) and (2)."""
    la1, lo1, la2, lo2 = map(np.radians, (np.asarray(lat1)[:, None], np.asarray(lon1)[:, None],
                                          np.asarray(lat2)[None, :], np.asarray(lon2)[None, :]))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH * np.arcsin(np.sqrt(a))


def count_within(nb, lat, lon, weights=None, chunk=20000):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    w = np.ones(len(lat)) if weights is None else np.asarray(weights, float)
    out = np.zeros(len(nb))
    for i in range(0, len(lat), chunk):
        d = hav(nb.lat.values, nb.lon.values, lat[i:i + chunk], lon[i:i + chunk])
        out += (d <= R_KM) @ w[i:i + chunk]
    return out


# ---------------------------------------------------------------- neighborhoods
nb = pd.read_csv(FP / "data" / "la_nbhd_centroids.csv", dtype={"RegionID": str})
info = pd.read_csv(FP / "data" / "la_nbhd_zhvi.csv", usecols=["RegionID", "RegionName", "City", "CountyName"],
                   dtype={"RegionID": str})
nb = nb[["RegionID", "lat", "lon", "matched"]].merge(info, on="RegionID", how="left")
import re
def geo_ok(r):
    if pd.isna(r.lat) or pd.isna(r.matched):
        return False
    return re.search(rf"\b{re.escape(str(r.City))}\b(?! County)", str(r.matched)) is not None
nb["geocode_ok"] = nb.apply(geo_ok, axis=1)
print("neighborhoods:", len(nb), "| with coordinates:", nb.lat.notna().sum(), "| geocode_ok:", nb.geocode_ok.sum())
ALL = nb
nb = nb[nb.lat.notna()].copy()

# ---------------------------------------------------------------- population (CDC PLACES tract totals, ACS-based)
tp = json.load(open(A / "tract_pop.json"))
tp = pd.DataFrame({"pop": [float(r["totalpopulation"]) for r in tp],
                   "lon": [r["geolocation"]["coordinates"][0] for r in tp],
                   "lat": [r["geolocation"]["coordinates"][1] for r in tp]})
nb["population"] = count_within(nb, tp.lat, tp.lon, tp["pop"])

# ---------------------------------------------------------------- schools (CDE directory, active public schools)
sc = pd.read_csv(A / "pubschls.txt", sep="\t", dtype=str, encoding="latin-1", usecols=["StatusType", "School",
                 "Latitude", "Longitude"])
sc = sc[(sc.StatusType == "Active") & sc.School.notna()]
sc["Latitude"], sc["Longitude"] = pd.to_numeric(sc.Latitude, errors="coerce"), pd.to_numeric(sc.Longitude, errors="coerce")
sc = sc.dropna(subset=["Latitude", "Longitude"])
sc = sc[sc.Latitude.between(33.3, 35.0) & sc.Longitude.between(-119.2, -117.3)]
print("active public schools in area:", len(sc))
nb["schools"] = count_within(nb, sc.Latitude, sc.Longitude)

# ---------------------------------------------------------------- parks (OSM leisure=park)
def ring_area_ha(pts):
    lat0 = np.radians(np.mean([p["lat"] for p in pts]))
    x = np.array([p["lon"] for p in pts]) * 111.320 * np.cos(lat0)
    y = np.array([p["lat"] for p in pts]) * 110.574
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) * 100   # km^2 -> ha

parks = []
for e in json.load(open(A / "parks_osm.json"))["elements"]:
    if e["type"] == "way" and e.get("geometry"):
        g = e["geometry"]
        parks.append((np.mean([p["lat"] for p in g]), np.mean([p["lon"] for p in g]),
                      ring_area_ha(g) if g[0] == g[-1] and len(g) > 3 else 0.0))
    elif e["type"] == "relation":
        outer = [m["geometry"] for m in e.get("members", []) if m.get("geometry") and m.get("role") == "outer"]
        inner = [m["geometry"] for m in e.get("members", []) if m.get("geometry") and m.get("role") == "inner"]
        if not outer:
            continue
        pts = [p for g in outer for p in g]
        area = sum(ring_area_ha(g) for g in outer if g[0] == g[-1]) - sum(ring_area_ha(g) for g in inner if g[0] == g[-1])
        parks.append((np.mean([p["lat"] for p in pts]), np.mean([p["lon"] for p in pts]), max(area, 0.0)))
pk = pd.DataFrame(parks, columns=["lat", "lon", "ha"])
print("parks:", len(pk), "| total ha:", round(pk.ha.sum()))
nb["parks"] = count_within(nb, pk.lat, pk.lon)
nb["park_ha"] = count_within(nb, pk.lat, pk.lon, pk.ha)

# ---------------------------------------------------------------- crime (City of LA only)
c21 = pd.read_csv(A / "crime_2021.csv")
c21 = c21[(c21.lat.abs() > 1) & (c21.lon.abs() > 1)]
c25 = pd.read_csv(A / "crime_2025.csv").rename(columns={"hndrdth_lat": "lat", "hndrdth_lon": "lon"})
c25 = c25[(c25.lat.abs() > 1) & (c25.lon.abs() > 1)]
c25["caseno"] = c25.uniquenibrno.str.split("_").str[0]
inc25 = c25.groupby("caseno").agg(lat=("lat", "first"), lon=("lon", "first"),
                                  person=("crime_against", lambda s: (s == "Person").any()))
print(f"crime rows 2021: {len(c21)}, 2025 incidents: {len(inc25)} ({inc25.person.sum()} with a crime against a person)")
la = nb.City == "Los Angeles"
nb["crime_2021"] = np.nan; nb["crime_2025"] = np.nan; nb["violent_2025"] = np.nan
sub = nb[la]
nb.loc[la, "crime_2021"] = count_within(sub, c21.lat, c21.lon)
nb.loc[la, "crime_2025"] = count_within(sub, inc25.lat, inc25.lon)
nb.loc[la, "violent_2025"] = count_within(sub, inc25.lat[inc25.person], inc25.lon[inc25.person])
for c in ["crime_2021", "crime_2025", "violent_2025"]:
    nb[c + "_per1k"] = 1000 * nb[c] / nb.population.where(nb.population > 0)

nb.loc[la & (nb.crime_2025 < 50), "geocode_ok"] = False   # a City-of-LA point with ~no crime is off-target
nb = ALL[["RegionID"]].merge(nb.drop(columns=[]), on="RegionID", how="left").merge(
    ALL[["RegionID", "RegionName", "City", "CountyName", "matched"]], on="RegionID", how="left", suffixes=("_x", ""))
nb = nb[[c for c in nb.columns if not c.endswith("_x")]]
nb["geocode_ok"] = nb.geocode_ok.fillna(False).astype(bool)
nb["radius_km"] = R_KM
cols = ["RegionID", "RegionName", "City", "CountyName", "lat", "lon", "geocode_ok", "radius_km", "population", "schools", "parks",
        "park_ha", "crime_2021", "crime_2025", "violent_2025", "crime_2021_per1k", "crime_2025_per1k",
        "violent_2025_per1k", "matched"]
nb = nb[cols]
out = nb.copy()
num = ["population", "schools", "parks", "park_ha", "crime_2021", "crime_2025", "violent_2025",
       "crime_2021_per1k", "crime_2025_per1k", "violent_2025_per1k"]
out[num] = out[num].round(2)
out.to_csv(A / "nbhd_amenities.csv", index=False)
print("saved", A / "nbhd_amenities.csv")

# ---------------------------------------------------------------- sanity checks
show = ["Bel Air", "Downtown", "Venice", "South Los Angeles", "Southeast Los Angeles", "Santa Monica", "Irvine",
        "Hollywood", "Westwood", "Watts", "Encino"]
print(nb[nb.RegionName.isin(show)][["RegionName", "City", "population", "schools", "parks", "park_ha",
                                     "crime_2021_per1k", "crime_2025_per1k", "violent_2025_per1k"]].round(1).to_string())
print("\npopulation 0:", (nb.population == 0).sum(), "| schools 0:", (nb.schools == 0).sum())
print(nb[["population", "schools", "parks", "park_ha", "crime_2021_per1k", "crime_2025_per1k"]].describe().round(1).to_string())
# flag geocodes whose match string names a different city
bad = nb[~nb.apply(lambda r: str(r.City).split(",")[0] in str(r.matched), axis=1)]
print(f"\ngeocode matches that do not mention the neighborhood's city: {len(bad)}")
print(bad[["RegionName", "City", "matched"]].head(15).to_string())
print("crime per 1k extremes:")
print(nb.dropna(subset=["crime_2025_per1k"]).sort_values("crime_2025_per1k")[["RegionName", "population",
      "crime_2025", "crime_2025_per1k"]].iloc[[0, 1, 2, -3, -2, -1]].round(1).to_string())

# ---------------------------------------------------------------- regressions
shock = pd.read_csv(FIG / "nbhd_rate_shock.csv", dtype={"RegionID": str})[["RegionID", "value_dec2021", "drawdown_%"]]
trend = pd.read_csv(FIG / "nbhd_trend_now.csv", dtype={"RegionID": str})[["RegionID", "value_aug2026", "chg_12m_%"]]
d = nb.merge(shock, on="RegionID", how="left").merge(trend, on="RegionID", how="left")
d = d.rename(columns={"drawdown_%": "drawdown", "chg_12m_%": "chg12"})
d["log_price21"] = np.log(d.value_dec2021)
d["log_price26"] = np.log(d.value_aug2026)
d["log_crime21"] = np.log(d.crime_2021_per1k)
d["log_crime25"] = np.log(d.crime_2025_per1k)
d["log_park_ha"] = np.log1p(d.park_ha)
d = d.replace([np.inf, -np.inf], np.nan)
print("rows in regressions (geocode_ok):", d.geocode_ok.sum())
d = d[d.geocode_ok]

specs = [
    ("drawdown", "all", "drawdown ~ log_price21", d),
    ("drawdown", "all", "drawdown ~ log_price21 + schools + parks + log_park_ha", d),
    ("drawdown", "City of LA", "drawdown ~ log_price21 + schools + parks + log_park_ha", d[d.City == "Los Angeles"]),
    ("drawdown", "City of LA", "drawdown ~ log_price21 + schools + parks + log_park_ha + log_crime21", d[d.City == "Los Angeles"]),
    ("chg12", "all", "chg12 ~ log_price26", d),
    ("chg12", "all", "chg12 ~ log_price26 + schools + parks + log_park_ha", d),
    ("chg12", "City of LA", "chg12 ~ log_price26 + schools + parks + log_park_ha", d[d.City == "Los Angeles"]),
    ("chg12", "City of LA", "chg12 ~ log_price26 + schools + parks + log_park_ha + log_crime25", d[d.City == "Los Angeles"]),
]
rows = []
for i, (y, sample, f, data) in enumerate(specs, 1):
    m = smf.ols(f, data=data).fit(cov_type="HC1")
    print(f"\n[{i}] {f}   ({sample}, n={int(m.nobs)}, R2={m.rsquared:.3f}, adj R2={m.rsquared_adj:.3f})")
    print(pd.DataFrame({"coef": m.params, "t": m.tvalues, "p": m.pvalues}).round(4).to_string())
    for k in m.params.index:
        rows.append({"model": i, "y": y, "sample": sample, "formula": f, "n": int(m.nobs), "r2": m.rsquared,
                     "adj_r2": m.rsquared_adj, "term": k, "coef": m.params[k], "t": m.tvalues[k], "p": m.pvalues[k]})
    if "log_crime" in f or "+ schools" in f:
        r = m.f_test(" = 0, ".join([t for t in m.params.index if t not in ("Intercept",) and not t.startswith("log_price")]) + " = 0")
        print(f"    joint test amenities = 0: F={float(np.squeeze(r.fvalue)):.2f}, p={float(r.pvalue):.4f}")
        rows.append({"model": i, "y": y, "sample": sample, "formula": f, "n": int(m.nobs), "r2": m.rsquared,
                     "adj_r2": m.rsquared_adj, "term": "joint F amenities=0", "coef": float(np.squeeze(r.fvalue)),
                     "t": np.nan, "p": float(r.pvalue)})
pd.DataFrame(rows).round(4).to_csv(FIG / "amenities_regression.csv", index=False)

print("\ncorrelations (all neighborhoods / City of LA for crime):")
cc = ["drawdown", "chg12", "log_price21", "schools", "parks", "log_park_ha", "log_crime21", "log_crime25"]
print(d[cc].corr().round(2).to_string())
print("saved", FIG / "amenities_regression.csv")
