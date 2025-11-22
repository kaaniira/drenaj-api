from flask import Flask, request, jsonify
import requests
import math
from statistics import mode

app = Flask(__name__)

def clip(x, a=0, b=1):
    return max(a, min(b, x))

# ---------------------------
# 1) EĞİM (OpenTopography)
# ---------------------------
OPENTOPO_API_KEY = "fbf76fcd2c25137d9bd8b5b46ec06250"

def get_elevation(lat, lon):
    url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL3&lat={lat}&lon={lon}&key={OPENTOPO_API_KEY}"
    )
    r = requests.get(url).json()
    return r["data"]["elevation"]

def compute_slope(lat, lon, dist_m=100):
    dlat = dist_m / 111320
    h0 = get_elevation(lat, lon)
    h1 = get_elevation(lat + dlat, lon)
    slope_percent = abs((h1 - h0) / dist_m) * 100
    S = clip(slope_percent / 60)
    return S, slope_percent

# ---------------------------
# 2) OSM YOĞUNLUK + LANDUSE
# ---------------------------
def fetch_osm(lat, lon, radius=100):
    query = f"""
    [out:json];
    (
      way(around:{radius},{lat},{lon})["building"];
      way(around:{radius},{lat},{lon})["landuse"];
    );
    out tags;
    """
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": query}).json()
    buildings = 0
    landuses = []
    for el in r.get("elements", []):
        tags = el.get("tags", {})
        if "building" in tags:
            buildings += 1
        if "landuse" in tags:
            landuses.append(tags["landuse"])
    return buildings, landuses

def compute_density(buildings, radius):
    area_km2 = math.pi * (radius/1000)**2
    density = buildings / area_km2 if area_km2 else 0
    return clip(density / 12000), density

def compute_K(landuses):
    if not landuses:
        return 0.5
    LU = mode(landuses)
    mapping = {
        "forest": 0.85, "meadow": 0.8, "grass": 0.8,
        "greenfield": 0.7, "farmland": 0.6,
        "orchard": 0.6, "residential": 0.35,
        "commercial": 0.25, "industrial": 0.25
    }
    return clip(mapping.get(LU, 0.5))

# ---------------------------
# 3) YAĞIŞ (Open-Meteo)
# ---------------------------
def fetch_rain(lat, lon):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        "&start_date=2015-01-01&end_date=2024-12-31"
        "&daily=precipitation_sum&timezone=UTC"
    )
    r = requests.get(url).json()
    precip = r["daily"]["precipitation_sum"]

    total = sum(precip)
    W_star = clip((total / 10) / 1000)

    max_daily = max(precip)
    p99 = sorted(precip)[int(len(precip) * 0.99)]
    R = clip(0.6*(max_daily/150) + 0.4*(p99/80))

    return W_star, R, max_daily, p99

# ---------------------------
# 4) MODEL (v2.2)
# ---------------------------
def compute_model(S, D, K, W_star, R):
    C = clip(0.5*D + 0.5*(1-K))
    W_eff = W_star * C
    B = clip(0.7*C + 0.3*K*(1-S))
    L = clip(0.6*W_eff + 0.4*R)
    FloodRisk = clip(0.6*L + 0.4*B)

    Score_DEN = 0.55*S + 0.25*L + 0.20*(1-B)
    Score_PAR = 0.50*(1-S) + 0.30*K + 0.20*(1-FloodRisk)
    Score_RET = 0.40*C + 0.35*L + 0.25*(1-S)
    Score_HYB = 0.60*FloodRisk + 0.40*(1 - abs(S-0.5)*2)

    scores = {
        "dendritik": Score_DEN,
        "paralel": Score_PAR,
        "retikuler": Score_RET,
        "hibrit": Score_HYB
    }

    system = max(scores, key=lambda k: scores[k])

    explanation = f"""
Eğim (S={S:.2f}), yüzey akış davranışını etkiler.
C = {C:.2f}, W_eff = {W_eff:.2f}, B = {B:.2f}
L = {L:.2f}, FloodRisk = {FloodRisk:.2f}
Bu nedenle en uygun sistem: {system.upper()}
"""

    internals = {
        "C": C, "W_eff": W_eff, "B": B,
        "L": L, "FloodRisk": FloodRisk
    }

    return system, scores, internals, explanation

# ---------------------------
# API ROUTE
# ---------------------------
@app.route("/api", methods=["POST"])
def api():
    d = request.get_json()
    lat = float(d["lat"])
    lon = float(d["lon"])

    S, raw_slope = compute_slope(lat, lon)
    buildings, landuses = fetch_osm(lat, lon)
    D, density_raw = compute_density(buildings, 100)
    K = compute_K(landuses)
    W_star, R, max_daily, p99 = fetch_rain(lat, lon)

    system, scores, internals, explanation = compute_model(S, D, K, W_star, R)

    return jsonify({
        "system": system,
        "scores": scores,
        "explanation": explanation,
        "internal": internals,
        "raw": {
            "slope_percent": raw_slope,
            "buildings": buildings,
            "density_km2": density_raw,
            "landuse_sample": landuses[:5],
            "W_star": W_star,
            "R": R,
            "max_daily": max_daily,
            "p99": p99
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
