from flask import Flask, request, jsonify
import math
import requests
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# ---------------------------------------------------------
# 0) Yardımcı Fonksiyonlar
# ---------------------------------------------------------

def clamp01(x):
    return max(0, min(1, x))

def normalize_slope(slope, max_percent=30):
    return clamp01(slope / max_percent)

def classify_flood_risk(fr):
    if fr < 0.3: return "Düşük"
    if fr < 0.6: return "Orta"
    if fr < 0.8: return "Yüksek"
    return "Çok Yüksek"

# ---------------------------------------------------------
# 1) DEM (Eğim) — Open Meteo + Open Elevation (Failover)
# ---------------------------------------------------------

def get_elevation_openmeteo(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return float(r.json()["elevation"][0])

def get_elevation_openelevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return float(r.json()["results"][0]["elevation"])

def get_dem(lat, lon):
    errors = []

    # 1) Open-Meteo
    try:
        return get_elevation_openmeteo(lat, lon), None
    except Exception as e:
        errors.append("Open-Meteo Elevation başarısız.")

    # 2) Open-Elevation fallback
    try:
        return get_elevation_openelevation(lat, lon), None
    except:
        errors.append("Open-Elevation başarısız.")

    # 3) ikisi de çökerse:
    return None, " / ".join(errors)


def estimate_slope(lat, lon):
    h1, err1 = get_dem(lat, lon)

    if err1:
        return None, err1

    delta_deg = 100 / 111320
    h2, err2 = get_dem(lat + delta_deg, lon)

    if err2:
        return None, err2

    slope = abs(h2 - h1) / 100 * 100
    return slope, None


# ---------------------------------------------------------
# 2) Yağış — Open-Meteo
# ---------------------------------------------------------

def fetch_rain(lat, lon):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        "&start_date=2015-01-01&end_date=2024-12-31"
        "&daily=precipitation_sum&timezone=UTC"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        daily = r.json()["daily"]["precipitation_sum"]
        total = sum(daily)
        mean = total / 10
        max_d = max(daily)
        p99 = sorted(daily)[int(len(daily)*0.99)]
        return mean, max_d, p99, None
    except:
        return None, None, None, "Yağış API hatası"


def compute_idf(max_daily):
    if max_daily is None:
        return 0
    a = max_daily * 1.3
    b = 12
    c = 0.75
    t = 15
    return a / ((t+b)**c)


# ---------------------------------------------------------
# 3) OSM — Bina Yoğunluğu ve Arazi
# ---------------------------------------------------------

def fetch_osm(lat, lon, radius=200):
    q = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius},{lat},{lon})["building"];
      nwr(around:{radius},{lat},{lon})["landuse"];
    );
    out tags;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": q}, timeout=30)
        r.raise_for_status()
        e = r.json()["elements"]

        building_count = 0
        landuses = []

        for i in e:
            tg = i.get("tags", {})
            if "building" in tg:
                building_count += 1
            if "landuse" in tg:
                landuses.append(tg["landuse"])

        return building_count, landuses, None

    except:
        return 0, [], "OSM API hatası"

def normalize_density_Turkey(buildings_per_km2):
    low = 500
    high = 5000
    if buildings_per_km2 <= low:
        return 0
    if buildings_per_km2 >= high:
        return 1
    return (buildings_per_km2 - low) / (high - low)

def permeability_from_landuse(land):
    if not land:
        return 0.5
    mc = Counter(land).most_common(1)[0][0]
    table = {
        "forest":0.85,"meadow":0.8,"grass":0.8,"greenfield":0.75,
        "farmland":0.6,"orchard":0.6,"residential":0.35,
        "commercial":0.30,"industrial":0.25
    }
    return table.get(mc,0.5)


# ---------------------------------------------------------
# 4) RİSK MODELİ
# ---------------------------------------------------------

def compute_risks(S, D, K, W_star, R_extreme):

    C = 0.5*D + 0.5*(1-K)
    W_eff = W_star * C
    B = 0.7*C + 0.3*K*(1-S)
    L = 0.6*W_eff + 0.4*R_extreme
    Flood = 0.6*L + 0.4*B

    return C, W_eff, B, L, Flood


# ---------------------------------------------------------
# 5) Drenaj Tipi
# ---------------------------------------------------------

def choose_system(S, D, K, C, W_eff, B, L, Flood):

    Score_DEN = 0.55*S + 0.25*L + 0.20*(1-B)
    Score_PAR = 0.50*(1-S) + 0.30*K + 0.20*(1-Flood)
    Score_RET = 0.40*C + 0.35*L + 0.25*(1-S)
    Score_HYB = 0.60*Flood + 0.40*(1 - abs(2*S-1))

    scores = {
        "dendritic": Score_DEN,
        "parallel": Score_PAR,
        "reticular": Score_RET,
        "hybrid": Score_HYB
    }

    sel = max(scores, key=scores.get)
    return sel, scores


# ---------------------------------------------------------
# 6) Boru & Hidrolik
# ---------------------------------------------------------

def manning_velocity(n, R_h, S):
    if S <= 0: S = 0.0001
    return (1/n) * (R_h**(2/3)) * (S**0.5)

def pipe_diameter(Q, v):
    if v <= 0 or Q <= 0:
        return 0
    return math.sqrt(4*Q/(math.pi*v))


# ---------------------------------------------------------
# 7) Ana API Endpoint
# ---------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    d = request.get_json()
    lat = float(d["lat"])
    lon = float(d["lon"])

    # --- DEM / EĞİM ---
    slope_percent, dem_error = estimate_slope(lat, lon)

    if slope_percent is None:
        S = None
    else:
        S = normalize_slope(slope_percent)

    # --- YAĞIŞ ---
    meanA, maxD, p99, rain_error = fetch_rain(lat, lon)
    if meanA:
        W_star = clamp01(meanA / 1000)
        R_extreme = clamp01(0.6*(maxD/150) + 0.4*(p99/80))
    else:
        W_star, R_extreme = 0.5, 0.5

    # --- OSM ---
    bcount, land, osm_error = fetch_osm(lat, lon)
    area_km2 = math.pi*(0.2**2)
    dens_km2 = bcount / area_km2 if area_km2>0 else 0
    D = normalize_density_Turkey(dens_km2)
    K = clamp01(permeability_from_landuse(land))

    # --- RİSK ---
    if S is not None:
        C, W_eff, B, L, Flood = compute_risks(S, D, K, W_star, R_extreme)
    else:
        C=W_eff=B=L=Flood=0

    # --- SİSTEM ---
    if S is not None:
        selected, scores = choose_system(S, D, K, C, W_eff, B, L, Flood)
    else:
        selected, scores = None, {}

    # --- HİDROLİK ---
    i_mm_h = compute_idf(maxD if maxD else 50)
    A_m2 = 5000
    Q = 0.00278 * C * i_mm_h * (A_m2/10000)
    v = manning_velocity(0.013, 0.5, (S/100) if S else 0.001)
    D_mm = pipe_diameter(Q, v)*1000

    return jsonify({
        "selected_system": selected,
        "scores": scores,
        "slope_percent": slope_percent,
        "S": S,
        "building_count": bcount,
        "density_bld_per_km2": dens_km2,
        "D": D,
        "K": K,
        "W_star": W_star,
        "R_extreme": R_extreme,
        "C": C,
        "W_eff": W_eff,
        "B": B,
        "L": L,
        "FloodRisk": Flood,
        "FloodRiskLevel": classify_flood_risk(Flood),
        "Q_m3_s": Q,
        "velocity_m_s": v,
        "pipe_diameter_mm": D_mm,
        "dem_error": dem_error,
        "rain_error": rain_error,
        "osm_error": osm_error
    })


@app.route("/")
def home():
    return "Biyomimikri Drenaj API v3.2 — Çalışıyor."

if __name__ == "__main__":
    app.run()
application = app
