# ============================================================
#  BİYOMİMİKRİ TABANLI DRENAJ SİSTEMİ API (v4.1)
#  TÜBİTAK için bilimsel olarak kalibre edilmiş son sürüm
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
from collections import Counter

app = Flask(__name__)
CORS(app)   # WordPress için CORS açık

# ============================================================
#  GENEL YARDIMCI FONKSİYONLAR
# ============================================================

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def safe(v, default=None):
    return v if (v is not None and v == v) else default

def classify_flood(F):
    if F < 0.3: return "Düşük"
    if F < 0.6: return "Orta"
    if F < 0.8: return "Yüksek"
    return "Çok Yüksek"

# ============================================================
#  DEM (EĞİM) HESABI — Open-Meteo + Open-Elevation Failover
# ============================================================

def get_elev_openmeteo(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return float(r.json()["elevation"][0])

def get_elev_openelev(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return float(r.json()["results"][0]["elevation"])

def get_elevation(lat, lon):
    errors = []

    try:
        return get_elev_openmeteo(lat, lon), None
    except:
        errors.append("Open-Meteo başarısız.")

    try:
        return get_elev_openelev(lat, lon), None
    except:
        errors.append("Open-Elevation başarısız.")

    return None, " / ".join(errors)

def estimate_slope_percent(lat, lon):
    h1, e1 = get_elevation(lat, lon)
    if h1 is None:
        return None, e1

    delta_lat = 100 / 111320
    h2, e2 = get_elevation(lat + delta_lat, lon)
    if h2 is None:
        return None, e2

    slope_percent = abs(h2 - h1)
    return slope_percent, None

# ============================================================
#  YAĞIŞ (10 YIL) — Open-Meteo Archive
# ============================================================

def fetch_precip(lat, lon):
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            "&start_date=2015-01-01&end_date=2024-12-31"
            "&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        arr = r.json()["daily"]["precipitation_sum"]

        total = sum(arr)
        mean_annual = total / 10
        max_daily = max(arr)
        p99 = sorted(arr)[int(len(arr) * 0.99)]
        return mean_annual, max_daily, p99, None

    except:
        return None, None, None, "Yağış API hatası"

def compute_idf_intensity(max_daily):
    if max_daily is None:
        return 0
    a = max_daily * 1.3
    b = 12
    c = 0.75
    t = 15
    return a / ((t + b) ** c)

# ============================================================
#  OSM — BİNA YOĞUNLUĞU + LANDUSE
# ============================================================

def fetch_osm(lat, lon, radius=200):
    query = f"""
    [out:json][timeout:25];
    (
      nwr(around:{radius},{lat},{lon})["building"];
      nwr(around:{radius},{lat},{lon})["landuse"];
    );
    out tags;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=30)
        r.raise_for_status()
        elements = r.json()["elements"]

        buildings = 0
        lands = []
        for el in elements:
            t = el.get("tags", {})
            if "building" in t:
                buildings += 1
            if "landuse" in t:
                lands.append(t["landuse"])

        return buildings, lands, None

    except:
        return 0, [], "OSM API hatası"

def normalize_density_turkey(dens):
    low = 500
    high = 5000
    if dens <= low: return 0
    if dens >= high: return 1
    return (dens - low) / (high - low)

def permeability_from_landuse(lands):
    if not lands:
        return 0.5
    mc = Counter(lands).most_common(1)[0][0]
    table = {
        "forest":0.85,
        "meadow":0.80,
        "grass":0.80,
        "greenfield":0.75,
        "farmland":0.60,
        "orchard":0.60,
        "residential":0.35,
        "commercial":0.30,
        "industrial":0.25
    }
    return table.get(mc, 0.5)

# ============================================================
#  RİSK HESAPLARI
# ============================================================

def compute_risks(S, D, K, W_star, R_ext):
    C = 0.5*D + 0.5*(1-K)
    W_eff = W_star * C
    B = 0.7*C + 0.3*K*(1-S)
    L = 0.6*W_eff + 0.4*R_ext
    Flood = 0.6*L + 0.4*B
    return C, W_eff, B, L, Flood

# ============================================================
#  DRENAJ TİPİ – AHP NORMALİZE SKORLARI (Toplam = 1)
# ============================================================

def choose_system(S, D, K, C, W_eff, B, L, Flood):

    Score_DEN = 0.45*S + 0.30*L + 0.25*(1-K)
    Score_PAR = 0.45*(1-S) + 0.30*K + 0.25*(1-Flood)
    Score_RET = 0.50*D + 0.30*(1-K) + 0.20*L
    Score_HYB = 0.40*(1 - abs(2*S - 1)) + 0.35*Flood + 0.25*(1-K)

    scores = {
        "dendritic": Score_DEN,
        "parallel": Score_PAR,
        "reticular": Score_RET,
        "hybrid": Score_HYB
    }

    selected = max(scores, key=scores.get)
    return selected, scores

# ============================================================
#  MANNING – ANALİTİK BORU ÇAPI
# ============================================================

def manning_diameter(Q, n, S):
    if Q <= 0 or S <= 0:
        return 0
    return ((4**(5/3) * n * Q) / (math.pi * (S**0.5))) ** (3/8)

# ============================================================
#  ANA API
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    lat = float(data["lat"])
    lon = float(data["lon"])

    # 1 — EĞİM
    slope_percent, dem_error = estimate_slope_percent(lat, lon)
    S = clamp((slope_percent / 30)) if slope_percent else 0.0

    # 2 — YAĞIŞ
    meanA, maxD, p99, rain_error = fetch_precip(lat, lon)
    W_star = clamp((meanA / 1000)) if meanA else 0.5
    R_ext = clamp(0.6*(safe(maxD,0)/150) + 0.4*(safe(p99,0)/80))

    # 3 — OSM
    bcount, lands, osm_error = fetch_osm(lat, lon)
    area_km2 = math.pi * (0.2**2)
    dens_km2 = bcount / area_km2 if area_km2 > 0 else 0
    D = normalize_density_turkey(dens_km2)
    K = clamp(permeability_from_landuse(lands))

    # 4 — RİSKLER
    C, W_eff, B, L, Flood = compute_risks(S, D, K, W_star, R_ext)

    # 5 — SİSTEM SEÇİMİ
    selected, scores = choose_system(S, D, K, C, W_eff, B, L, Flood)

    # 6 — HİDROLİK
    i_mm_h = compute_idf_intensity(maxD)
    A_m2 = 5000
    Q = 0.00278 * C * i_mm_h * (A_m2 / 10000)
    D_mm = manning_diameter(Q, 0.013, max(S,0.001)) * 1000

    return jsonify({
        "selected_system": selected,
        "scores": scores,
        "slope_percent": slope_percent,
        "S": S,
        "building_count": bcount,
        "density_km2": dens_km2,
        "D": D,
        "K": K,
        "W_star": W_star,
        "R_extreme": R_ext,
        "C": C,
        "W_eff": W_eff,
        "B": B,
        "L": L,
        "FloodRisk": Flood,
        "FloodRisk_Level": classify_flood(Flood),
        "Q_m3_s": Q,
        "pipe_diameter_mm": D_mm,
        "dem_error": dem_error,
        "rain_error": rain_error,
        "osm_error": osm_error
    })

@app.route("/")
def home():
    return "Drenaj API v4.1 — TÜBİTAK Bilimsel Kalibrasyonlu"

if __name__ == "__main__":
    app.run()
