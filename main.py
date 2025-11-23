# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — OTO HAVZA HESAPLAMALI
#  TÜBİTAK V5.0 (Profesyonel Hidroloji Modeli)
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
from collections import Counter

app = Flask(__name__)
CORS(app)


# ============================================================
#  YARDIMCI FONKSİYONLAR
# ============================================================

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def safe(v, default=None):
    return v if (v is not None and v == v) else default


# ============================================================
#  DEM → EĞİM (Open-Meteo + OpenElevation Failover)
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
    h1, err = get_elevation(lat, lon)
    if h1 is None:
        return None, err

    delta = 100 / 111320.0
    h2, err2 = get_elevation(lat + delta, lon)
    if h2 is None:
        return None, err2

    slope = abs(h2 - h1)
    return slope, None


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
        meanA = total / 10
        maxD = max(arr)
        p99 = sorted(arr)[int(len(arr) * 0.99)]

        return meanA, maxD, p99, None
    except:
        return None, None, None, "Yağış API hatası"

def compute_idf_intensity(max_daily):
    if not max_daily:
        return 0
    return (max_daily * 1.3) / ((15 + 12)**0.75)


# ============================================================
#  OSM LANDUSE + BUILDINGS
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
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=30)
        r.raise_for_status()
        elements = r.json()["elements"]

        buildings = 0
        lands = []

        for el in elements:
            tags = el.get("tags", {})
            if "building" in tags:
                buildings += 1
            if "landuse" in tags:
                lands.append(tags["landuse"])

        return buildings, lands, None
    except:
        return 0, [], "OSM API hatası"


# ============================================================
#  YOĞUNLUK NORMALİZASYONU (Türkiye Kalibrasyonu)
# ============================================================

def normalize_density_turkey(dens):
    low = 500
    high = 5000
    if dens <= low:
        return 0
    if dens >= high:
        return 1
    return (dens - low) / (high - low)


# ============================================================
#  GEÇİRGENLİK
# ============================================================

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
#  OSM ROADS → DİNAMİK HAVZA ALANI (A_m2)
# ============================================================

def fetch_osm_roads(lat, lon, radius=200):
    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius},{lat},{lon})["highway"];
    );
    out geom;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=30)
        r.raise_for_status()
        elements = r.json()["elements"]

        total_len = 0.0

        for el in elements:
            geom = el.get("geometry", [])
            for i in range(len(geom)-1):
                lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
                lat2, lon2 = geom[i+1]["lat"], geom[i+1]["lon"]

                dx = (lon2 - lon1) * 85_000
                dy = (lat2 - lat1) * 111_320
                total_len += (dx*dx + dy*dy)**0.5

        return total_len, None
    except:
        return 0, "Roads API hatası"


def estimate_catchment_area(total_road_m, D, K):
    if total_road_m <= 0:
        return 30_000

    W_avg = 10
    A_roads = total_road_m * W_avg * 1.3
    A_density = A_roads * (1 + 0.6 * D)
    A_final = A_density * (1 + 0.5*(1-K))
    return A_final


# ============================================================
#  RİSK HESAPLARI
# ============================================================

def compute_risks(S, D, K, Wstar, Rext):
    C = 0.5*D + 0.5*(1-K)
    W_eff = Wstar * C
    B = 0.7*C + 0.3*K*(1-S)
    L = 0.6*W_eff + 0.4*Rext
    Flood = 0.6*L + 0.4*B
    return C, W_eff, B, L, Flood


# ============================================================
#  AHP DRENAJ TİPİ SEÇİMİ
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
#  MANNING BORU ÇAPI
# ============================================================

def manning_diameter(Q, n, S):
    if Q <= 0 or S <= 0:
        return 0
    return ((4**(5/3) * n * Q) / (math.pi * math.sqrt(S)))**(3/8)


# ============================================================
#  ANA API
# ============================================================

def classify_scale(D_mm, Q, A_m2):
    A_ha = A_m2 / 10000.0

    # Street Drain
    if D_mm < 500 and Q < 1.5 and A_ha < 3:
        return "Sokak Hattı", "🟩"

    # Secondary Collector
    if (500 <= D_mm < 1000) or (1.5 <= Q < 5) or (3 <= A_ha < 10):
        return "Mahalle Kolektörü", "🟨"

    # Major Trunk
    return "Ana Kolektör / Trunk", "🟥"



@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()

    lat = float(data["lat"])
    lon = float(data["lon"])

    # --- Eğim ---
    slope_percent, dem_error = estimate_slope_percent(lat, lon)
    # Normalize eğim (AHP için)
    S = clamp((slope_percent / 30)) if slope_percent else 0.0

    # Manning için taban eğimi (m/m) – 0.3% ile 3% arasında sınırla
    raw_bed_slope = (slope_percent or 0) / 100.0
    S_bed = max(0.003, min(raw_bed_slope, 0.03))

    # --- Yağış ---
    meanA, maxD, p99, rain_error = fetch_precip(lat, lon)
    Wstar = clamp((meanA / 1000)) if meanA else 0.5
    Rext = clamp(0.6*(safe(maxD,0)/150) + 0.4*(safe(p99,0)/80))

    # --- OSM yapı ---
    bcount, lands, osm_error = fetch_osm(lat, lon)
    area_km2 = math.pi * (0.2**2)
    dens_km2 = bcount / area_km2 if area_km2 > 0 else 0
    D = normalize_density_turkey(dens_km2)
    K = clamp(permeability_from_landuse(lands))

    # --- Roads → otomatik A_m2 ---
    road_len, roads_error = fetch_osm_roads(lat, lon)
    A_m2 = estimate_catchment_area(road_len, D, K)

    # --- Riskler ---
    C, W_eff, B, L, Flood = compute_risks(S, D, K, Wstar, Rext)

    # --- Sistem seçimi ---
    selected, scores = choose_system(S, D, K, C, W_eff, B, L, Flood)

    # --- Hidrolik ---
    i_mm_h = compute_idf_intensity(maxD)
    A_ha = A_m2 / 10000.0        # m² → hektar
    Q = 0.278 * C * i_mm_h * A_ha
    D_mm = manning_diameter(Q, 0.013, S_bed) * 1000
    velocity = (Q / (math.pi*(D_mm/1000)**2/4)) if D_mm > 0 else 0

    scale_class = classify_scale(D_mm, Q, A_m2)
    scale_name, scale_icon = classify_scale(D_mm, Q, A_m2)


    return jsonify({
        "selected_system": selected,
        "scores": scores,
        "slope_percent": slope_percent,
        "S": S,
        "building_count": bcount,
        "density_bld_per_km2": dens_km2,
        "D": D,
        "lands": lands,
        "K": K,
        "W_star": Wstar,
        "R_extreme": Rext,
        "C": C,
        "W_eff": W_eff,
        "B": B,
        "L": L,
        "FloodRisk": Flood,
        "FloodRiskLevel": (
            "Düşük" if Flood < 0.3 else
            "Orta" if Flood < 0.6 else
            "Yüksek" if Flood < 0.8 else
            "Çok Yüksek"
        ),
        "road_length_m": road_len,
        "catchment_area_m2": A_m2,
        "Q_m3_s": Q,
        "pipe_diameter_mm": D_mm,
        "velocity_m_s": velocity,
        "dem_error": dem_error,
        "rain_error": rain_error,
        "osm_error": osm_error,
        "roads_error": roads_error,
        "scale": scale_class,
        "scale": scale_name,
        "scale_icon": scale_icon,

    })

@app.route("/")
def home():
    return "Drenaj API v5.0 — Oto Havza + AHP + Manning"

if __name__ == "__main__":
    app.run()
application = app
