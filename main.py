# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK V6.0 (Final API)
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
from collections import Counter

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ------------------------------------------------------------

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def safe(v, default=None):
    return v if (v is not None and v == v) else default

# ------------------------------------------------------------
# Eğim Hesabı (Open-Meteo + OpenElevation fallback)
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 10 Yıllık Yağış Verisi — Open-Meteo Archive
# ------------------------------------------------------------

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



# ------------------------------------------------------------
# OSM — Yapılar + Landuse
# ------------------------------------------------------------

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
            tags = el.get("tags", {})
            if "building" in tags:
                buildings += 1
            if "landuse" in tags:
                lands.append(tags["landuse"])

        return buildings, lands, None
    except:
        return 0, [], "OSM API hatası"


# ------------------------------------------------------------
# Yoğunluk — Türkiye Kalibrasyonu
# ------------------------------------------------------------

def normalize_density_turkey(dens):
    low = 300
    high = 3000
    if dens <= low:
        return 0
    if dens >= high:
        return 1
    return (dens - low) / (high - low)


# ------------------------------------------------------------
# Landuse → Geçirgenlik
# ------------------------------------------------------------

def permeability_from_landuse(lands):
    if not lands:
        return 0.5
    mc = Counter(lands).most_common(1)[0][0]
    table = {
        "forest": 0.85,
        "meadow": 0.80,
        "grass": 0.80,
        "greenfield": 0.70,
        "farmland": 0.60,
        "orchard": 0.55,
        "residential": 0.35,
        "commercial": 0.30,
        "industrial": 0.25
    }
    return table.get(mc, 0.50)


# ------------------------------------------------------------
# OSM Roads → Havza Alanı
# ------------------------------------------------------------

def fetch_osm_roads(lat, lon, radius=200):
    query = f"""
    [out:json][timeout:25];
    way(around:{radius},{lat},{lon})["highway"];
    out geom;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=30)
        r.raise_for_status()
        elements = r.json()["elements"]

        total_len = 0.0

        for el in elements:
            geom = el.get("geometry", [])
            for i in range(len(geom) - 1):
                lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
                lat2, lon2 = geom[i+1]["lat"], geom[i+1]["lon"]

                dx = (lon2 - lon1) * 85000
                dy = (lat2 - lat1) * 111320
                total_len += (dx*dx + dy*dy)**0.5

        return total_len, None
    except:
        return 0, "Roads API hatası"


def estimate_catchment_area(total_road_m, D, K):
    if total_road_m <= 0:
        return 30000

    A_roads = total_road_m * 10 * 1.25
    A_density = A_roads * (1 + 0.55 * D)
    A_final = A_density * (1 + 0.45*(1-K))
    return A_final


# ------------------------------------------------------------
# Yeni Sel Riski v3.0
# ------------------------------------------------------------

def compute_risks_advanced(S, D, K, Wstar, Rext):

    C = 0.45*D + 0.55*(1-K)

    W_eff = Wstar * C

    S_risk = clamp(4 * (S - 0.5)**2)

    B = clamp(0.65*C + 0.35*K*(1-S))

    Flood = clamp(
        0.32 * W_eff +
        0.26 * Rext +
        0.22 * B +
        0.12 * C +
        0.08 * S_risk
    )

    return C, W_eff, B, S_risk, Flood


# ------------------------------------------------------------
# AHP Sistem Seçimi
# ------------------------------------------------------------

def choose_system(S, D, K, C, W_eff, B, S_risk, Flood):
    Score_DEN = 0.45*S + 0.30*W_eff + 0.25*(1-K)
    Score_PAR = 0.45*(1-S) + 0.30*K + 0.25*(1-Flood)
    Score_RET = 0.50*D + 0.30*(1-K) + 0.20*W_eff
    Score_HYB = 0.40*(1 - abs(2*S - 1)) + 0.35*Flood + 0.25*(1-K)

    scores = {
        "dendritic": Score_DEN,
        "parallel": Score_PAR,
        "reticular": Score_RET,
        "hybrid": Score_HYB
    }

    selected = max(scores, key=scores.get)
    return selected, scores


# ------------------------------------------------------------
# Manning Boru Çapı
# ------------------------------------------------------------

def manning_diameter(Q, n, S):
    if Q <= 0 or S <= 0:
        return 0
    return ((4**(5/3) * n * Q) / (math.pi * math.sqrt(S)))**(3/8)


# ------------------------------------------------------------
# Boru Kategorisi
# ------------------------------------------------------------

def classify_scale(D_mm, Q, A_m2):
    A_ha = A_m2 / 10000.0

    if D_mm < 500 and Q < 1.5 and A_ha < 3:
        return "Sokak Hattı", "🟩"

    if (500 <= D_mm < 1000) or (1.5 <= Q < 5) or (3 <= A_ha < 10):
        return "Mahalle Kolektörü", "🟨"

    return "Ana Kolektör / Trunk", "🟥"


def recommend_material(D_mm, velocity, Q):
    if D_mm >= 1200:
        return "GRP (Cam Elyaf Takviyeli Polyester)"
    if 600 <= D_mm < 1200:
        return "Betonarme Boru"
    if 200 <= D_mm < 600:
        return "PE100 / HDPE"
    return "PVC veya PP"


# ------------------------------------------------------------
# Açıklama Motoru
# ------------------------------------------------------------

def explanation_text(selected, S, D, K, Flood):

    parts = []

    if S > 0.5:
        parts.append("Eğim yüksek olduğu için su doğal olarak dallanarak akar, bu nedenle dendritik yapı avantaj sağlar.")
    else:
        parts.append("Eğim düşük olduğu için akış doğrusal davranır, bu da paralel hatları daha verimli kılar.")

    if D > 0.6:
        parts.append("Bina yoğunluğu yüksek olduğu için retiküler ağ tipi sistem güçlü bir alternatifti.")
    else:
        parts.append("Yoğunluk düşük olduğundan geniş ağ yapısına gerek duyulmadı.")

    if K < 0.4:
        parts.append("Yer geçirgenliği düşük olduğundan yüzey akışı artıyor, bu da sistemin daha güçlü olması gerektiğini gösteriyor.")

    if Flood > 0.6:
        parts.append("Sel riski yüksek olduğu için kapasitesi büyük borular tercih edilmelidir.")

    final = f"Bu bölgede **{selected.upper()}** sistemi seçilmiştir. " + " ".join(parts)
    return final


# ------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    lat = float(data["lat"])
    lon = float(data["lon"])

    # Eğim
    slope_percent, dem_error = estimate_slope_percent(lat, lon)
    S = clamp((slope_percent / 30)) if slope_percent else 0.0
    S_bed = max(0.003, min((slope_percent or 0)/100.0, 0.03))

    # Yağış
    meanA, maxD, p99, rain_error = fetch_precip(lat, lon)
    Wstar = clamp(meanA / 1000) if meanA else 0.5
    Rext = clamp(0.6*(safe(maxD,0)/150) + 0.4*(safe(p99,0)/80))

    # OSM — Bina + Landuse
    bcount, lands, osm_error = fetch_osm(lat, lon)
    area_km2 = math.pi * (0.2**2)
    dens_km2 = bcount / area_km2 if area_km2 > 0 else 0
    D = normalize_density_turkey(dens_km2)
    K = clamp(permeability_from_landuse(lands))

    # OSM — Roads → Havza Alanı
    road_len, roads_error = fetch_osm_roads(lat, lon)
    A_m2 = estimate_catchment_area(road_len, D, K)

    # Yeni Sel Riski
    C, W_eff, B, S_risk, Flood = compute_risks_advanced(S, D, K, Wstar, Rext)

    # Sistem seçimi
    selected, scores = choose_system(S, D, K, C, W_eff, B, S_risk, Flood)

    # Hidrolik
    i_mm_h = compute_idf_intensity(maxD)
    A_ha = A_m2 / 10000
    Q = 0.278 * C * i_mm_h * A_ha
    D_mm = manning_diameter(Q, 0.013, S_bed) * 1000
    velocity = (Q / (math.pi*(D_mm/1000)**2/4)) if D_mm > 0 else 0

    scale_name, scale_icon = classify_scale(D_mm, Q, A_m2)
    material = recommend_material(D_mm, velocity, Q)

    explanation = explanation_text(selected, S, D, K, Flood)

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
        "S_risk": S_risk,
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
        "scale_name": scale_name,
        "scale_icon": scale_icon,
        "material": material,
        "explanation": explanation,
        "dem_error": dem_error,
        "rain_error": rain_error,
        "osm_error": osm_error,
        "roads_error": roads_error
    })


@app.route("/")
def home():
    return "Drenaj API v6.0 — Yeni Sel Riski + Açıklama Motoru"

application = app
