# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK V6.0
#  Son formüle göre güncellenmiş API
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
    except Exception:
        errors.append("Open-Meteo başarısız.")
    try:
        return get_elev_openelev(lat, lon), None
    except Exception:
        errors.append("Open-Elevation başarısız.")
    return None, " / ".join(errors) if errors else "DEM kaynağı hatası"

def estimate_slope_percent(lat, lon):
    """
    ~100 m kuzeye gidip yükseklik farkından eğimi hesaplıyoruz.
    Eğim % cinsinden.
    """
    h1, err = get_elevation(lat, lon)
    if h1 is None:
        return None, err

    delta_deg = 100.0 / 111320.0  # ~100 m enlem farkı
    h2, err2 = get_elevation(lat + delta_deg, lon)
    if h2 is None:
        return None, err2

    dh = h2 - h1
    slope_percent = abs(dh)  # 100 m’ye bölünmüş olduğu için ~% olarak alıyoruz
    return slope_percent, None


# ============================================================
#  YAĞIŞ (10 YIL) — Open-Meteo Archive
# ============================================================

def fetch_precip(lat, lon):
    """
    2015-01-01 – 2024-12-31 arası günlük toplam yağış
    """
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            "&start_date=2015-01-01&end_date=2024-12-31"
            "&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        daily = r.json()["daily"]["precipitation_sum"]
        if not daily:
            return None, None, None, "Yağış verisi boş geldi"

        total = sum(daily)
        meanA = total / 10.0
        maxD = max(daily)
        sorted_p = sorted(daily)
        p99_index = int(0.99 * len(sorted_p))
        p99 = sorted_p[p99_index]

        return meanA, maxD, p99, None
    except Exception:
        return None, None, None, "Yağış API hatası"

def compute_idf_intensity(max_daily):
    """
    Basit IDF yaklaşımı: 27 dakikalık kısa süreli şiddete indirgeme.
    max_daily mm/gün --> mm/saat civarı
    """
    if not max_daily:
        return 0.0
    return (max_daily * 1.3) / ((15 + 12) ** 0.75)


# ============================================================
#  OSM: Binalar + Arazi Kullanımı
# ============================================================

def fetch_osm(lat, lon, radius=200):
    """
    radius m yarıçaplı alanda building ve landuse etiketleri
    """
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
        elements = r.json().get("elements", [])
    except Exception:
        return 0, [], "OSM API hatası"

    buildings = 0
    lands = []
    for el in elements:
        tags = el.get("tags", {})
        if "building" in tags:
            buildings += 1
        if "landuse" in tags:
            lands.append(tags["landuse"])

    return buildings, lands, None


# ============================================================
#  YOĞUNLUK NORMALİZASYONU (Türkiye Kalibrasyonu)
# ============================================================

def normalize_density_turkey(dens_bld_per_km2):
    """
    500 bina/km²: düşük (0)
    5000 bina/km²: çok yüksek (1)
    Arası lineer.
    """
    low = 500.0
    high = 5000.0
    if dens_bld_per_km2 <= low:
        return 0.0
    if dens_bld_per_km2 >= high:
        return 1.0
    return (dens_bld_per_km2 - low) / (high - low)


# ============================================================
#  GEÇİRGENLİK: Landuse → K
# ============================================================

def permeability_from_landuse(lands):
    """
    Arazi kullanımına göre yaklaşık geçirgenlik.
    Değerler DSİ/YTDDSHY'deki C aralıklarının tersine göre ayarlandı.
    """
    if not lands:
        return 0.5
    mc = Counter(lands).most_common(1)[0][0]
    table = {
        "forest": 0.85,
        "meadow": 0.80,
        "grass": 0.80,
        "greenfield": 0.75,
        "farmland": 0.60,
        "orchard": 0.60,
        "residential": 0.35,
        "commercial": 0.30,
        "industrial": 0.25
    }
    return table.get(mc, 0.5)


# ============================================================
#  OSM ROADS → DİNAMİK HAVZA ALANI (A_m2)
# ============================================================

def fetch_osm_roads(lat, lon, radius=200):
    """
    radius m yarıçaplı alanda highway objelerinin toplam uzunluğu
    """
    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius},{lat},{lon})["highway"];
    );
    out geom;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=30)
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except Exception:
        return 0.0, "Roads API hatası"

    total_len = 0.0
    for el in elements:
        geom = el.get("geometry", [])
        for i in range(len(geom) - 1):
            lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
            lat2, lon2 = geom[i + 1]["lat"], geom[i + 1]["lon"]

            dx = (lon2 - lon1) * 85000.0
            dy = (lat2 - lat1) * 111320.0
            total_len += math.hypot(dx, dy)

    return total_len, None


def estimate_catchment_area(total_road_m, D, K):
    """
    Yol uzunluğuna, yoğunluğa ve geçirgenliğe göre etkili havza alanı.
    """
    if total_road_m <= 0:
        return 30000.0  # ~3 ha varsayılan

    W_avg = 10.0  # ortalama yol genişliği
    A_roads = total_road_m * W_avg * 1.3          # yol yüzeyi
    A_density = A_roads * (1.0 + 0.6 * D)        # bina yoğunluğu çarpanı
    A_final = A_density * (1.0 + 0.5 * (1.0 - K))  # geçirimsizlik çarpanı
    return A_final


# ============================================================
#  SEL RİSKİ BLOKLARI (SON FORMÜL)
# ============================================================

def compute_blocks(S, D, K, W_star, R_extreme):
    """
    Son risk formülündeki blokların hesaplanması:
    - W_block
    - C (kentsel)
    - S_flat
    - FloodRisk
    """
    # Kentsel etki
    C = 0.5 * D + 0.5 * (1.0 - K)

    # Yağış bloğu
    W_block = 0.6 * W_star + 0.4 * R_extreme

    # Düzlük (düşük eğim) etkisi
    S_flat = 1.0 - S

    # Nihai sel riski (0–1)
    FloodRisk = 0.40 * W_block + 0.35 * C + 0.25 * S_flat

    return C, W_block, S_flat, FloodRisk


# ============================================================
#  AHP DRENAJ TİPİ SEÇİMİ (SON KATSAYILAR)
# ============================================================

def choose_system(S, D, K, C, FloodRisk):
    """
    Dendritik / Paralel / Retiküler / Hibrit skorları
    """

    # Orta eğimi vurgulayan terim
    S_mid = 1.0 - abs(2.0 * S - 1.0)

    # Dendritik: eğim + risk + geçirimsizlik
    Score_DEN = 0.50 * S + 0.30 * FloodRisk + 0.20 * (1.0 - K)

    # Paralel: düşük eğim + geçirgenlik + düşük risk
    Score_PAR = 0.45 * (1.0 - S) + 0.30 * K + 0.25 * (1.0 - FloodRisk)

    # Retiküler: yoğunluk + kentsel etki + risk
    Score_RET = 0.50 * D + 0.30 * C + 0.20 * FloodRisk

    # Hibrit: orta eğim + risk + kentsel etki
    Score_HYB = 0.35 * FloodRisk + 0.35 * C + 0.30 * S_mid

    scores = {
        "dendritic": Score_DEN,
        "parallel": Score_PAR,
        "reticular": Score_RET,
        "hybrid": Score_HYB
    }

    selected = max(scores, key=scores.get)
    return selected, scores, S_mid


# ============================================================
#  MANNING BORU ÇAPI
# ============================================================

def manning_diameter(Q, n, S_bed):
    """
    D = ((4^(5/3) * n * Q) / (pi * sqrt(S)))^(3/8)
    """
    if Q <= 0 or S_bed <= 0:
        return 0.0
    num = (4.0 ** (5.0 / 3.0)) * n * Q
    den = math.pi * math.sqrt(S_bed)
    return (num / den) ** (3.0 / 8.0)


# ============================================================
#  ÖLÇEK / MALZEME SINIFLANDIRMASI
# ============================================================

def classify_scale(D_mm, Q, A_m2):
    """
    Basit mühendislik ölçek sınıflandırması
    """
    A_ha = A_m2 / 10000.0

    # Street Drain
    if D_mm < 500 and Q < 1.5 and A_ha < 3:
        return "Sokak Hattı", "🟩"

    # Secondary Collector
    if (500 <= D_mm < 1000) or (1.5 <= Q < 5) or (3 <= A_ha < 10):
        return "Mahalle Kolektörü", "🟨"

    # Major Trunk
    return "Ana Kolektör / Trunk", "🟥"

def recommend_material(D_mm, velocity, Q):
    """
    Çapa göre kabaca malzeme önerisi.
    """
    if D_mm >= 1200:
        return "GRP (Cam Elyaf Takviyeli Polyester)"
    if 600 <= D_mm < 1200:
        return "Betonarme Boru"
    if 200 <= D_mm < 600:
        return "PE100 / HDPE"
    return "PVC veya PP"


# ============================================================
#  ANA API ENDPOINT
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    lat = float(data["lat"])
    lon = float(data["lon"])

    # --------------------------------------------------------
    # 1) EĞİM
    # --------------------------------------------------------
    slope_percent, dem_error = estimate_slope_percent(lat, lon)
    if slope_percent is None:
        slope_percent = 0.0

    # Eğim skoru S (0–1, 30% üstü doyuyor)
    S = clamp(slope_percent / 30.0)

    # Manning için taban eğimi (m/m) – 0.3% ile 3% arasına sıkıştır
    raw_bed_slope = (slope_percent or 0.0) / 100.0
    S_bed = max(0.003, min(raw_bed_slope, 0.03))

    # --------------------------------------------------------
    # 2) YAĞIŞ
    # --------------------------------------------------------
    meanA, maxD, p99, rain_error = fetch_precip(lat, lon)

    if meanA is None:
        W_star = 0.5
    else:
        W_star = clamp(meanA / 1000.0)  # 1000 mm/yıl ve üzeri doyum

    if maxD is None or p99 is None:
        R_extreme = 0.5
    else:
        R_extreme = clamp(0.6 * (maxD / 150.0) + 0.4 * (p99 / 80.0))

    # --------------------------------------------------------
    # 3) OSM BİNA & LANDUSE
    # --------------------------------------------------------
    bcount, lands, osm_error = fetch_osm(lat, lon)
    area_km2 = math.pi * (0.2 ** 2)  # 200 m yarıçaplı daire
    dens_km2 = bcount / area_km2 if area_km2 > 0 else 0.0
    D = normalize_density_turkey(dens_km2)
    K = clamp(permeability_from_landuse(lands))

    # --------------------------------------------------------
    # 4) ROADS → HAVZA ALANI
    # --------------------------------------------------------
    road_len, roads_error = fetch_osm_roads(lat, lon)
    A_m2 = estimate_catchment_area(road_len, D, K)
    A_ha = A_m2 / 10000.0

    # --------------------------------------------------------
    # 5) RİSK BLOKLARI ve FLOODRISK
    # --------------------------------------------------------
    C, W_block, S_flat, FloodRisk = compute_blocks(S, D, K, W_star, R_extreme)

    # --------------------------------------------------------
    # 6) SİSTEM SEÇİMİ (AHP)
    # --------------------------------------------------------
    selected, scores, S_mid = choose_system(S, D, K, C, FloodRisk)

    # --------------------------------------------------------
    # 7) HİDROLİK (Q, D_mm, hız)
    # --------------------------------------------------------
    i_mm_h = compute_idf_intensity(maxD) if maxD is not None else 0.0
    Q = 0.278 * C * i_mm_h * A_ha
    D_m = manning_diameter(Q, n=0.013, S_bed=S_bed)
    D_mm = D_m * 1000.0
    area_pipe = math.pi * (D_m ** 2) / 4.0 if D_m > 0 else 0.0
    velocity = Q / area_pipe if area_pipe > 0 else 0.0

    scale_name, scale_icon = classify_scale(D_mm, Q, A_m2)
    material = recommend_material(D_mm, velocity, Q)

    # --------------------------------------------------------
    # 8) FLOODRISK SEVİYE METNİ
    # --------------------------------------------------------
    if FloodRisk < 0.3:
        FloodRiskLevel = "Düşük"
    elif FloodRisk < 0.6:
        FloodRiskLevel = "Orta"
    elif FloodRisk < 0.8:
        FloodRiskLevel = "Yüksek"
    else:
        FloodRiskLevel = "Çok Yüksek"

    # --------------------------------------------------------
    # 9) JSON ÇIKTI
    # --------------------------------------------------------
    return jsonify({
        "selected_system": selected,
        "scores": scores,

        "slope_percent": slope_percent,
        "S": S,
        "S_flat": S_flat,
        "S_mid": S_mid,

        "building_count": bcount,
        "density_bld_per_km2": dens_km2,
        "D": D,
        "lands": lands,
        "K": K,

        "W_star": W_star,
        "R_extreme": R_extreme,
        "W_block": W_block,

        "C": C,
        "FloodRisk": FloodRisk,
        "FloodRiskLevel": FloodRiskLevel,

        "road_length_m": road_len,
        "catchment_area_m2": A_m2,

        "Q_m3_s": Q,
        "pipe_diameter_mm": D_mm,
        "velocity_m_s": velocity,

        "scale_name": scale_name,
        "scale_icon": scale_icon,
        "material": material,

        "dem_error": dem_error,
        "rain_error": rain_error,
        "osm_error": osm_error,
        "roads_error": roads_error
    })


@app.route("/")
def home():
    return "Drenaj API v6.0 — Biyomimikri + AHP + Manning (Son formül)"


if __name__ == "__main__":
    app.run()

# Render / gunicorn için:
application = app
