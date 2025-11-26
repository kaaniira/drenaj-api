# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK v7.1 (DÜZELTİLMİŞ)
#  K (Geçirgenlik) Sorunu Giderildi + 10m Çözünürlük Eklendi
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
from collections import Counter
import numpy as np
import ee
import json
import os

app = Flask(__name__)
CORS(app)

# Render'da "Secret Files" olarak eklediğin dosya yolu
# Eğer Render kullanmıyorsan kendi yerel yolunu yaz
SERVICE_ACCOUNT = "earthengine-service@drenaj-v6.iam.gserviceaccount.com"
KEY_PATH = "/etc/secrets/service-account.json" 

# Yerelde test ediyorsan hata almamak için kontrol
if not os.path.exists(KEY_PATH):
    print(f"UYARI: {KEY_PATH} bulunamadı. GEE çalışmayabilir.")
else:
    try:
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
        ee.Initialize(credentials)
        print("Google Earth Engine Başarıyla Başlatıldı.")
    except Exception as e:
        print(f"GEE Başlatma Hatası: {e}")

# ============================================================
#  DÜZELTİLMİŞ GEÇİRGENLİK (K) FONKSİYONU
#  Eski kodda "impervious" bandı yoktu, "urban-coverfraction" olmalıydı.
#  Ayrıca 100m yerine 10m çözünürlüklü Dynamic World kullanıyoruz.
# ============================================================

def get_impervious_K(lat, lon):
    try:
        # YÖNTEM 1: Dynamic World (10m Çözünürlük - En Hassas)
        # Güncel (2023-2024) veriyi alır.
        point = ee.Geometry.Point([lon, lat])
        
        # Son 1.5 yıldaki en temiz görüntüyü alalım
        dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(point) \
            .filterDate('2023-01-01', '2024-12-30') \
            .select('built') \
            .mean() # Olasılıkların ortalamasını al

        # 50 metrelik bir yarıçapta ortalama bina yoğunluğuna bak
        # (Sokak ölçeği için 150m çok genişti, 50m'ye düşürdük)
        region = point.buffer(50).bounds()
        
        value = dw.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=10,  # 10 metre hassasiyet
            maxPixels=1e9
        ).get("built").getInfo()

        # Eğer Dynamic World boş dönerse Copernicus'a (Yedek) geç
        if value is None:
            raise ValueError("Dynamic World verisi boş, yedeğe geçiliyor.")

        # value burada 0.0 ile 1.0 arasında bir "yapılaşma olasılığı"dır.
        # built (yapılaşma) = geçirimsizlik
        # K (Geçirgenlik) = 1 - built
        K = 1.0 - float(value)
        
        print(f"K Değeri (Dynamic World): {K}")
        return max(0.0, min(1.0, K))

    except Exception as e:
        print(f"Dynamic World Hatası ({e}), Copernicus'a (100m) geçiliyor...")
        try:
            # YÖNTEM 2: Copernicus (Yedek - 100m)
            # DÜZELTME: Band ismi 'urban-coverfraction' yapıldı.
            dataset = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
            impervious_layer = dataset.select("urban-coverfraction")
            
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(100).bounds()
            
            val_backup = impervious_layer.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=100,
                maxPixels=1e9
            ).get("urban-coverfraction").getInfo()
            
            if val_backup is None:
                return 0.5
            
            # urban-coverfraction 0-100 arası gelir, 100'e bölmeliyiz.
            K_backup = 1.0 - (float(val_backup) / 100.0)
            return max(0.0, min(1.0, K_backup))
            
        except Exception as e2:
            print("Tüm GEE kaynakları başarısız:", e2)
            return 0.5

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
    h1, err = get_elevation(lat, lon)
    if h1 is None:
        return None, err

    delta_deg = 100.0 / 111320.0  # ~100 m
    h2, err2 = get_elevation(lat + delta_deg, lon)
    if h2 is None:
        return None, err2

    dh = h2 - h1
    slope_percent = abs(dh) 
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
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        daily = r.json()["daily"]["precipitation_sum"]
        if not daily:
            return None, None, None, "Yağış verisi boş geldi"

        # None değerleri filtrele (bazen API null döndürebilir)
        daily = [d for d in daily if d is not None]
        
        if not daily:
             return 0, 0, 0, "Yağış verisi yetersiz"

        total = sum(daily)
        meanA = total / 10.0
        maxD = max(daily)
        sorted_p = sorted(daily)
        p99_index = int(0.99 * len(sorted_p))
        p99 = sorted_p[p99_index] if sorted_p else 0

        return meanA, maxD, p99, None
    except Exception as e:
        print("Yağış Hatası:", e)
        return None, None, None, "Yağış API hatası"

def compute_idf_intensity(max_daily):
    if not max_daily:
        return 0.0
    return (max_daily * 1.3) / ((15 + 12) ** 0.75)


# ============================================================
#  OSM: Binalar + Arazi Kullanımı 
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


def estimate_catchment_area(total_road_m, K):
    if total_road_m <= 0:
        return 30000.0 

    W_avg = 10.0 
    A_roads = total_road_m * W_avg * 1.3            
    A_final = A_roads * (1.0 + 0.5 * (1.0 - K))     
    return A_final


# ============================================================
#  SEL RİSKİ BLOKLARI
# ============================================================

def compute_blocks(S, K, W_star, R_extreme):
    C = 1.0 - K
    W_block = 0.65 * W_star + 0.35 * R_extreme
    S_flat = 1.0 - S

    FloodRisk_linear = (
        0.50 * C +
        0.30 * W_block +
        0.15 * S_flat
    )

    extreme_boost = max(0.0, R_extreme - 0.85) * 0.35
    FloodRisk = clamp(FloodRisk_linear + extreme_boost)

    return C, W_block, S_flat, FloodRisk


# ============================================================
#  AHP DRENAJ TİPİ SEÇİMİ
# ============================================================

def choose_system(S, K, C, FloodRisk):
    S_mid = 1.0 - abs(2.0 * S - 1.0)

    Score_DEN = 0.50 * S + 0.30 * FloodRisk + 0.20 * (1.0 - K)
    Score_PAR = 0.45 * (1.0 - S) + 0.30 * K + 0.25 * (1.0 - FloodRisk)
    Score_RET = 0.40 * C + 0.60 * FloodRisk
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
#  HİDROLİK HESAPLAR
# ============================================================

def manning_diameter(Q, n, S_bed):
    if Q <= 0 or S_bed <= 0:
        return 0.0
    num = (4.0 ** (5.0 / 3.0)) * n * Q
    den = math.pi * math.sqrt(S_bed)
    return (num / den) ** (3.0 / 8.0)

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


# ============================================================
#  ANA API ENDPOINT
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    lat = float(data["lat"])
    lon = float(data["lon"])

    # 1) EĞİM
    slope_percent, dem_error = estimate_slope_percent(lat, lon)
    if slope_percent is None:
        slope_percent = 0.0
    S = clamp(slope_percent / 30.0)
    raw_bed_slope = (slope_percent or 0.0) / 100.0
    S_bed = max(0.003, min(raw_bed_slope, 0.03))

    # 2) YAĞIŞ
    meanA, maxD, p99, rain_error = fetch_precip(lat, lon)
    if meanA is None:
        W_star = 0.5
    else:
        W_star = clamp(meanA / 1000.0)
    
    if maxD is None or p99 is None:
        R_extreme = 0.5
    else:
        R_extreme = clamp(0.6 * (maxD / 150.0) + 0.4 * (p99 / 80.0))

    # 3) GEÇİRGENLİK (K) - ARTIK DÜZELTİLDİ
    K = get_impervious_K(lat, lon)
    
    # 4) HAVZA
    road_len, roads_error = fetch_osm_roads(lat, lon)
    A_m2 = estimate_catchment_area(road_len, K)
    A_ha = A_m2 / 10000.0

    # 5) RİSK & SİSTEM
    C, W_block, S_flat, FloodRisk = compute_blocks(S, K, W_star, R_extreme)
    selected, scores, S_mid = choose_system(S, K, C, FloodRisk)

    # 6) HİDROLİK
    i_mm_h = compute_idf_intensity(maxD) if maxD is not None else 0.0
    Q = 0.278 * C * i_mm_h * A_ha
    D_m = manning_diameter(Q, n=0.013, S_bed=S_bed)
    D_mm = D_m * 1000.0
    area_pipe = math.pi * (D_m ** 2) / 4.0 if D_m > 0 else 0.0
    velocity = Q / area_pipe if area_pipe > 0 else 0.0

    scale_name, scale_icon = classify_scale(D_mm, Q, A_m2)
    material = recommend_material(D_mm, velocity, Q)

    if FloodRisk < 0.20: FloodRiskLevel = "Çok Düşük"
    elif FloodRisk < 0.40: FloodRiskLevel = "Düşük"
    elif FloodRisk < 0.60: FloodRiskLevel = "Orta"
    elif FloodRisk < 0.75: FloodRiskLevel = "Yüksek"
    else: FloodRiskLevel = "Çok Yüksek"

    return jsonify({
        "selected_system": selected,
        "scores": scores,
        "slope_percent": slope_percent,
        "S": S,
        "K": K, # Artık doğru değer dönecek
        "C": C,
        "W_star": W_star,
        "R_extreme": R_extreme,
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
        "errors": {
            "dem": dem_error,
            "rain": rain_error,
            "osm": roads_error
        }
    })

@app.route("/")
def home():
    return "Drenaj API v7.1 — Google Earth Engine Fix"

if __name__ == "__main__":
    app.run()

application = app
