# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v16.0 (FINAL MASTER)
# ============================================================

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import math
import ee
import os
import google.auth
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- CORS AYARLARI (GARANTİ ERİŞİM) ---
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# --- GEE BAŞLATMA ---
def initialize_gee():
    try:
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
        )
        ee.Initialize(credentials, project=project)
        logging.info("GEE Başlatıldı.")
    except Exception as e:
        logging.error(f"GEE Başlatma Hatası: {e}")

initialize_gee()

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

# --- GEE VERİ ÇEKME GÖREVİ (PARALEL ÇALIŞIR) ---
def get_gee_data_task(geometry):
    """
    Bu fonksiyon 10m hassasiyetle verileri çeker.
    Büyük alanlarda hata vermemesi için tileScale=8 kullanır.
    """
    try:
        # 1. Sentinel-2 (NDVI)
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
              .filterBounds(geometry)
              .filterDate("2023-06-01", "2023-09-30")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
              .median())
        ndvi_img = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")

        # 2. ESA WorldCover (Arazi Örtüsü & K Katsayısı)
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        k_img = dataset.remap(
            [10, 20, 30, 40, 50, 60, 80, 90, 95, 100], 
            [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90], 
            0.5
        ).rename("k_value")
        land_cls_img = dataset.rename("land_class")

        # 3. Toprak ve DEM (Eğim/Yükseklik)
        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil_type")
        dem = ee.Image("USGS/SRTMGL1_003")
        
        # 4. Hepsini Birleştir
        combined = ee.Image.cat([
            ndvi_img, k_img, land_cls_img, soil_img, 
            ee.Terrain.slope(dem).rename("slope"), 
            dem.select("elevation").rename("elevation")
        ])
        
        # 5. İsteği Gönder (10 Metre Scale)
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=geometry, 
            scale=10,       # <-- 10 Metre Sabit Hassasiyet
            bestEffort=True, 
            maxPixels=1e9,
            tileScale=8     # Bellek hatasını önler
        ).getInfo()

        if not stats: return None

        return {
            "ndvi": float(stats.get("ndvi", 0.0)),
            "k": float(stats.get("k_value", 0.5)),
            "slope": float(stats.get("slope", 0.0)),
            "elev": float(stats.get("elevation", 0.0)),
            "land": round(stats.get("land_class", 0)),
            "soil": round(stats.get("soil_type", 0))
        }

    except Exception as e:
        logging.error(f"GEE Task Hatası: {e}")
        return None

# --- HAVA DURUMU GÖREVİ (PARALEL ÇALIŞIR) ---
def get_weather_data_task(lat, lon):
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=UTC"
        r = requests.get(url, timeout=5).json()
        clean = [d for d in r.get("daily", {}).get("precipitation_sum", []) if d is not None]
        
        if not clean: return 0.5, 0.5, 50.0, 500.0
        
        meanA = sum(clean) / 10.0
        maxD = max(clean)
        return clamp(meanA/1000.0), clamp(maxD/120.0), maxD, meanA
    except Exception as e:
        logging.error(f"Weather Task Hatası: {e}")
        return 0.5, 0.5, 50.0, 500.0

# --- ANA ENDPOINT ---
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    # Preflight Check (CORS için)
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        d = request.get_json(force=True)
        mode = d.get("mode", "point")
        radius = float(d.get("radius", 100.0))

        # --- GEOMETRİ OLUŞTURMA ---
        if mode == "line":
            start, end = d["start"], d["end"]
            line = ee.Geometry.LineString([[start["lon"], start["lat"]], [end["lon"], end["lat"]]])
            ee_geometry = line.buffer(radius)
            center_lat, center_lon = (start["lat"] + end["lat"])/2.0, (start["lon"] + end["lon"])/2.0
            L_flow = line.length().getInfo() 
            analysis_area_m2 = L_flow * (radius * 2)
        else:
            lat, lon = float(d["lat"]), float(d["lon"])
            center_lat, center_lon = lat, lon
            ee_geometry = ee.Geometry.Point([lon, lat]).buffer(radius)
            analysis_area_m2 = math.pi * (radius ** 2.0)
            L_flow = radius * 2.0

        # --- PARALEL İŞLEM BAŞLATMA (HIZLANDIRICI) ---
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_gee = executor.submit(get_gee_data_task, ee_geometry)
            future_weather = executor.submit(get_weather_data_task, center_lat, center_lon)

            # Sonuçları Bekle
            gee_res = future_gee.result()
            weather_res = future_weather.result()

        # --- KRİTİK KONTROL: VERİ GELDİ Mİ? ---
        if not gee_res:
            # Eğer GEE hiç cevap vermediyse (Timeout veya boş veri), 
            # bunu "Su" sanma, direkt hata ver.
            logging.error("GEE verisi boş döndü (Timeout olabilir).")
            return jsonify({
                "status": "error", 
                "msg": "Sunucu yoğunluğu veya bulutluluk nedeniyle harita verisi alınamadı. Lütfen tekrar deneyin."
            }), 500

        # --- VERİLERİ İŞLE ---
        ndvi = gee_res["ndvi"]
        K_cover = 1.0 - clamp((1.0 - gee_res["k"]) * 1.0)
        
        land_map = {10: "Ormanlık", 20: "Çalılık", 30: "Çayır/Park", 40: "Tarım", 50: "Kentsel/Beton", 60: "Çıplak", 80: "Su"}
        land_type = land_map.get(int(round(gee_res["land"]/10)*10), "Karma Alan")
        
        soil_factor, soil_desc = 1.0, "Normal Toprak"
        if gee_res["soil"] in [1, 2, 3]: soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
        elif gee_res["soil"] in [9, 10, 11, 12]: soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"
        
        slope_pct = gee_res["slope"] * 1.5
        elevation = gee_res["elev"]

        W_star, R_ext, maxRain, meanRain = weather_res

        # --- SU KONTROLÜ (İYİLEŞTİRİLMİŞ) ---
        # Sadece GEE kesin "Su" dediyse VEYA rakım 0'ın altındaysa su kabul et.
        # Ama eğer burası Kentsel/Beton ise, su uyarısını yoksay (Gölge olabilir).
        is_water_strict = land_type == "Su" or elevation <= 0
        is_urban_or_agriculture = land_type in ["Kentsel/Beton", "Tarım", "Karma Alan"]
        
        if is_water_strict and not is_urban_or_agriculture:
             return jsonify({
                 "status": "success", 
                 "location_type": "Su Kütlesi", 
                 "selected_system": "hybrid", 
                 "material": "-", 
                 "system_reasoning": "Analiz edilen alan su kütlesi (deniz, göl, nehir) olarak tespit edildi.", 
                 "scores": {}, 
                 "eco_stats": {"harvest":0, "bio_solution":"-"}, 
                 "rain_stats": {"mean":0, "max":0}, 
                 "Q_flow": 0, "pipe_diameter_mm": 0, "FloodRisk": 0, "FloodRiskLevel": "N/A", 
                 "K_value": 0, "C_value": 0, "ndvi": 0, "slope_percent": 0, "debug_analysis_area_ha": 0
             })

        # --- HESAPLAMALAR ---
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - (ndvi * 0.30) if ndvi > 0.2 else 1.0
        
        # C KATSAYISI
        C = clamp((1.0 - K_cover) * soil_factor * veg_factor)
        K_final = 1.0 - C
        
        FloodRisk = clamp((0.45*(0.6*W_star+0.4*R_ext) + 0.45*C + 0.1*abs(2*S-1)) + max(0, R_ext-0.75)*0.4 + (1-W_star)*0.3)
        S_mid = 1.0 - abs(2.0 * S - 1.0)

        scores = {
            "dendritic": round(0.40 * S_mid + 0.40 * FloodRisk + 0.20 * K_final, 3),
            "parallel": round(0.50 * S + 0.30 * K_final + 0.20 * (1 - FloodRisk), 3),
            "reticular": round(0.80 * C + 0.20 * FloodRisk, 3),
            "pinnate": round(0.50 * S + 0.30 * C + 0.20 * W_star, 3),
            "radial": round(0.70 * (1.0 - S) + 0.20 * K_final + 0.10 * FloodRisk, 3),
            "meandering": round(0.80 * S + 0.20 * (1 - C), 3),
            "hybrid": round(0.35 * FloodRisk + 0.35 * C + 0.30 * S_mid, 3),
        }
        selected = max(scores, key=scores.get)
        
        S_metric = max(0.01, slope_pct / 100.0)
        t_c = max(5.0, min(45.0, 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))))
        
        # IDF (Türkiye/Global)
        is_turkey = (35.5 <= center_lat <= 42.5) and (25.0 <= center_lon <= 45.0)
        if is_turkey:
             i_val = (maxRain / (24.0 ** 0.46)) * ((max(t_c/60.0, 0.1)) ** -0.54)
        else:
             i_val = 0.12 * ((maxRain * 3.0) / ((max(t_c/60.0, 0.1) + 0.15) ** 0.7))

        Q_future = (0.278 * C * i_val * (analysis_area_m2/1e6)) * 1.15
        n_rough = 0.025 if selected in ["meandering", "radial"] else 0.013
        D_mm = (((4**(5/3)) * n_rough * Q_future) / (math.pi * math.sqrt(max(0.005, S_metric))))**(3/8) * 1000.0

        mat = "PVC"
        if selected in ["meandering", "radial"]: mat = "Doğal Taş"
        elif D_mm >= 500: mat = "Betonarme"
        elif D_mm >= 200: mat = "Koruge (HDPE)"
        
        bio = "Standart Peyzaj"
        if C > 0.7: bio = "Yeşil Çatı"
        elif slope_pct > 15: bio = "Teraslama"
        elif selected == "radial": bio = "Yağmur Bahçesi"
        elif selected == "dendritic": bio = "Biyo-Hendek"

        reasons = {
            "dendritic": f"Eğim (%{slope_pct:.1f}) ve vadi yapısı.",
            "parallel": f"Düzenli tek yönlü eğim (%{slope_pct:.1f}).",
            "reticular": "Yüksek geçirimsizlik ve kentsel doku.",
            "pinnate": "Dar alan ve dik eğim.",
            "radial": "Merkezi toplanma havzası.",
            "meandering": "Erozyonu önlemek için kıvrımlı yapı.",
            "hybrid": "Karmaşık topografya."
        }

        return jsonify({
            "status": "success",
            "location_type": f"{land_type} ({soil_desc})",
            "ndvi": round(ndvi, 2),
            "slope_percent": round(slope_pct, 2),
            "K_value": round(K_final, 2),
            "C_value": round(C, 2), # C DEĞERİ EKLENDİ
            "FloodRisk": round(FloodRisk, 2),
            "FloodRiskLevel": ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Kritik"][min(int(FloodRisk*4.9), 4)],
            "selected_system": selected,
            "system_reasoning": reasons.get(selected, "Analiz edildi."),
            "scores": scores,
            "pipe_diameter_mm": round(D_mm, 0),
            "material": mat,
            "Q_flow": round(Q_future, 3),
            "rain_stats": {"mean": round(meanRain, 1), "max": round(maxRain, 1)},
            "eco_stats": {"harvest": round(analysis_area_m2*(meanRain/1000)*0.85*(1-K_final), 0), "bio_solution": bio},
            "debug_analysis_area_ha": round(analysis_area_m2/10000.0, 2)
        })

    except Exception as e:
        logging.error(f"ANALIZ HATASI: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

application = app
