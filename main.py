# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v14.0 (TURBO & DYNAMIC SCALE)
# ============================================================

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import math
import ee
import os
import google.auth
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# CORS: Her yerden erişime izin ver
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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

# --- OPTİMİZE EDİLMİŞ VERİ ÇEKME ---
def get_all_map_data(geometry, scale_val):
    """
    scale_val parametresi ile çözünürlüğü dinamik ayarlar.
    Line modunda 30m, Point modunda 10m çalışır.
    """
    try:
        # 1. Görüntüleri Hazırla (Lazy Evaluation - Henüz işlem yapmaz)
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
              .filterBounds(geometry)
              .filterDate("2023-06-01", "2023-09-30")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
              .median())
        ndvi_img = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")

        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        from_cls = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k = [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        k_img = dataset.remap(from_cls, to_k, 0.5).rename("k_value")
        land_cls_img = dataset.rename("land_class")

        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil_type")
        
        dem = ee.Image("USGS/SRTMGL1_003")
        elev_img = dem.select("elevation").rename("elevation")
        slope_img = ee.Terrain.slope(dem).rename("slope")

        # 2. Hepsini Paketle
        combined = ee.Image.cat([ndvi_img, k_img, land_cls_img, soil_img, slope_img, elev_img])
        
        # 3. Sunucuya Gönder (CRITICAL STEP)
        # bestEffort=True: Eğer alan çok büyükse ve 30m yetmezse, GEE otomatik olarak ölçeği büyütüp hata vermeden cevap döner.
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=geometry, 
            scale=scale_val, 
            bestEffort=True, 
            maxPixels=1e9
        ).getInfo()

        if not stats: 
            return 0.0, 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0, 0.0

        # 4. Sonuçları İşle
        ndvi_val = float(stats.get("ndvi", 0.0))
        mean_k = float(stats.get("k_value", 0.5))
        slope_val = float(stats.get("slope", 0.0))
        elev_val = float(stats.get("elevation", 0.0))
        land_cls_val = round(stats.get("land_class", 0))
        soil_cls_val = round(stats.get("soil_type", 0))

        land_map = {10: "Ormanlık", 20: "Çalılık", 30: "Çayır/Park", 40: "Tarım", 50: "Kentsel/Beton", 60: "Çıplak", 80: "Su"}
        land_type = land_map.get(int(round(land_cls_val/10)*10), "Karma Alan")
        
        soil_factor, soil_desc = 1.0, "Normal Toprak"
        if soil_cls_val in [1, 2, 3]: soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
        elif soil_cls_val in [9, 10, 11, 12]: soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"
        
        return ndvi_val, mean_k, soil_factor, land_type, soil_desc, slope_val * 1.5, elev_val

    except Exception as e:
        logging.error(f"GEE Data Hatası: {e}")
        return 0.0, 0.5, 1.0, "Hata", "Hata", 0.0, 0.0

def get_rain_10years(lat, lon):
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=UTC"
        r = requests.get(url, timeout=4).json() # Timeout düşürüldü
        clean = [d for d in r.get("daily", {}).get("precipitation_sum", []) if d is not None]
        if not clean: return 0.5, 0.5, 50.0, 500.0
        return clamp(sum(clean)/10000.0), clamp(max(clean)/120.0), max(clean), sum(clean)/10.0
    except Exception:
        return 0.5, 0.5, 50.0, 500.0

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        d = request.get_json(force=True)
        mode = d.get("mode", "point")
        radius = float(d.get("radius", 100.0))
        
        # --- DİNAMİK ÖLÇEK AYARI ---
        # Line (Hat) modu çok geniş alan taradığı için 30m kullanıyoruz.
        # Point (Nokta) modu küçük olduğu için 10m hassas tarıyoruz.
        dynamic_scale = 10 if mode == "point" else 30

        if mode == "line":
            start, end = d["start"], d["end"]
            line = ee.Geometry.LineString([[start["lon"], start["lat"]], [end["lon"], end["lat"]]])
            ee_geometry = line.buffer(radius)
            center_lat, center_lon = (start["lat"] + end["lat"])/2.0, (start["lon"] + end["lon"])/2.0
            length_m = line.length().getInfo()
            analysis_area_m2 = length_m * (radius * 2)
            L_flow = length_m
        else:
            lat, lon = float(d["lat"]), float(d["lon"])
            center_lat, center_lon = lat, lon
            ee_geometry = ee.Geometry.Point([lon, lat]).buffer(radius)
            analysis_area_m2 = math.pi * (radius ** 2.0)
            L_flow = radius * 2.0

        # Verileri Çek
        ndvi, K_cover, soil_factor, land_type, soil_desc, slope_pct, elevation = get_all_map_data(ee_geometry, dynamic_scale)
        
        if elevation <= 0 or land_type == "Su":
             return jsonify({"status": "success", "location_type": "Su Kütlesi", "selected_system": "hybrid", "material": "-", "system_reasoning": "Su kütlesi.", "scores": {}, "eco_stats": {"harvest":0, "bio_solution":"-"}, "rain_stats": {"mean":0, "max":0}, "Q_flow": 0, "pipe_diameter_mm": 0, "FloodRisk": 0, "FloodRiskLevel": "N/A", "K_value": 0, "ndvi": 0, "slope_percent": 0, "debug_analysis_area_ha": 0})

        W_star, R_ext, maxRain, meanRain = get_rain_10years(center_lat, center_lon)

        # HESAPLAMALAR
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - (ndvi * 0.30) if ndvi > 0.2 else 1.0
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
