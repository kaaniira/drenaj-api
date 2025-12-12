# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v16.0 (FINAL MASTER)
#  PATCH: NDVI opsiyonel (yoksa hata YOK), su kütlesi ayrımı fix
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

# --- CORS AYARLARI ---
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
            scopes=[
                'https://www.googleapis.com/auth/earthengine',
                'https://www.googleapis.com/auth/cloud-platform'
            ]
        )
        ee.Initialize(credentials, project=project)
        logging.info("GEE Başlatıldı.")
    except Exception as e:
        logging.error(f"GEE Başlatma Hatası: {e}")

initialize_gee()

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

# ============================================================
#  GERÇEK SU ORANI (mean ile DEĞİL, oran ile)
# ============================================================
def water_fraction(geometry, radius=100.0):
    try:
        wc = ee.ImageCollection("ESA/WorldCover/v200").first()
        land = wc.rename("land_class")

        scale = 10 if radius <= 100 else 30
        water = land.eq(80).rename("water")

        stats = water.reduceRegion(
            reducer=ee.Reducer.sum().combine(ee.Reducer.count(), sharedInputs=True),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=8
        ).getInfo() or {}

        w_sum = float(stats.get("water_sum", 0) or 0)
        w_cnt = float(stats.get("water_count", 0) or 0)

        if w_cnt < 5:
            return None

        return w_sum / w_cnt
    except Exception:
        return None

# ============================================================
#  GEE VERİ ÇEKME (NDVI OPSİYONEL)
# ============================================================
def get_gee_data_task(geometry, radius=100.0):
    try:
        # Sentinel-2 NDVI
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR")
              .filterBounds(geometry)
              .filterDate("2023-06-01", "2023-09-30")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
              .median())
        ndvi_img = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")

        # WorldCover
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        k_img = dataset.remap(
            [10, 20, 30, 40, 50, 60, 80, 90, 95, 100],
            [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90],
            0.5
        ).rename("k_value")
        land_cls_img = dataset.rename("land_class")

        # Soil + DEM
        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02") \
                     .select("b0").rename("soil_type")
        dem = ee.Image("USGS/SRTMGL1_003")

        combined = ee.Image.cat([
            ndvi_img, k_img, land_cls_img, soil_img,
            ee.Terrain.slope(dem).rename("slope"),
            dem.select("elevation").rename("elevation")
        ])

        scale = 10 if radius <= 100 else 30

        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=8
        )

        counts = combined.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=1e9,
            tileScale=8
        )

        stats_i = stats.getInfo() or {}
        counts_i = counts.getInfo() or {}

        land_count = int(counts_i.get("land_class", 0) or 0)
        ndvi_count = int(counts_i.get("ndvi", 0) or 0)

        # ❗ SADECE land cover yoksa iptal
        if land_count < 5:
            return {"ok": False, "reason": "gee_no_land", "counts": counts_i}

        return {
            "ok": True,
            "ndvi": float(stats_i.get("ndvi", 0.0) or 0.0),
            "ndvi_ok": ndvi_count >= 5,   # sadece bilgi
            "k": float(stats_i.get("k_value", 0.5) or 0.5),
            "slope": float(stats_i.get("slope", 0.0) or 0.0),
            "elev": float(stats_i.get("elevation", 0.0) or 0.0),
            "land": round(stats_i.get("land_class", 0) or 0),
            "soil": round(stats_i.get("soil_type", 0) or 0),
            "counts": counts_i
        }

    except Exception as e:
        logging.error(f"GEE Task Hatası: {e}")
        return {"ok": False, "reason": "gee_exception", "error": str(e)}

# ============================================================
#  HAVA DURUMU
# ============================================================
def get_weather_data_task(lat, lon):
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date=2015-01-01&end_date=2024-12-31"
            f"&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=5).json()
        clean = [d for d in r.get("daily", {}).get("precipitation_sum", []) if d is not None]

        if not clean:
            return 0.5, 0.5, 50.0, 500.0

        meanA = sum(clean) / 10.0
        maxD = max(clean)
        return clamp(meanA/1000.0), clamp(maxD/120.0), maxD, meanA
    except Exception:
        return 0.5, 0.5, 50.0, 500.0

# ============================================================
#  ANA ENDPOINT
# ============================================================
@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        d = request.get_json(force=True)
        mode = d.get("mode", "point")
        radius = float(d.get("radius", 100.0))

        if mode == "line":
            start, end = d["start"], d["end"]
            line = ee.Geometry.LineString([[start["lon"], start["lat"]],
                                           [end["lon"], end["lat"]]])
            ee_geometry = line.buffer(radius)
            center_lat = (start["lat"] + end["lat"]) / 2
            center_lon = (start["lon"] + end["lon"]) / 2
            L_flow = line.length().getInfo()
            analysis_area_m2 = L_flow * (radius * 2)
        else:
            center_lat = float(d["lat"])
            center_lon = float(d["lon"])
            ee_geometry = ee.Geometry.Point([center_lon, center_lat]).buffer(radius)
            analysis_area_m2 = math.pi * radius ** 2
            L_flow = radius * 2

        with ThreadPoolExecutor(max_workers=2) as executor:
            gee_res = executor.submit(get_gee_data_task, ee_geometry, radius).result()
            weather_res = executor.submit(
                get_weather_data_task, center_lat, center_lon
            ).result()

        if not gee_res or not gee_res.get("ok"):
            return jsonify({
                "status": "error",
                "msg": "Harita verisi alınamadı (arazi örtüsü yok).",
                "debug": gee_res
            }), 503

        # ------------------ VERİLER ------------------
        ndvi = gee_res["ndvi"]
        if not gee_res.get("ndvi_ok", True):
            ndvi = 0.2  # NDVI YOKSA → NÖTR

        K_cover = 1.0 - clamp((1.0 - gee_res["k"]))
        slope_pct = gee_res["slope"] * 1.5
        elevation = gee_res["elev"]

        W_star, R_ext, maxRain, meanRain = weather_res

        # ------------------ SUYA YAKINLIK ------------------
        wf = water_fraction(ee_geometry, radius)
        water_penalty = 0.0
        if wf is not None and wf > 0.15:
            water_penalty = 0.25

        # ------------------ HESAPLAR ------------------
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - (ndvi * 0.30)
        C = clamp((1.0 - K_cover) * veg_factor)
        K_final = 1.0 - C

        FloodRisk = clamp(
            (0.45*(0.6*W_star+0.4*R_ext) +
             0.45*C +
             0.1*abs(2*S-1) +
             water_penalty)
        )

        # ------------------ SKORLAR (AYNI) ------------------
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

        # ------------------ HİDROLİK ------------------
        S_metric = max(0.01, slope_pct / 100.0)
        t_c = max(5.0, min(45.0, 0.0195 * ((L_flow ** 0.77) / (S_metric ** 0.385))))

        i_val = (maxRain / (24.0 ** 0.46)) * ((max(t_c/60.0, 0.1)) ** -0.54)
        Q_future = (0.278 * C * i_val * (analysis_area_m2/1e6)) * 1.15
        D_mm = (((4**(5/3)) * 0.013 * Q_future) /
                (math.pi * math.sqrt(max(0.005, S_metric))))**(3/8) * 1000.0

        return jsonify({
            "status": "success",
            "ndvi": round(ndvi, 2),
            "slope_percent": round(slope_pct, 2),
            "FloodRisk": round(FloodRisk, 2),
            "selected_system": selected,
            "scores": scores,
            "pipe_diameter_mm": round(D_mm, 0),
            "Q_flow": round(Q_future, 3),
            "rain_stats": {"mean": round(meanRain, 1), "max": round(maxRain, 1)},
            "debug_water_fraction": wf,
            "debug_ndvi_ok": gee_res.get("ndvi_ok")
        })

    except Exception as e:
        logging.error(f"ANALIZ HATASI: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

application = app
