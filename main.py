# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v12.3 (ULTIMATE MERGE)
#  Matematik: Kullanıcının Gelişmiş Formülleri (Mode, Tan, Fix)
#  Özellikler: Web Arayüzü İçin Tam Uyumluluk (Skorlar, Boru Çapı vb.)
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
import ee
import os
import google.auth

app = Flask(__name__)
CORS(app)

# --- 1. YARDIMCI FONKSİYONLAR ---
def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

# --- 2. GEE BAŞLATMA ---
def initialize_gee():
    try:
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
        )
        ee.Initialize(credentials, project=project)
        print("✅ GEE Başlatıldı (v12.3 Ultimate)")
    except Exception as e:
        print(f"❌ GEE Hatası: {e}")

initialize_gee()

# --- 3. TÜRKİYE VE IDF FORMÜLLERİ ---
def is_in_turkey(lat, lon):
    return (35.5 <= lat <= 42.5) and (25.0 <= lon <= 45.0)

IDF_B_MGM = 0.54
IDF_GLOBAL_SCALE = 0.12

def idf_intensity_turkey(maxRain_24h_mm, t_c_minutes):
    t_h = max(t_c_minutes / 60.0, 0.1)
    # MGM formülü
    a_local = maxRain_24h_mm / (24.0 ** (1.0 - IDF_B_MGM))
    i_val = a_local * (t_h ** (-IDF_B_MGM))
    return i_val

def idf_intensity_global_old(maxRain_mm, F_iklim, t_c_minutes):
    t_h = max(t_c_minutes / 60.0, 0.1)
    i_old = (maxRain_mm * F_iklim) / ((t_h + 0.15) ** 0.7)
    return IDF_GLOBAL_SCALE * i_old

# --- 4. GELİŞMİŞ GEE VERİ ÇEKME (SENİN MATEMATİĞİNLE) ---
def get_ndvi_data(geometry):
    try:
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") # Güncel koleksiyon
            .filterBounds(geometry)
            .filterDate("2023-06-01", "2023-09-30")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .select(["B8", "B4"]) # FIX: Bant hatasını önler
            .median()
        )
        ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
        
        val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=geometry, 
            scale=10,
            bestEffort=True, 
            maxPixels=1e9
        ).get("NDVI").getInfo()
        
        return float(val) if val is not None else 0.0
    except:
        return 0.0

def get_advanced_area_data(geometry):
    """
    FIX-1: 'Mode' reducer kullanarak daha doğru sınıflandırma.
    FIX-2: Tanjant ile gerçek eğim yüzdesi.
    """
    try:
        # A. Veri Setleri
        wc = ee.ImageCollection("ESA/WorldCover/v200").first()
        
        # WorldCover K Değerleri
        k_img = wc.remap(
            [10, 20, 30, 40, 50, 60, 80, 90, 95, 100],
            [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90],
            0.5
        ).rename("k_value")
        
        land_cls_img = wc.rename("land_class")

        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil_type")

        dem = ee.Image("USGS/SRTMGL1_003")
        slope_img = ee.Terrain.slope(dem).rename("slope") # Derece cinsinden
        elev_img = dem.select("elevation").rename("elevation")

        # B. Birleştirme
        combined = ee.Image.cat([k_img, land_cls_img, soil_img, slope_img, elev_img])

        # C. İstatistik (Mode + Mean)
        # Kategorik veriler (land, soil) için MODE, Sayısal veriler (k, slope, elev) için MEAN
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.mode(), sharedInputs=True),
            geometry=geometry, 
            scale=10,        
            bestEffort=True, 
            maxPixels=1e9
        ).getInfo()

        if not stats:
            return 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0, 0.0

        # D. Veri Çözümleme (Gelişmiş)
        # Kategorik verilerde _mode soneki aranır
        land_cls_val = int(stats.get("land_class_mode", stats.get("land_class_mean", 50)))
        soil_cls_val = int(stats.get("soil_type_mode", stats.get("soil_type_mean", 0)))
        
        # Sayısal verilerde _mean
        mean_k = float(stats.get("k_value_mean", 0.5))
        slope_deg_val = float(stats.get("slope_mean", 0.0))
        elev_val = float(stats.get("elevation_mean", 0.0))

        # FIX-2: Derece -> Yüzde Dönüşümü (Fiziksel Doğruluk)
        slope_pct = math.tan(math.radians(slope_deg_val)) * 100.0

        # E. Sözel Tanımlar
        land_map = {
            10: "Ormanlık", 20: "Çalılık", 30: "Çayır/Park",
            40: "Tarım", 50: "Kentsel/Beton", 60: "Çıplak", 80: "Su"
        }
        land_type = land_map.get(land_cls_val, "Karma Alan")

        soil_factor = 1.0
        soil_desc = "Normal Toprak"
        if soil_cls_val in [1, 2, 3]:
            soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
        elif soil_cls_val in [9, 10, 11, 12]:
            soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"

        return mean_k, soil_factor, land_type, soil_desc, slope_pct, elev_val

    except Exception as e:
        print(f"GEE Fetch Error: {e}")
        return 0.5, 1.0, "Hata", "Hata", 0.0, 0.0

# --- 5. YAĞIŞ VERİSİ ---
def get_rain_10years(lat, lon):
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            "&start_date=2015-01-01&end_date=2024-12-31"
            "&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=5).json()
        clean = [d for d in r.get("daily", {}).get("precipitation_sum", []) if d is not None]
        
        if not clean: return 0.5, 0.5, 50.0, 500.0
            
        meanA = sum(clean) / 10.0   
        maxD = max(clean)           
        return clamp(meanA / 1000.0), clamp(maxD / 120.0), maxD, meanA
    except:
        return 0.5, 0.5, 50.0, 500.0

# ============================================================
#  ANA ANALİZ (FULL FRONTEND UYUMLU)
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        
        # 1. Geometri ve Mod
        mode = d.get("mode", "point")
        r = float(d.get("radius", 100.0))
        
        if mode == "line":
            s, e = d.get("start"), d.get("end")
            line_geom = ee.Geometry.LineString([[s["lon"], s["lat"]], [e["lon"], e["lat"]]])
            ee_geometry = line_geom.buffer(r)
            center_lat, center_lon = (s["lat"] + e["lat"]) / 2, (s["lon"] + e["lon"]) / 2
            L_flow = max(line_geom.length().getInfo(), 50.0)
            analysis_area_m2 = L_flow * (r * 2)
        else:
            lat, lon = float(d["lat"]), float(d["lon"])
            ee_geometry = ee.Geometry.Point([lon, lat]).buffer(r)
            center_lat, center_lon = lat, lon
            L_flow = r * 2.0
            analysis_area_m2 = math.pi * (r ** 2.0)

        analysis_area_km2 = analysis_area_m2 / 1_000_000.0

        # 2. Verileri Çek (Geliştirilmiş Fonksiyonlar)
        K_cover, soil_factor, land_type, soil_desc, slope_pct, elevation = \
            get_advanced_area_data(ee_geometry)

        if elevation <= 0 or land_type == "Su":
            return jsonify({
               "status": "water",
               "msg": "Analiz alanı su kütlesidir.",
               "location_type": f"Su (Rakım: {elevation:.1f}m)",
               "selected_system": "water",
               "FloodRiskLevel": "N/A",
            })

        W_star, R_ext, maxRain, meanRain = get_rain_10years(center_lat, center_lon)
        ndvi = get_ndvi_data(ee_geometry)

        # 3. Matematiksel Model (V12.2 Logic)
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - (ndvi * 0.30) if ndvi > 0.2 else 1.0

        raw_C = 1.0 - K_cover
        C = clamp(raw_C * soil_factor * veg_factor)
        K_final = 1.0 - C

        W_blk = 0.6 * W_star + 0.4 * R_ext
        S_risk = abs(2.0 * S - 1.0)

        Risk_Lin = 0.45 * W_blk + 0.45 * C + 0.10 * S_risk
        Risk_Pik = max(0, R_ext - 0.75) * 0.4
        
        # Kuraklık Cezası (Senin kodundan alındı)
        Arid_Penalty = (1.0 - W_star) * 0.3
        
        FloodRisk = clamp(Risk_Lin + Risk_Pik + Arid_Penalty)

        # 4. Sistem Seçimi ve Skorlama (Frontend için gerekli!)
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

        # 5. Açıklamalar ve Hidrolik Hesaplar
        reasons = {
           "dendritic": f"Orta eğim (%{slope_pct:.1f}) ve doğal akış hatları Dendritik yapıyı öne çıkarıyor.",
           "parallel": f"Düzenli ve tek yönlü eğim (%{slope_pct:.1f}), paralel tahliyeyi gerektiriyor.",
           "reticular": f"Yüksek geçirimsizlik (K={K_final:.2f}) ve kentsel doku ağsı yapıyı gerektiriyor.",
           "pinnate": f"Dik eğim (%{slope_pct:.1f}) suyu hızlı tahliye etmek için balık kılçığı modelini seçtirdi.",
           "radial": f"Düz arazi (%{slope_pct:.1f}) ve geçirgen zemin, suyu merkezi toplamaya uygun.",
           "meandering": f"Yüksek eğimde (%{slope_pct:.1f}) erozyonu önlemek için kıvrımlı yapı seçildi.",
           "hybrid": "Karmaşık topografya hibrit çözüm gerektiriyor."
        }
        reason_txt = reasons.get(selected, "Karmaşık yapı.")

        # Boru Çapı Hesabı
        Climate_Factor = 1.15
        n_roughness = 0.025 if selected in ["meandering", "radial"] else 0.013
        S_metric = max(0.01, slope_pct / 100.0)
        
        t_c_raw = 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))
        t_c = max(5.0, min(45.0, t_c_raw))

        F_iklim = 3.0
        if meanRain > 1500: F_iklim = 2.5
        elif meanRain < 400: F_iklim = 3.5

        if is_in_turkey(center_lat, center_lon):
            i_val = idf_intensity_turkey(maxRain, t_c)
        else:
            i_val = idf_intensity_global_old(maxRain, F_iklim, t_c)

        Q_future = (0.278 * C * i_val * analysis_area_km2) * Climate_Factor
        S_bed = max(0.005, S_metric)
        D_mm = (((4 ** (5 / 3)) * n_roughness * Q_future) / (math.pi * math.sqrt(S_bed))) ** (3 / 8) * 1000.0

        # Malzeme ve Biyo Çözüm
        mat = "PVC"
        if selected in ["meandering", "radial"]: mat = "Doğal Taş Kanal"
        elif D_mm >= 500: mat = "Betonarme"
        elif D_mm >= 200: mat = "Koruge (HDPE)"

        bio_solutions = {
           "radial": "Yağmur Bahçesi (Rain Garden)",
           "dendritic": "Biyo-Hendek (Bioswale)"
        }
        bio_solution = bio_solutions.get(selected, "Standart Peyzaj")
        if C > 0.7: bio_solution = "Yeşil Çatı & Geçirimli Beton"
        elif slope_pct > 15: bio_solution = "Teraslama & Erozyon Önleyici"

        harvest = analysis_area_m2 * (meanRain / 1000.0) * 0.85 * (1.0 - K_final)
        lvl_idx = min(int(FloodRisk * 4.9), 4)
        lvl = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Kritik"][lvl_idx]

        # 6. JSON Yanıt (Frontend'in Beklediği Her Şey)
        return jsonify({
           "status": "success",
           "mode": mode,
           "location_type": f"{land_type} ({soil_desc})",
           "ndvi": round(ndvi, 2),
           "slope_percent": round(slope_pct, 2),
           "K_value": round(K_final, 2),
           "FloodRisk": round(FloodRisk, 2),
           "FloodRiskLevel": lvl,
           "selected_system": selected,
           "system_reasoning": reason_txt,
           "scores": scores,
           "pipe_diameter_mm": round(D_mm, 0),
           "material": mat,
           "Q_flow": round(Q_future, 3),
           "rain_stats": {"mean": round(meanRain, 1), "max": round(maxRain, 1)},
           "eco_stats": {"harvest": round(harvest, 0), "bio_solution": bio_solution},
           "debug_analysis_area_ha": round(analysis_area_m2 / 10000.0, 2),
           "debug_info": {
               "L_flow_m": round(L_flow, 1),
               "area_m2": round(analysis_area_m2, 0)
           }
        })

    except Exception as e:
        app.logger.error(f"Analysis Error: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

application = app
