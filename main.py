# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v13.0 (Dynamic Mode: Point/Line)
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS # Bu kütüphanenin yüklü olduğundan emin olun
import requests
import math
import ee
import os
import google.auth

app = Flask(__name__)

# --- CORS AYARI (GÜNCELLENMİŞ) ---
# allow_headers: Content-Type gibi başlıkların geçmesine izin verir
# origins: "*" diyerek her yerden gelen isteği kabul eder (Localhost ve great-site.net dahil)
CORS(app, resources={r"/*": {"origins": "*"}}, 
     supports_credentials=True, 
     allow_headers=["Content-Type", "Authorization"], 
     methods=["GET", "POST", "OPTIONS"])

# --- YARDIMCI VE BAŞLATMA ---
def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def initialize_gee():
    try:
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
        )
        ee.Initialize(credentials, project=project)
        print("GEE Başlatıldı (v13 - Dynamic Geometry)")
    except Exception as e:
        print(f"GEE Başlatma Hatası: {e}")

initialize_gee()

# --- VERİ ÇEKME FONKSİYONLARI (Geometri Bağımsız) ---

def get_ndvi_data(geometry):
    try:
        # Tarih aralığı dinamik olabilir, şimdilik sabit
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate("2023-06-01", "2023-09-30")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .median()
        )
        ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
        
        val = ndvi.reduceRegion(ee.Reducer.mean(), geometry, 10).get("NDVI").getInfo()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

def get_advanced_area_data(geometry):
    """
    Artık lat/lon yerine doğrudan EE Geometry nesnesi alıyor.
    Böylece hem daire (buffer) hem koridor (line buffer) için çalışır.
    """
    try:
        # 1. Hazırla
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        from_cls = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k = [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        k_img = dataset.remap(from_cls, to_k, 0.5).rename("k_value")
        land_cls_img = dataset.rename("land_class")
        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil_type")
        dem = ee.Image("USGS/SRTMGL1_003")
        elev_img = dem.select("elevation").rename("elevation")
        slope_img = ee.Terrain.slope(dem).rename("slope")

        # 2. Birleştir
        combined = ee.Image.cat([k_img, land_cls_img, soil_img, slope_img, elev_img])

        # 3. İste (Scale 10m)
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=geometry, 
            scale=10, 
            bestEffort=True,
            maxPixels=1e9
        ).getInfo()

        if not stats:
            return 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0, 0.0

        # 4. Parse Et
        mean_k = float(stats.get("k_value", 0.5))
        slope_val = float(stats.get("slope", 0.0))
        elev_val = float(stats.get("elevation", 0.0))
        
        land_cls_val = round(stats.get("land_class", 0))
        soil_cls_val = round(stats.get("soil_type", 0))

        land_map = {10: "Ormanlık", 20: "Çalılık", 30: "Çayır/Park", 40: "Tarım", 50: "Kentsel/Beton", 60: "Çıplak", 80: "Su"}
        land_key = int(round(land_cls_val / 10.0) * 10)
        land_type = land_map.get(land_key, "Karma Alan")

        soil_factor, soil_desc = 1.0, "Normal Toprak"
        if soil_cls_val in [1, 2, 3]: soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
        elif soil_cls_val in [9, 10, 11, 12]: soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"

        slope_pct = slope_val * 1.5
        return mean_k, soil_factor, land_type, soil_desc, slope_pct, elev_val

    except Exception as e:
        print(f"GEE Fetch Error: {e}")
        return 0.5, 1.0, "Hata", "Hata", 0.0, 0.0

def get_rain_10years(lat, lon):
    # Yağış verisi için tek bir merkez koordinata ihtiyacımız var (Open-Meteo için)
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=UTC"
        r = requests.get(url, timeout=5).json()
        clean = [d for d in r.get("daily", {}).get("precipitation_sum", []) if d is not None]
        
        if not clean: return 0.5, 0.5, 50.0, 500.0
        meanA = sum(clean) / 10.0   
        maxD = max(clean)
        return clamp(meanA / 1000.0), clamp(maxD / 120.0), maxD, meanA
    except Exception:
        return 0.5, 0.5, 50.0, 500.0

# --- ANALİZ ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        
        # PARAMETRELER
        mode = d.get("mode", "point") # 'point' veya 'line'
        radius = float(d.get("radius", 100.0))
        
        # Geometri Oluşturma
        ee_geometry = None
        center_lat, center_lon = 0, 0
        
        if mode == "line":
            # Hat Modu: İki nokta arası çizgi ve buffer
            start = d["start"] # {lat, lon}
            end = d["end"]     # {lat, lon}
            line = ee.Geometry.LineString([[start["lon"], start["lat"]], [end["lon"], end["lat"]]])
            ee_geometry = line.buffer(radius) # Hat boyunca koridor
            
            # Hava durumu için orta noktayı bul
            center_lat = (start["lat"] + end["lat"]) / 2.0
            center_lon = (start["lon"] + end["lon"]) / 2.0
            
            # Alan Hesabı (Dikdörtgen benzeri)
            # Kabaca: Uzunluk * Genişlik (2*radius)
            length = line.length().getInfo() # metre
            analysis_area_m2 = length * (radius * 2) 

        else:
            # Nokta Modu: Daire
            lat, lon = float(d["lat"]), float(d["lon"])
            center_lat, center_lon = lat, lon
            pt = ee.Geometry.Point([lon, lat])
            ee_geometry = pt.buffer(radius)
            analysis_area_m2 = math.pi * (radius ** 2.0)

        analysis_area_km2 = analysis_area_m2 / 1_000_000.0
        
        # Akış uzunluğu (Kirpich için)
        if mode == "line":
             # Hat modunda akış uzunluğu hattın kendi uzunluğudur (veya eğime göre değişir ama hattı baz alıyoruz)
             L_flow = math.sqrt((start["lat"]-end["lat"])**2 + (start["lon"]-end["lon"])**2) * 111000 
        else:
             L_flow = radius * 2.0

        # --- VERİLERİ TOPLA ---
        K_cover, soil_factor, land_type, soil_desc, slope_pct, elevation = get_advanced_area_data(ee_geometry)

        # Su Kontrolü
        if elevation <= 0 or land_type == "Su":
            return jsonify({"status": "water", "msg": "Su kütlesi.", "location_type": "Su", "selected_system": "water", "FloodRiskLevel": "N/A"})

        W_star, R_ext, maxRain, meanRain = get_rain_10years(center_lat, center_lon)
        ndvi = get_ndvi_data(ee_geometry)

        # --- HESAPLAMALAR (Aynı Mantık) ---
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - (ndvi * 0.30) if ndvi > 0.2 else 1.0
        raw_C = 1.0 - K_cover
        C = clamp(raw_C * soil_factor * veg_factor)
        K_final = 1.0 - C

        W_blk = 0.6 * W_star + 0.4 * R_ext
        S_risk = abs(2.0 * S - 1.0)
        Baseline_Risk = (0.45 * W_blk + 0.45 * C + 0.10 * S_risk) + max(0, R_ext - 0.75) * 0.4
        FloodRisk = clamp(Baseline_Risk + (1.0 - W_star) * 0.3)

        # Sistem Seçimi
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

        # Hidrolik
        S_metric = max(0.01, slope_pct / 100.0)
        t_c = max(5.0, min(45.0, 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))))
        
        # IDF
        F_iklim = 3.0
        if meanRain > 1500: F_iklim = 2.5
        elif meanRain < 400: F_iklim = 3.5
        
        # Basit Global/Local Ayrımı
        is_turkey = (35.5 <= center_lat <= 42.5) and (25.0 <= center_lon <= 45.0)
        if is_turkey:
             i_val = (maxRain / (24.0 ** (1.0 - 0.54))) * ((max(t_c/60.0, 0.1)) ** (-0.54))
        else:
             i_old = (maxRain * F_iklim) / ((max(t_c/60.0, 0.1) + 0.15) ** 0.7)
             i_val = 0.12 * i_old

        Q_future = (0.278 * C * i_val * analysis_area_km2) * 1.15
        n_rough = 0.025 if selected in ["meandering", "radial"] else 0.013
        D_mm = (((4**(5/3)) * n_rough * Q_future) / (math.pi * math.sqrt(max(0.005, S_metric))))**(3/8) * 1000.0

        # Ek Açıklamalar
        mat = "PVC"
        if selected in ["meandering", "radial"]: mat = "Doğal Taş Kanal"
        elif D_mm >= 500: mat = "Betonarme"
        elif D_mm >= 200: mat = "Koruge (HDPE)"
        
        reasons = {
            "dendritic": f"Eğim (%{slope_pct:.1f}) ve vadi yapısı, suyu doğal akışla toplamayı öneriyor.",
            "parallel": f"Tek yönlü düzenli eğim (%{slope_pct:.1f}), paralel tahliye gerektiriyor.",
            "reticular": f"Yüksek geçirimsizlik (K={K_final:.2f}) ve şehir dokusu ağsı yapıyı zorunlu kılıyor.",
            "pinnate": f"Dar alan ve eğim (%{slope_pct:.1f}) balık kılçığı modelini öne çıkardı.",
            "radial": "Merkezi toplanma için uygun topografya.",
            "meandering": "Yüksek eğimde hızı kırmak için kıvrımlı yapı.",
            "hybrid": "Karmaşık arazi hibrit çözüm gerektiriyor."
        }

        harvest = analysis_area_m2 * (meanRain / 1000.0) * 0.85 * (1.0 - K_final)
        lvl = ["Çok Düşük", "Düşük", "Orta", "Yüksek", "Kritik"][min(int(FloodRisk*4.9), 4)]

        # Yeşil Altyapı
        bio_solution = "Standart Peyzaj"
        if C > 0.7: bio_solution = "Yeşil Çatı & Geçirimli Beton"
        elif slope_pct > 15: bio_solution = "Teraslama"
        elif selected == "radial": bio_solution = "Yağmur Bahçesi"
        elif selected == "dendritic": bio_solution = "Biyo-Hendek"

        return jsonify({
            "status": "success",
            "location_type": f"{land_type} ({soil_desc})",
            "ndvi": round(ndvi, 2),
            "slope_percent": round(slope_pct, 2),
            "K_value": round(K_final, 2),
            "FloodRisk": round(FloodRisk, 2),
            "FloodRiskLevel": lvl,
            "selected_system": selected,
            "system_reasoning": reasons.get(selected, "Analiz edildi."),
            "scores": scores,
            "pipe_diameter_mm": round(D_mm, 0),
            "material": mat,
            "Q_flow": round(Q_future, 3),
            "rain_stats": {"mean": round(meanRain, 1), "max": round(maxRain, 1)},
            "eco_stats": {"harvest": round(harvest, 0), "bio_solution": bio_solution},
            "debug_analysis_area_ha": round(analysis_area_m2/10000.0, 2),
            "mode": mode
        })

    except Exception as e:
        app.logger.error(f"Analysis Error: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

application = app
