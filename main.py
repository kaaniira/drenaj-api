# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v12.2 (Enhanced for Line/Point)
#  TABAN: Kod 1 (Sözel Çıktılı & Türkiye Odaklı)
#  YENİLİK: Çizgi Hattı (LineString) + Nokta (Point) Desteği
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

# --- 2. GEE YETKİLENDİRME (CLOUD RUN ADC) ---
def initialize_gee():
    try:
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/cloud-platform']
        )
        ee.Initialize(credentials, project=project)
        print("GEE Başlatıldı (v12.2 - Line/Point Support)")
    except Exception as e:
        print(f"GEE Başlatma Hatası: {e}")

initialize_gee()

# ------------------------------------------------------------
#  TÜRKİYE SINIR KONTROLÜ + IDF FONKSİYONLARI
# ------------------------------------------------------------

def is_in_turkey(lat, lon):
    return (35.5 <= lat <= 42.5) and (25.0 <= lon <= 45.0)

IDF_B_MGM = 0.54       
IDF_GLOBAL_SCALE = 0.12 

def idf_intensity_turkey(maxRain_24h_mm, t_c_minutes):
    t_h = max(t_c_minutes / 60.0, 0.1)
    b = IDF_B_MGM
    a_local = maxRain_24h_mm / (24.0 ** (1.0 - b))
    i_val = a_local * (t_h ** (-b))
    return i_val

def idf_intensity_global_old(maxRain_mm, F_iklim, t_c_minutes):
    t_h = max(t_c_minutes / 60.0, 0.1)
    i_old = (maxRain_mm * F_iklim) / ((t_h + 0.15) ** 0.7)
    return IDF_GLOBAL_SCALE * i_old


# --- 3. HARİTA VERİLERİ (GEOMETRİ BAZLI REVİZE) ---

def get_ndvi_data(geometry):
    """
    Artık lat/lon yerine 'geometry' nesnesi alır.
    Böylece hem nokta tamponu hem de çizgi tamponu için çalışır.
    """
    try:
        # Tarih aralığı ve bulut filtresi
        s2 = (
           ee.ImageCollection("COPERNICUS/S2_SR")
           .filterBounds(geometry)
           .filterDate("2023-06-01", "2023-09-30")
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
            .median()
        )
        ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
        
        # Geometri üzerinden ortalama al
        val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=geometry, 
            scale=10,
            maxPixels=1e9,
            bestEffort=True
        ).get("NDVI").getInfo()
        
        return float(val) if val is not None else 0.0
    except Exception as e:
        print(f"NDVI Error: {e}")
        return 0.0

def get_advanced_area_data(geometry):
    """
    Veriler tek seferde çekilir ama scale=10m kullanılarak
    maksimum topografik hassasiyet sağlanır.
    Girdi olarak GEE Geometry nesnesi alır.
    """
    try:
        # A. Veri Setlerini Hazırla
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        from_cls = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k = [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        k_img = dataset.remap(from_cls, to_k, 0.5).rename("k_value")
        land_cls_img = dataset.rename("land_class")

        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil_type")

        dem = ee.Image("USGS/SRTMGL1_003")
        elev_img = dem.select("elevation").rename("elevation")
        slope_img = ee.Terrain.slope(dem).rename("slope")

        # B. Birleştir (Stacking)
        combined = ee.Image.cat([k_img, land_cls_img, soil_img, slope_img, elev_img])

        # C. Tek seferde iste (10 METRE HASSASİYET)
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(), 
            geometry=geometry, 
            scale=10,        
            bestEffort=True, 
            maxPixels=1e9
        ).getInfo()

        if not stats:
            return 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0, 0.0

        # D. Verileri Çözümle
        mean_k = float(stats.get("k_value", 0.5))
        slope_val = float(stats.get("slope", 0.0))
        elev_val = float(stats.get("elevation", 0.0))
        
        land_cls_val = round(stats.get("land_class", 0))
        soil_cls_val = round(stats.get("soil_type", 0))

        # E. Sözel Açıklamalar (KOD 1 ÖZELLİĞİ)
        land_map = {
            10: "Ormanlık", 20: "Çalılık", 30: "Çayır/Park",
            40: "Tarım", 50: "Kentsel/Beton", 60: "Çıplak", 80: "Su"
        }
        land_key = int(round(land_cls_val / 10.0) * 10)
        land_type = land_map.get(land_key, "Karma Alan")

        soil_factor = 1.0
        soil_desc = "Normal Toprak"
        if soil_cls_val in [1, 2, 3]:
            soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
        elif soil_cls_val in [9, 10, 11, 12]:
            soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"

        slope_pct = slope_val * 1.5

        return mean_k, soil_factor, land_type, soil_desc, slope_pct, elev_val

    except Exception as e:
        print(f"GEE Fetch Error: {e}")
        return 0.5, 1.0, "Hata", "Hata", 0.0, 0.0


# --- 4. YAĞIŞ VERİSİ (Hala Nokta Bazlı) ---
def get_rain_10years(lat, lon):
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            "&start_date=2015-01-01&end_date=2024-12-31"
            "&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=5).json()
        if "daily" not in r or "precipitation_sum" not in r["daily"]:
             return 0.5, 0.5, 50.0, 500.0
             
        clean = [d for d in r["daily"]["precipitation_sum"] if d is not None]
        if not clean:
            return 0.5, 0.5, 50.0, 500.0
            
        meanA = sum(clean) / 10.0   
        maxD = max(clean)           

        W_star = clamp(meanA / 1000.0)
        R_ext = clamp(maxD / 120.0)

        return W_star, R_ext, maxD, meanA
    except Exception:
        return 0.5, 0.5, 50.0, 500.0


# ============================================================
#  ANA ANALİZ (MOD DESTEĞİ EKLENDİ)
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        
        # --- 1. MOD VE GEOMETRİ AYARI ---
        mode = d.get("mode", "point") # Varsayılan: Nokta
        GEE_BUFFER_RADIUS_M = float(d.get("radius", 100.0))
        
        ee_geometry = None
        center_lat = 0
        center_lon = 0
        L_flow = 0
        analysis_area_m2 = 0
        
        if mode == "line":
            # Çizgi Hattı Modu (Manuel veya Çizim)
            start = d.get("start", {})
            end = d.get("end", {})
            s_lat, s_lon = float(start.get("lat")), float(start.get("lon"))
            e_lat, e_lon = float(end.get("lat")), float(end.get("lon"))
            
            # GEE LineString Oluştur
            line_geom = ee.Geometry.LineString([[s_lon, s_lat], [e_lon, e_lat]])
            ee_geometry = line_geom.buffer(GEE_BUFFER_RADIUS_M)
            
            # Merkez Noktayı Bul (Hava durumu için)
            center_lat = (s_lat + e_lat) / 2
            center_lon = (s_lon + e_lon) / 2
            
            # Akış Uzunluğu = Çizginin kendisi
            L_flow_raw = line_geom.length().getInfo()
            L_flow = max(L_flow_raw, 50.0) # En az 50m olsun
            
            # Alan (Yaklaşık dikdörtgen + uçlar)
            analysis_area_m2 = L_flow * (GEE_BUFFER_RADIUS_M * 2)

        else:
            # Nokta Modu (Varsayılan Kod 1 Davranışı)
            lat = float(d.get("lat"))
            lon = float(d.get("lon"))
            
            point_geom = ee.Geometry.Point([lon, lat])
            ee_geometry = point_geom.buffer(GEE_BUFFER_RADIUS_M)
            
            center_lat = lat
            center_lon = lon
            
            # Akış Uzunluğu = Çap
            L_flow = GEE_BUFFER_RADIUS_M * 2.0
            analysis_area_m2 = math.pi * (GEE_BUFFER_RADIUS_M ** 2.0)

        analysis_area_km2 = analysis_area_m2 / 1_000_000.0

        # --- 2. VERİ ÇEKME (Revize edilmiş fonksiyonlar) ---
        # Artık 'ee_geometry' nesnesini gönderiyoruz
        K_cover, soil_factor, land_type, soil_desc, slope_pct, elevation = (
            get_advanced_area_data(ee_geometry)
        )

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

        # --- 3. MATEMATİKSEL MODEL (Kod 1 Mantığı) ---
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - (ndvi * 0.30) if ndvi > 0.2 else 1.0

        raw_C = 1.0 - K_cover
        C = clamp(raw_C * soil_factor * veg_factor)
        K_final = 1.0 - C

        W_blk = 0.6 * W_star + 0.4 * R_ext
        S_risk = abs(2.0 * S - 1.0)

        Risk_Lin = 0.45 * W_blk + 0.45 * C + 0.10 * S_risk
        Risk_Pik = max(0, R_ext - 0.75) * 0.4
        Baseline_Risk = Risk_Lin + Risk_Pik

        is_arid_factor = 1.0 - W_star
        Arid_Urban_Penalty = is_arid_factor * 0.3
        FloodRisk = clamp(Baseline_Risk + Arid_Urban_Penalty)

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

        # Sözel Açıklamalar (Kod 1'e özgü)
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

        # --- 4. HİDROLİK HESAPLAMA ---
        Climate_Factor = 1.15
        n_roughness = 0.025 if selected in ["meandering", "radial"] else 0.013
        S_metric = max(0.01, slope_pct / 100.0)
        
        # Manning Time of Concentration (L_flow artık değişkene bağlı)
        t_c_raw = 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))
        t_c = max(5.0, min(45.0, t_c_raw))

        F_iklim = 3.0
        if meanRain > 1500: F_iklim = 2.5
        elif meanRain < 400: F_iklim = 3.5

        # Türkiye Kontrolü (Merkez noktasına göre)
        if is_in_turkey(center_lat, center_lon):
            i_val = idf_intensity_turkey(maxRain, t_c)
        else:
            i_val = idf_intensity_global_old(maxRain, F_iklim, t_c)

        Q_future = (0.278 * C * i_val * analysis_area_km2) * Climate_Factor

        S_bed = max(0.005, S_metric)
        D_mm = (((4 ** (5 / 3)) * n_roughness * Q_future) / (math.pi * math.sqrt(S_bed))) ** (3 / 8) * 1000.0

        # --- 5. MALZEME VE PEYZAJ ÖNERİLERİ (Kod 1'e özgü) ---
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
