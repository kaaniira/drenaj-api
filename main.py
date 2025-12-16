# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v13.0 (PLATINUM EDITION)
#  Rainfall Logic: Gumbel Distribution (EV-1) -> ACADEMIC STANDARD
#  Risk Logic: Smart Penalty Gates (V12.4) -> PHYSICAL ACCURACY
#  Feature: Full Frontend Compatibility
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
import ee
import os
import google.auth
import statistics
from collections import defaultdict

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
        print("✅ GEE Başlatıldı (v13.0 Platinum)")
    except Exception as e:
        print(f"❌ GEE Hatası: {e}")

initialize_gee()

# --- 3. İSTATİSTİKSEL YAĞIŞ MODELLERİ (GUMBEL) ---

def get_rain_series_10y(lat, lon):
    """Open-Meteo'dan 10 yıllık günlük veriyi çeker."""
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            "&start_date=2015-01-01&end_date=2024-12-31"
            "&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=5).json()
        daily = r.get("daily", {})
        return daily.get("precipitation_sum", []), daily.get("time", [])
    except:
        return [], []

def annual_max_series(daily_precip, daily_dates):
    """Her yılın en yüksek yağış değerini bulur."""
    yearly = defaultdict(list)
    for p, d in zip(daily_precip, daily_dates):
        if p is not None:
            yearly[int(d[:4])].append(p)
    return [max(v) for v in yearly.values() if v]

def gumbel_return_level(data, T=10):
    """Gumbel (Extreme Value Type I) dağılımı ile T yıllık tekerrür yağışını hesaplar."""
    if len(data) < 5: return max(data) if data else 50.0
    
    mean_x = statistics.mean(data)
    std_x = statistics.stdev(data) if len(data) > 1 else 0
    
    # Gumbel Faktörü (K_T)
    K_T = -(math.sqrt(6) / math.pi) * (0.5772 + math.log(math.log(T / (T - 1))))
    
    return mean_x + K_T * std_x

def idf_intensity_turkey_gumbel(annual_max, t_c_minutes, T=10):
    """Türkiye için MGM katsayıları + Gumbel Yağış."""
    P_T = gumbel_return_level(annual_max, T) # 24 saatlik Gumbel tahmini
    t_h = max(t_c_minutes / 60.0, 0.1)
    IDF_B_MGM = 0.54
    
    # Şiddet hesabı
    a = P_T / (24.0 ** (1.0 - IDF_B_MGM))
    return a * (t_h ** (-IDF_B_MGM))

def idf_intensity_global(maxRain, F_iklim, t_c_minutes):
    """Basit Global Formül (Yedek)"""
    t_h = max(t_c_minutes / 60.0, 0.1)
    IDF_GLOBAL_SCALE = 0.12
    return IDF_GLOBAL_SCALE * ((maxRain * F_iklim) / ((t_h + 0.15) ** 0.7))

def is_in_turkey(lat, lon):
    return (35.5 <= lat <= 42.5) and (25.0 <= lon <= 45.0)

# --- 4. GEE VERİ ÇEKME (V12.4 GOLD MASTER MANTIĞI) ---
def get_ndvi_data(geometry):
    try:
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(geometry)
              .filterDate("2023-06-01", "2023-09-30")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
              .select(["B8", "B4"])
              .median())
        ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
        val = ndvi.reduceRegion(ee.Reducer.mean(), geometry, scale=10, bestEffort=True, maxPixels=1e9).get("NDVI").getInfo()
        return float(val) if val is not None else 0.0
    except:
        return 0.0

def get_advanced_area_data(geometry):
    try:
        wc = ee.ImageCollection("ESA/WorldCover/v200").first()
        k_img = wc.remap([10,20,30,40,50,60,80,90,95,100], [0.90,0.80,0.85,0.60,0.15,0.50,0.00,0.00,0.90,0.90], 0.5).rename("k_value")
        land = wc.rename("land_class")
        soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0").rename("soil_type")
        dem = ee.Image("USGS/SRTMGL1_003")
        slope = ee.Terrain.slope(dem).rename("slope")
        elev = dem.select("elevation").rename("elevation")

        combined = ee.Image.cat([k_img, land, soil, slope, elev])
        
        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.mode(), sharedInputs=True),
            geometry=geometry, scale=10, bestEffort=True, maxPixels=1e9
        ).getInfo()

        if not stats: return 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0, 0.0

        land_cls = int(stats.get("land_class_mode", 50))
        soil_cls = int(stats.get("soil_type_mode", 0))
        slope_deg = float(stats.get("slope_mean", 0.0))
        elev_val = float(stats.get("elevation_mean", 0.0))
        mean_k = float(stats.get("k_value_mean", 0.5))
        
        slope_pct = math.tan(math.radians(slope_deg)) * 100.0

        land_map = {10:"Ormanlık", 20:"Çalılık", 30:"Çayır/Park", 40:"Tarım", 50:"Kentsel/Beton", 60:"Çıplak", 80:"Su"}
        land_type = land_map.get(land_cls, "Karma Alan")

        soil_factor, soil_desc = 1.0, "Normal Toprak"
        if soil_cls in [1,2,3]: soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
        elif soil_cls in [9,10,11,12]: soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"

        return mean_k, soil_factor, land_type, soil_desc, slope_pct, elev_val
    except:
        return 0.5, 1.0, "Hata", "Hata", 0.0, 0.0

# ============================================================
#  ANA ANALİZ
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        mode = d.get("mode", "point")
        r = float(d.get("radius", 100))

        # 1. Geometri Hesabı
        if mode == "line":
            s, e = d["start"], d["end"]
            line = ee.Geometry.LineString([[s["lon"], s["lat"]], [e["lon"], e["lat"]]])
            geom = line.buffer(r)
            center_lat, center_lon = (s["lat"]+e["lat"])/2, (s["lon"]+e["lon"])/2
            L_flow = max(line.length().getInfo(), 50)
            area_m2 = L_flow * (2 * r)
        else:
            lat, lon = float(d["lat"]), float(d["lon"])
            geom = ee.Geometry.Point([lon, lat]).buffer(r)
            center_lat, center_lon = lat, lon
            L_flow = 2 * r
            area_m2 = math.pi * (r ** 2)

        area_km2 = area_m2 / 1e6

        # 2. Arazi Verileri (Mode + Tan Slope)
        K_cover, soil_factor, land_type, soil_desc, slope_pct, elev = get_advanced_area_data(geom)

        if elev <= 0 or land_type == "Su":
            return jsonify({"status": "water", "msg": "Su kütlesi veya deniz seviyesi altı."})

        # 3. Gelişmiş Yağış Hesabı (GUMBEL)
        rain_daily, rain_dates = get_rain_series_10y(center_lat, center_lon)
        if rain_daily:
            annual_max = annual_max_series(rain_daily, rain_dates)
            meanRain = sum(rain_daily)/10.0
            maxRain = max(rain_daily)
        else:
            annual_max = [50.0]*10
            meanRain, maxRain = 500.0, 50.0

        W_star = clamp(meanRain/1000.0)
        R_ext = clamp(maxRain/120.0) # Ekstrem olay indeksi
        
        # NDVI
        ndvi = get_ndvi_data(geom)

        # 4. Matematiksel Risk Modeli (SMART GATES)
        S = clamp(slope_pct / 20.0)
        veg_factor = 1.0 - ndvi * 0.30 if ndvi > 0.2 else 1.0
        C = clamp((1 - K_cover) * soil_factor * veg_factor)
        K_final = 1.0 - C

        W_blk = 0.6 * W_star + 0.4 * R_ext
        S_risk = abs(2 * S - 1)

        Risk_Lin = 0.45 * W_blk + 0.45 * C + 0.10 * S_risk
        Risk_Pik = max(0, R_ext - 0.75) * 0.4
        
        # --- KRİTİK BİRLEŞME: SMART GATES ---
        urban_gate = clamp((C - 0.35) / 0.45)   
        storm_gate = clamp((R_ext - 0.20) / 0.60) 
        Arid_Urban_Penalty = 0.30 * (1.0 - W_star) * urban_gate * storm_gate

        FloodRisk = clamp(Risk_Lin + Risk_Pik + Arid_Urban_Penalty)

        # 5. Hidrolik Hesaplar (GUMBEL ENTEGRASYONU)
        Climate_Factor = 1.15
        S_metric = max(0.01, slope_pct / 100.0)
        t_c = max(5.0, min(45.0, 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))))
        
        # Gumbel ile i_val hesapla
        if is_in_turkey(center_lat, center_lon):
            i_val = idf_intensity_turkey_gumbel(annual_max, t_c, T=10) # 10 Yıllık tekerrür
        else:
            i_val = idf_intensity_global(maxRain, 3.0, t_c)
            
        Q_future = (0.278 * C * i_val * area_km2) * Climate_Factor
        D_mm = (((4 ** (5/3)) * 0.013 * Q_future) / (math.pi * math.sqrt(max(0.005, S_metric)))) ** (3/8) * 1000.0

        # 6. Sistem Seçimi (Frontend Verileri)
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

        reasons = {
           "dendritic": f"Orta eğim (%{slope_pct:.1f}) ve doğal akış hatları için uygundur.",
           "parallel": f"Düzenli eğim (%{slope_pct:.1f}) paralel tahliyeyi gerektiriyor.",
           "reticular": f"Yüksek geçirimsizlik (K={K_final:.2f}) ağsı yapıyı öne çıkarıyor.",
           "pinnate": f"Dik eğim (%{slope_pct:.1f}) hızlı tahliye için balık kılçığı modeli.",
           "radial": "Merkezi toplanma için uygun topografya.",
           "meandering": f"Yüksek eğimde (%{slope_pct:.1f}) erozyonu önlemek için kıvrımlı yapı.",
           "hybrid": "Karmaşık topografya hibrit çözüm gerektiriyor."
        }

        mat = "PVC"
        if selected in ["meandering", "radial"]: mat = "Doğal Taş Kanal"
        elif D_mm >= 500: mat = "Betonarme"
        elif D_mm >= 200: mat = "Koruge (HDPE)"

        bio_solutions = {"radial": "Yağmur Bahçesi", "dendritic": "Biyo-Hendek"}
        bio_solution = bio_solutions.get(selected, "Standart Peyzaj")
        if C > 0.7: bio_solution = "Yeşil Çatı & Geçirimli Beton"
        elif slope_pct > 15: bio_solution = "Teraslama"

        harvest = area_m2 * (meanRain / 1000.0) * 0.85 * (1.0 - K_final)
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
            "system_reasoning": reasons.get(selected, "Analiz sonucu."),
            "scores": scores,
            "pipe_diameter_mm": round(D_mm, 0),
            "material": mat,
            "Q_flow": round(Q_future, 3),
            "rain_stats": {"mean": round(meanRain, 1), "max": round(maxRain, 1)},
            "eco_stats": {"harvest": round(harvest, 0), "bio_solution": bio_solution},
            "debug_analysis_area_ha": round(area_km2 * 100, 2),
            "debug_info": {"L_flow_m": round(L_flow, 1), "i_val": round(i_val,2)}
        })

    except Exception as e:
        app.logger.error(f"Analysis Error: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

application = app
