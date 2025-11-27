# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK v8.1 (CLIMATE READY)
#  Yenilikler: İklim Değişikliği Projeksiyonu (%15 Artış)
#              Biyo-Tabanlı Çözüm Önerileri (Yeşil Altyapı)
#              Maliyet İndeksi Tahmini
#  Veri: Sentinel-2, ESA 10m, OpenLandMap, SRTM, Open-Meteo
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
import ee
import os
import datetime

app = Flask(__name__)
CORS(app)

# --- GEE YETKİLENDİRME ---
SERVICE_ACCOUNT = "earthengine-service@drenaj-v6.iam.gserviceaccount.com"
KEY_PATH = "/etc/secrets/service-account.json" 

if os.path.exists(KEY_PATH):
    try:
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
        ee.Initialize(credentials)
        print("GEE Başlatıldı (v8.1 - Climate Ready)")
    except Exception as e:
        print(f"GEE Hatası: {e}")
else:
    print("UYARI: GEE Key yok.")

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

# --- 1. SENTINEL-2 NDVI ---
def get_ndvi_data(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(50)
        s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(point) \
            .filterDate('2023-06-01', '2023-09-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .median()
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        val = ndvi.reduceRegion(ee.Reducer.mean(), area, 10).get('NDVI').getInfo()
        return float(val) if val else 0.0
    except: return 0.0

# --- 2. GELİŞMİŞ ALAN ANALİZİ ---
def get_advanced_area_data(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(50)

        # ESA 10m
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        from_cls = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k =     [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        k_img = dataset.remap(from_cls, to_k, 0.5)
        mean_k = k_img.reduceRegion(ee.Reducer.mean(), area, 10).get("remapped").getInfo() or 0.5
        mode_cls = dataset.reduceRegion(ee.Reducer.mode(), area, 10).get("Map").getInfo()
        land_map = {10:"Ormanlık", 20:"Çalılık", 30:"Çayır/Park", 40:"Tarım", 50:"Kentsel/Beton", 60:"Çıplak", 80:"Su"}
        land_type = land_map.get(mode_cls, "Karma Alan")

        # OpenLandMap (Toprak)
        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0")
        soil_mode = soil_img.reduceRegion(ee.Reducer.mode(), area, 250).get("b0").getInfo()
        soil_factor = 1.0
        soil_desc = "Normal Toprak"
        if soil_mode:
            sm = int(soil_mode)
            if sm in [1, 2, 3]: soil_factor, soil_desc = 1.25, "Killi (Sıkı)"
            elif sm in [9, 10, 11, 12]: soil_factor, soil_desc = 0.85, "Kumlu (Gevşek)"

        # SRTM (Eğim)
        dem = ee.Image("USGS/SRTMGL1_003")
        slope_val = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), area, 30).get("slope").getInfo()
        slope_pct = float(slope_val) * 1.5 if slope_val else 0.0

        return mean_k, soil_factor, land_type, soil_desc, slope_pct
    except: return 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0

# --- 3. 10 YILLIK YAĞIŞ ---
def get_rain_10years(lat, lon):
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=UTC"
        r = requests.get(url, timeout=15).json()
        clean = [d for d in r["daily"]["precipitation_sum"] if d is not None]
        if not clean: return 0.5, 0.5, 50.0, 500.0
        meanA = sum(clean) / 10.0
        maxD = max(clean)
        return clamp(meanA/1000.0), clamp(maxD/120.0), maxD, meanA
    except: return 0.5, 0.5, 50.0, 500.0

# ============================================================
#  ANA ANALİZ
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        lat, lon = float(d["lat"]), float(d["lon"])

        # 1. VERİLERİ AL
        K_cover, soil_factor, land_type, soil_desc, slope_pct = get_advanced_area_data(lat, lon)
        W_star, R_ext, maxRain, meanRain = get_rain_10years(lat, lon)
        ndvi = get_ndvi_data(lat, lon)

        # 2. HİDROLOJİK KATSAYILAR
        S = clamp(slope_pct / 25.0)
        veg_factor = 1.0 - (ndvi * 0.15) if ndvi > 0.2 else 1.0
        
        raw_C = 1.0 - K_cover
        C = clamp(raw_C * soil_factor * veg_factor)
        K_final = 1.0 - C

        # 3. RİSK HESABI
        W_blk = 0.6*W_star + 0.4*R_ext
        Risk_Lin = 0.36*W_blk + 0.34*C + 0.30*(1-S)
        FloodRisk = clamp(Risk_Lin + max(0, R_ext-0.75)*0.4)

        # 4. SİSTEM SEÇİMİ (AHP)
        Sc_DEN = 0.40*S + 0.40*FloodRisk + 0.20*(1-K_final)
        Sc_PAR = 0.50*(1-S) + 0.30*K_final + 0.20*(1-FloodRisk)
        Sc_RET = 0.60*C + 0.40*FloodRisk
        Sc_PIN = 0.50*S + 0.30*C + 0.20*W_star
        Sc_RAD = 0.60*(1.0-S) + 0.40*FloodRisk
        Sc_MEA = 0.80*S + 0.20*(1-C)
        S_mid = 1.0 - abs(2.0*S - 1.0)
        Sc_HYB = 0.35*FloodRisk + 0.35*C + 0.30*S_mid

        scores = {
            "dendritic": Sc_DEN, "parallel": Sc_PAR, "reticular": Sc_RET,
            "pinnate": Sc_PIN, "radial": Sc_RAD, "meandering": Sc_MEA, "hybrid": Sc_HYB
        }
        selected = max(scores, key=scores.get)

        # 5. İKLİM DEĞİŞİKLİĞİ VE HİDROLİK
        # IPCC: Gelecek 50 yılda aşırı yağışların %15 artması bekleniyor.
        Climate_Factor = 1.15 
        
        n_roughness = 0.025 if selected in ["meandering", "radial", "hybrid"] else 0.013
        
        # Kirpich
        L_flow, S_metric = 100.0, max(0.01, slope_pct / 100.0)
        t_c = max(5.0, min(45.0, 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))))

        # Debi (İklim Faktörlü)
        i_val = (maxRain * 1.5) / ((t_c/60.0 + 0.15)**0.7)
        Q_current = 0.278 * C * i_val * 1.5 
        Q_future = Q_current * Climate_Factor # Gelecek projeksiyonlu debi
        
        S_bed = max(0.005, S_metric)
        D_mm = (((4**(5/3)) * n_roughness * Q_future) / (math.pi * math.sqrt(S_bed)))**(3/8) * 1000.0
        
        # Malzeme & Maliyet İndeksi (Birim Maliyet Tahmini)
        cost_index = 1
        mat = "PVC"
        if selected in ["meandering", "radial"]:
            mat = "Doğal Taş Kanal / Gabion"
            cost_index = 3 # İşçilik yüksek
        elif D_mm >= 500:
            mat = "Betonarme / GRP"
            cost_index = 5 # Büyük çap pahalı
        elif D_mm >= 200:
            mat = "HDPE (Koruge)"
            cost_index = 2
        
        # 6. YEŞİL ALTYAPI ÖNERİLERİ (Biyo-Çözüm)
        # Sadece boru değil, tamamlayıcı çözüm öneriyoruz.
        bio_solution = "Standart Peyzaj"
        if C > 0.7: # Çok beton
            bio_solution = "Yeşil Çatı & Geçirimli Beton"
        elif slope_pct > 15: # Çok dik
            bio_solution = "Teraslama & Erozyon Önleyici Örtü"
        elif selected == "radial":
            bio_solution = "Yağmur Bahçesi (Rain Garden)"
        elif selected == "dendritic":
            bio_solution = "Bitkili Biyo-Hendek (Bioswale)"
        
        # 7. SU HASADI
        harvest_area = 15000.0
        harvest_potential = (harvest_area * (meanRain/1000.0) * 0.85 * (raw_C)) 

        lvl_idx = int(FloodRisk * 4.9)
        lvl = ["Çok Düşük","Düşük","Orta","Yüksek","Kritik"][min(lvl_idx, 4)]

        return jsonify({
            "status": "success",
            "location_type": f"{land_type} ({soil_desc})",
            "ndvi": round(ndvi, 2),
            "slope_percent": round(slope_pct, 2),
            "FloodRisk": round(FloodRisk, 2),
            "FloodRiskLevel": lvl,
            "selected_system": selected,
            "pipe_diameter_mm": round(D_mm, 0),
            "material": mat,
            "Q_flow": round(Q_future, 3), # İklim faktörlü debi
            "rain_stats": {
                "mean_annual_mm": round(meanRain, 1),
                "max_daily_mm": round(maxRain, 1)
            },
            "eco_stats": {
                "harvest_ton_year": round(harvest_potential, 0),
                "vegetation_factor": round(veg_factor, 2),
                "climate_projection": "IPCC +%15 Artış",
                "bio_solution": bio_solution,
                "cost_index": cost_index
            }
        })

    except Exception as e:
        return jsonify({"status":"error", "msg":str(e)}), 500

if __name__ == "__main__":
    app.run()

application = app
