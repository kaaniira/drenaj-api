# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — v11.4 (Kısmi Düzeltme)
#  KULLANICI TALEBİ: Menderes (Sc_MEA) sistemi ve gerekçesi (reason_txt),
#                   kullanıcı talebi üzerine orijinal (v11.3) mantığında (bilimsel olarak hatalı) bırakılmıştır.
#
#  UYGULANAN DİĞER HAKEM DÜZELTMELERİ:
#  Düzeltme (Su): v11.4 (Madde 4) - Su tespiti için ESA Sınıf 80 (land_type) eklendi.
#  Düzeltme (Kirpich): v11.4 (Madde 10) - L_flow (akış uzunluğu) yarıçap yerine çap (200m) olarak düzeltildi.
#  Düzeltme (Ölçek): v11.4 (Madde 3) - Toprak verisi (soil_mode) 250m buffer yerine 100m alana ve 10m ölçeğe çekildi.
#  Düzeltme (Risk): v11.4 (Madde 7) - AridPenalty'deki 'C' ile çifte sayım kaldırıldı, ağırlık 0.3'e kalibre edildi.
#  Düzeltme (Eğim S): v11.4 (Madde 8) - AHP eğim normalizasyonu (S) lineer (slope/15) yerine logaritmik fonksiyona geçirildi.
#  Düzeltme (NDVI): v11.4 (Madde 5) - veg_factor etkisi 0.15'ten 0.30'a yükseltildi.
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import math
import ee
import os

app = Flask(__name__)
CORS(app)

# --- GEE YETKİLENDİRME ---
SERVICE_ACCOUNT = "earthengine-service@drenaj-v6.iam.gserviceaccount.com"
KEY_PATH = "/etc/secrets/service-account.json" 

if os.path.exists(KEY_PATH):
    try:
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
        ee.Initialize(credentials)
        print("GEE Başlatıldı (v11.4 - Kısmi Düzeltme)")
    except Exception as e:
        print(f"GEE Hatası: {e}")
else:
    print("UYARI: GEE Key yok.")

def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

# --- 1. NDVI (Bitki) ---
def get_ndvi_data(lat, lon, buffer_radius_m):
    # (Bu fonksiyonda değişiklik yok)
    try:
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(buffer_radius_m) 
        s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(point) \
            .filterDate('2023-06-01', '2023-09-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .median()
        ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
        val = ndvi.reduceRegion(ee.Reducer.mean(), area, 10).get('NDVI').getInfo()
        return float(val) if val else 0.0
    except: return 0.0

# --- 2. ALAN VERİLERİ (ESA, SRTM, Toprak) ---
def get_advanced_area_data(lat, lon, buffer_radius_m):
    try:
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(buffer_radius_m) 

        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        from_cls = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k =     [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        k_img = dataset.remap(from_cls, to_k, 0.5)
        mean_k = k_img.reduceRegion(ee.Reducer.mean(), area, 10).get("remapped").getInfo() or 0.5
        
        mode_cls = dataset.reduceRegion(ee.Reducer.mode(), area, 10).get("Map").getInfo()
        # v11.4 Düzeltmesi (Madde 4) için 'Su' (Sınıf 80) eklendi
        land_map = {10:"Ormanlık", 20:"Çalılık", 30:"Çayır/Park", 40:"Tarım", 50:"Kentsel/Beton", 60:"Çıplak", 80:"Su"}
        land_type = land_map.get(mode_cls, "Karma Alan")

        soil_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select("b0")
        
        # --- v11.4 Ölçek Düzeltmesi (Madde 3) ---
        # 250m buffer ve 250m ölçek kaldırıldı. 
        # Analiz alanı 'area' (100m) ve ölçek '10m' ile (ESA gibi) eşleştirildi.
        soil_mode = soil_img.reduceRegion(ee.Reducer.mode(), area, 10).get("b0").getInfo()
        # --- (Düzeltme Sonu) ---
        
        soil_factor = 1.0
        soil_desc = "Normal Toprak"
        if soil_mode:
            sm = int(soil_mode)
            if sm in [1, 2, 3]: soil_factor, soil_desc = 1.25, "Killi (Geçirimsiz)"
            elif sm in [9, 10, 11, 12]: soil_factor, soil_desc = 0.85, "Kumlu (Geçirgen)"

        dem = ee.Image("USGS/SRTMGL1_003")
        slope_val = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), area.buffer(30), 30).get("slope").getInfo()
        slope_pct = float(slope_val) * 1.5 if slope_val else 0.0 

        # --- v11.3 DEĞİŞİKLİK (Korundu) ---
        elevation_result = dem.reduceRegion(ee.Reducer.mean(), area, 30).get("elevation")
        elevation = elevation_result.getInfo()
        elevation = float(elevation) if elevation is not None else 0.0 

        return mean_k, soil_factor, land_type, soil_desc, slope_pct, elevation
    except: 
        return 0.5, 1.0, "Bilinmiyor", "Bilinmiyor", 0.0, 0.0

# --- 3. YAĞIŞ ---
def get_rain_10years(lat, lon):
    # (Madde 2) - Bu fonksiyonda değişiklik yok, hesaplama doğru.
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2024-12-31&daily=precipitation_sum&timezone=UTC"
        r = requests.get(url, timeout=15).json()
        clean = [d for d in r["daily"]["precipitation_sum"] if d is not None]
        if not clean: return 0.5, 0.5, 50.0, 500.0
        meanA = sum(clean) / 10.0
        maxD = max(clean)
        
        W_star = clamp(meanA / 1000.0) 
        R_ext  = clamp(maxD / 120.0)   
        
        return W_star, R_ext, maxD, meanA
    except: return 0.5, 0.5, 50.0, 500.0

# ============================================================
#  ANA ANALİZ VE KARAR MEKANİZMASI (v11.4 - Kısmi)
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        lat, lon = float(d["lat"]), float(d["lon"])

        GEE_BUFFER_RADIUS_M = 100.0
        analysis_area_m2 = math.pi * (GEE_BUFFER_RADIUS_M ** 2.0)
        analysis_area_km2 = analysis_area_m2 / 1000000.0
        
        # --- v11.4 Kirpich L Düzeltmesi (Madde 10) ---
        L_flow = GEE_BUFFER_RADIUS_M * 2.0
        # --- (Düzeltme Sonu) ---

        # 1. VERİLERİ TOPLA
        K_cover, soil_factor, land_type, soil_desc, slope_pct, elevation = get_advanced_area_data(lat, lon, GEE_BUFFER_RADIUS_M)

        # --- v11.4 RAKIM + ARAZİ KONTROLÜ (Madde 4 Düzeltmesi) ---
        if elevation <= 0 or land_type == "Su":
            return jsonify({
                "status": "water", 
                "msg": "Analiz alanı (deniz, göl, nehir) bir su kütlesidir. Drenaj analizi yapılamaz.",
                "location_type": f"Su Kütlesi (Rakım: {elevation:.1f}m, Sınıf: {land_type})",
                "selected_system": "water",
                "FloodRiskLevel": "N/A"
            })
        # --- (Düzeltme Sonu) ---

        W_star, R_ext, maxRain, meanRain = get_rain_10years(lat, lon)
        ndvi = get_ndvi_data(lat, lon, GEE_BUFFER_RADIUS_M)

        # 2. HİDROLOJİK HESAPLAR
        
        # --- v11.4 Eğim Normalizasyonu Düzeltmesi (Madde 8) ---
        S = clamp(math.log(1 + slope_pct) / math.log(1 + 15.0))
        # --- (Düzeltme Sonu) ---

        # --- v11.4 veg_factor Kalibrasyonu (Madde 5) ---
        veg_factor = 1.0 - (ndvi * 0.30) if ndvi > 0.2 else 1.0
        # --- (Düzeltme Sonu) ---
        
        raw_C = 1.0 - K_cover
        C = clamp(raw_C * soil_factor * veg_factor) 
        K_final = 1.0 - C

        # --- v11.4 RİSK MODELİ (ARID PENALTY DÜZELTMESİ) ---
        W_blk = 0.6*W_star + 0.4*R_ext 
        S_risk = abs(2.0*S - 1.0) 
        
        Risk_Lin = 0.45*W_blk + 0.45*C + 0.10*S_risk 
        Risk_Pik = max(0, R_ext-0.75)*0.4 
        Baseline_Risk = Risk_Lin + Risk_Pik

        is_arid_factor = 1.0 - W_star
        
        # --- v11.4 AridPenalty Düzeltmesi (Madde 7) ---
        arid_penalty_weight = 0.3 
        Arid_Urban_Penalty = is_arid_factor * arid_penalty_weight
        # --- (Düzeltme Sonu) ---

        FloodRisk_raw = Baseline_Risk + Arid_Urban_Penalty
        FloodRisk = clamp(FloodRisk_raw)
        # --- (v11.4 Risk Modeli Sonu) ---


        # --- v11.4 AHP MODELİ (MENDERES HARİÇ DÜZELTİLDİ) ---
        S_mid = 1.0 - abs(2.0*S - 1.0)
        
        Sc_DEN = 0.40*S_mid + 0.40*FloodRisk + 0.20*(K_final)
        Sc_PAR = 0.50*S + 0.30*K_final + 0.20*(1-FloodRisk)
        Sc_RET = 0.80*C + 0.20*FloodRisk
        Sc_PIN = 0.50*S + 0.30*C + 0.20*W_star
        Sc_RAD = 0.70*(1.0-S) + 0.20*K_final + 0.10*FloodRisk
        
        # --- v11.4 MENDERES NOTU: KULLANICI TALEBİ ÜZERİNE ORİJİNAL (v11.3) HALİNDE BIRAKILDI ---
        # Bu formül, yüksek eğimi (S=1) ödüllendirir, düşük eğimi (S=0) cezalandırır.
        Sc_MEA = 0.80*S + 0.20*(1-C)
        # --- (Değişiklik Sonu) ---
        
        Sc_HYB = 0.35*FloodRisk + 0.35*C + 0.30*S_mid
        # --- (AHP Modeli Sonu) ---

        scores = {
            "dendritic": round(Sc_DEN, 3), "parallel": round(Sc_PAR, 3), "reticular": round(Sc_RET, 3),
            "pinnate": round(Sc_PIN, 3), "radial": round(Sc_RAD, 3), "meandering": round(Sc_MEA, 3), "hybrid": round(Sc_HYB, 3)
        }
        selected = max(scores, key=scores.get)

        # 5. KARAR AÇIKLAMASI (MENDERES HARİÇ DÜZELTİLDİ)
        reason_txt = "Bilinmiyor."
        if selected == "dendritic":
            reason_txt = f"Bölgedeki orta seviye eğim (%{slope_pct:.1f}) ve doğal akış hatlarının varlığı, suyu yerçekimiyle toplamak için en verimli olan ağaç dalları (Dendritik) yapısını öne çıkarmıştır."
        elif selected == "parallel":
            reason_txt = f"Arazi eğiminin tek yönlü ve düzenli olması (%{slope_pct:.1f}), suyun paralel hatlar boyunca en hızlı şekilde tahliye edilmesini gerektirmektedir."
        elif selected == "reticular":
            reason_txt = f"Bölgedeki yüksek geçirimsizlik (K={K_final:.2f}) ve karmaşık kentsel doku, çok yönlü akışa izin veren ağsı (Retiküler) yapıyı zorlu kılmıştır."
        elif selected == "pinnate":
            reason_txt = f"Dik eğimli (%{slope_pct:.1f}) ve dar koridor yapısındaki bu alanda, suyu ana hatta hızlıca iletmek için balık kılçığı (Pinnate) modeli seçilmiştir."
        elif selected == "radial":
            reason_txt = f"Arazinin düz yapısı (%{slope_pct:.1f} eğim) ve geçirgen zemini (K={K_final:.2f}), suyun merkezi bir yağmur bahçesi topladığı (Radyal) sistemi gerektirir."
        
        # --- v11.4 MENDERES NOTU: KULLANICI TALEBİ ÜZERİNE ORİJİNAL (v11.3) HALİNDE BIRAKILDI ---
        # Bu gerekçe, hatalı AHP formülüne uygun olarak "çok dik eğim" uyarısı verir.
        elif selected == "meandering":
            reason_txt = f"UYARI: Bölgedeki çok dik eğim (%{slope_pct:.1f}), suyun hızını ve erozyon riskini artırmaktadır. Bu nedenle suyu yavaşlatarak taşıyan kıvrımlı (Menderes) yapı seçilmiştir."
        # --- (Değişiklik Sonu) ---
        
        else:
            reason_txt = "Bölgedeki karmaşık topografya ve değişken zemin yapısı, birden fazla sistemin özelliklerini taşıyan Hibrit bir çözümü gerektirmektedir."

        # 6. HİDROLİK (v11.4 Kirpich düzeltmesi yukarıda yapıldı)
        Climate_Factor = 1.15 
        
        n_roughness = 0.025 if selected in ["meandering", "radial"] else 0.013
        
        S_metric = max(0.01, slope_pct / 100.0)
        # t_c hesabı artık L_flow = 200m kullandığı için doğru (daha uzun) bir süre verecektir.
        t_c = max(5.0, min(45.0, 0.0195 * (math.pow(L_flow, 0.77) / math.pow(S_metric, 0.385))))

        F_iklim = 2.5 
        if meanRain > 1500: F_iklim = 2.2 
        elif meanRain < 400: F_iklim = 3.5
        elif 800 < meanRain <= 1500: F_iklim = 3.0
        
        i_val = (maxRain * F_iklim) / ((t_c/60.0 + 0.15)**0.7)
        
        Q_future = (0.278 * C * i_val * analysis_area_km2) * Climate_Factor 
        
        S_bed = max(0.005, S_metric)
        D_mm = (((4**(5/3)) * n_roughness * Q_future) / (math.pi * math.sqrt(S_bed)))**(3/8) * 1000.0
        
        mat = "PVC"
        if selected in ["meandering", "radial"]: mat = "Doğal Taş Kanal"
        elif D_mm >= 500: mat = "Betonarme"
        elif D_mm >= 200: mat = "Koruge (HDPE)"
        
        bio_solution = "Standart Peyzaj"
        if C > 0.7: bio_solution = "Yeşil Çatı & Geçirimli Beton"
        elif slope_pct > 15: bio_solution = "Teraslama & Erozyon Önleyici"
        elif selected == "radial": bio_solution = "Yağmur Bahçesi (Rain Garden)"
        elif selected == "dendritic": bio_solution = "Biyo-Hendek (Bioswale)"
        
        harvest = (analysis_area_m2 * (meanRain/1000.0) * 0.85 * (1.0 - K_final)) 

        lvl_idx = int(FloodRisk * 4.9)
        lvl = ["Çok Düşük","Düşük","Orta","Yüksek","Kritik"][min(lvl_idx, 4)]

        return jsonify({
            "status": "success",
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
            "rain_stats": { "mean": round(meanRain, 1), "max": round(maxRain, 1) },
            "eco_stats": { "harvest": round(harvest, 0), "bio_solution": bio_solution },
            "debug_analysis_area_ha": round(analysis_area_m2 / 10000.0, 2),
            "debug_analysis_radius_m": GEE_BUFFER_RADIUS_M,
            "debug_elevation": elevation 
        })

    except Exception as e:
        app.logger.error(f"Analysis Error: {e}") 
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"status":"error", "msg":str(e)}), 500

if __name__ == "__main__":
    app.run()

application = app
