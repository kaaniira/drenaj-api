# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK v7.5 (GENİŞLETİLMİŞ)
#  Analiz: 7 Farklı Biyomimikri Modeli
#  Veri: ESA WorldCover 10m & SRTM & Open-Meteo
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
        print("GEE Başlatıldı (v7.5)")
    except Exception as e:
        print(f"GEE Hatası: {e}")
else:
    print("UYARI: GEE Key yok.")

# ============================================================
#  YARDIMCI FONKSİYONLAR
# ============================================================
def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def get_area_data(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(50) # 100m Çap (50m Yarıçap)

        # 1. ARAZİ TİPİ VE K (ESA 10m Verisi)
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()
        
        # Sınıfları Geçirgenlik (K) değerlerine çeviriyoruz
        # 10:Orman(0.9), 20:Çalı(0.8), 30:Çim(0.85), 40:Tarım(0.6), 50:Beton(0.15), 60:Çıplak(0.5), 80:Su(0.0)
        from_cls = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k =     [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        
        k_img = dataset.remap(from_cls, to_k, 0.5) # Eşleşmeyenler 0.5 olsun
        
        # Bölgedeki ORTALAMA K değeri
        mean_k = k_img.reduceRegion(ee.Reducer.mean(), area, 10).get("remapped").getInfo()
        
        # Bölgedeki BASKIN arazi tipi (Mod)
        mode_cls = dataset.reduceRegion(ee.Reducer.mode(), area, 10).get("Map").getInfo()
        
        land_map = {
            10:"Ormanlık", 20:"Çalılık", 30:"Çayır/Park", 40:"Tarım", 
            50:"Kentsel/Beton", 60:"Çıplak Arazi", 80:"Su Yüzeyi"
        }
        land_type = land_map.get(mode_cls, "Karma Alan")
        
        # 2. EĞİM (SRTM 30m)
        dem = ee.Image("USGS/SRTMGL1_003")
        slope_val = ee.Terrain.slope(dem).reduceRegion(ee.Reducer.mean(), area, 30).get("slope").getInfo()
        slope_pct = float(slope_val) * 1.5 if slope_val else 0.0

        return float(mean_k or 0.5), land_type, slope_pct
    except Exception as e:
        print(f"Alan Analiz Hatası: {e}")
        return 0.5, "Veri Yok", 0.0

def get_rain(lat, lon):
    try:
        # 2023 Yılı Yağış Verisi
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date=2023-12-31&daily=precipitation_sum&timezone=UTC"
        r = requests.get(url, timeout=4).json()
        daily = [d for d in r["daily"]["precipitation_sum"] if d is not None]
        if not daily: return 0.5, 0.5, 50
        
        meanA = sum(daily)
        maxD = max(daily)
        
        # W* (Yıllık Yağış Skoru) ve R_ext (Aşırı Yağış Skoru)
        return clamp(meanA/1000.0), clamp(maxD/120.0), maxD
    except:
        return 0.5, 0.5, 50.0

# ============================================================
#  ANA ANALİZ (7 SİSTEMLİ AHP)
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        lat, lon = float(d["lat"]), float(d["lon"])

        # Verileri Çek
        K, land, slope_pct = get_area_data(lat, lon)
        W_star, R_ext, maxRain = get_rain(lat, lon)

        # Temel Değişkenler
        S = clamp(slope_pct / 25.0)   # Eğim Skoru (1.0 = %25 ve üzeri)
        C = 1.0 - K                   # Betonlaşma/Geçirimsizlik
        
        # Risk Hesabı
        W_blk = 0.6*W_star + 0.4*R_ext
        Risk_Lin = 0.36*W_blk + 0.34*C + 0.30*(1-S)
        FloodRisk = clamp(Risk_Lin + max(0, R_ext-0.75)*0.4)

        # --- AHP SİSTEM SEÇİMİ (7 Model) ---
        
        # 1. Dendritik (Ağaçsı): Dengeli eğim ve orta risk
        Sc_DEN = 0.40*S + 0.40*FloodRisk + 0.20*(1-K)
        
        # 2. Paralel (Mısır Yaprağı): Tek yönlü, düzenli eğim
        Sc_PAR = 0.50*(1-S) + 0.30*K + 0.20*(1-FloodRisk)
        
        # 3. Retiküler (Ağsı): Yoğun şehir merkezi, çok beton
        Sc_RET = 0.60*C + 0.40*FloodRisk
        
        # 4. Pinnate (Tüysü/Eğrelti): Dik, dar koridorlar (Yollar için hızlı tahliye)
        Sc_PIN = 0.50*S + 0.30*C + 0.20*W_star
        
        # 5. Radial (Radyal/Örümcek): Düz alan, çukur veya meydan (Merkezi toplanma)
        Sc_RAD = 0.60*(1.0-S) + 0.40*FloodRisk
        
        # 6. Meandering (Menderes/Yılan): Çok dik yokuşlar (Erozyonu önlemek için yavaşlatma)
        # Eğim çok yüksekse (S > 0.8) bu sistemin puanı artar.
        Sc_MEA = 0.80*S + 0.20*(1-C) 
        
        # 7. Hibrit: Kararsız durumlar
        S_mid = 1.0 - abs(2.0*S - 1.0)
        Sc_HYB = 0.35*FloodRisk + 0.35*C + 0.30*S_mid

        scores = {
            "dendritic": Sc_DEN, "parallel": Sc_PAR, "reticular": Sc_RET,
            "pinnate": Sc_PIN, "radial": Sc_RAD, "meandering": Sc_MEA, "hybrid": Sc_HYB
        }
        selected = max(scores, key=scores.get)

        # Hidrolik Hesap (Manning)
        # Eğim arttıkça su hızlanır, konsantrasyon süresi (t_c) düşer
        t_c = max(5.0, 20.0 - S*10.0) 
        
        i_val = (maxRain * 1.5) / ((t_c/60.0 + 0.15)**0.7) # IDF
        Q = 0.278 * C * i_val * 1.5 # Debi (1.5 ha kabulü)
        
        S_bed = max(0.005, slope_pct/100.0) # Boru eğimi
        D_mm = (((4**(5/3))*0.013*Q) / (math.pi * math.sqrt(S_bed)))**(3/8) * 1000.0
        
        # Malzeme Önerisi
        mat = "PVC" if D_mm<200 else ("HDPE (Koruge)" if D_mm<500 else "Beton/GRP")
        
        # Risk Seviyesi Metni
        lvl_idx = int(FloodRisk * 4.9)
        levels = ["Çok Düşük","Düşük","Orta","Yüksek","Kritik"]
        lvl = levels[lvl_idx] if lvl_idx < 5 else "Kritik"

        return jsonify({
            "status": "success",
            "location_type": land,
            "K_value": round(K, 2),
            "slope_percent": round(slope_pct, 2),
            "FloodRisk": round(FloodRisk, 2),
            "FloodRiskLevel": lvl,
            "selected_system": selected,
            "scores": scores,
            "pipe_diameter_mm": round(D_mm, 0),
            "material": mat,
            "Q_flow": round(Q, 3)
        })

    except Exception as e:
        return jsonify({"status":"error", "msg":str(e)}), 500

@app.route("/")
def home():
    return "TÜBİTAK Drenaj API v7.5 (7 Model Aktif)"

if __name__ == "__main__":
    app.run()

# GUNICORN İÇİN GEREKLİ SATIR
application = app
