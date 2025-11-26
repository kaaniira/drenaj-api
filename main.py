# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK v7.3 (ALAN BAZLI)
#  Analiz Yöntemi: 100m Çaplı Mikro-Havza Ortalaması
#  Veri Kaynakları: ESA WorldCover 10m & NASA SRTM & Open-Meteo
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
        print("GEE: Alan bazlı analiz için hazır.")
    except Exception as e:
        print(f"GEE Başlatma Hatası: {e}")
else:
    print("UYARI: GEE Key dosyası bulunamadı.")

# ============================================================
#  1. ALAN BAZLI GEÇİRGENLİK (K) HESABI
#  Yöntem: 100m çapındaki alanda ağırlıklı ortalama
# ============================================================
def get_area_weighted_K(lat, lon):
    try:
        # Merkez nokta ve 50m yarıçaplı (100m çaplı) tampon bölge
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(50) 

        # ESA WorldCover 10m Verisi
        dataset = ee.ImageCollection("ESA/WorldCover/v200").first()

        # --- SINIFLARI K DEĞERLERİNE DÖNÜŞTÜRME (REMAPPING) ---
        # ESA Sınıfları: 10(Ağaç), 20(Çalı), 30(Çim), 40(Tarım), 50(Beton/Şehir), 60(Çıplak), 80(Su)
        # K Değerleri:   0.90,     0.80,     0.85,     0.60,      0.15,          0.50,       0.00
        
        from_classes = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_k_values  = [0.90, 0.80, 0.85, 0.60, 0.15, 0.50, 0.00, 0.00, 0.90, 0.90]
        
        # Görüntüdeki sınıf kodlarını K değerleriyle değiştiriyoruz
        k_image = dataset.remap(from_classes, to_k_values)

        # Belirlenen alandaki ORTALAMA K değerini hesapla
        mean_dict = k_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=10, # 10m çözünürlükte analiz
            maxPixels=1e9
        )
        
        # Sonuç "remapped" adında döner
        K_avg = mean_dict.get("remapped").getInfo()

        # Eğer hesaplanamazsa varsayılan dön
        if K_avg is None: 
            return 0.5, "Veri Yok"

        # Baskın türü bulmak için (Arayüzde göstermek maksatlı)
        mode_dict = dataset.reduceRegion(
            reducer=ee.Reducer.mode(),
            geometry=area,
            scale=10
        )
        dominant_class = mode_dict.get("Map").getInfo()
        
        class_names = {
            10: "Ormanlık Alan", 20: "Çalılık", 30: "Çayır/Park", 40: "Tarım Arazisi",
            50: "Kentsel/Beton", 60: "Çıplak Arazi", 80: "Su Yüzeyi"
        }
        land_desc = class_names.get(dominant_class, "Karma Alan")

        print(f"Alan Analizi: K={K_avg:.2f}, Tip={land_desc}")
        return float(K_avg), land_desc

    except Exception as e:
        print(f"GEE K Hatası: {e}")
        return 0.5, "Bilinmiyor"

# ============================================================
#  2. ALAN BAZLI EĞİM (S) HESABI
#  Yöntem: NASA SRTM verisinden 100m çapındaki ortalama eğim
# ============================================================
def get_area_avg_slope(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        area = point.buffer(50) # 50m yarıçap

        # NASA SRTM Dijital Yükseklik Modeli (DEM)
        dem = ee.Image("USGS/SRTMGL1_003")
        
        # Yükseklik verisinden EĞİM haritası üret (Derece cinsinden)
        slope_img = ee.Terrain.slope(dem)
        
        # Bölgedeki ortalama eğimi al
        mean_slope = slope_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=30 # SRTM çözünürlüğü 30m
        ).get("slope").getInfo()
        
        if mean_slope is None: return 0.0
        
        # Dereceyi Yüzdeye çevir (tan(derece) * 100)
        # Ancak küçük açılar için yaklaşık olarak birebir kullanabiliriz veya çevirebiliriz.
        # Basit yaklaşım: Eğim yüzdesi ≈ Eğim derecesi * 1.7
        slope_percent = float(mean_slope) * 1.5 
        
        return slope_percent

    except Exception as e:
        print(f"GEE Slope Hatası: {e}")
        return 2.0 # Varsayılan hafif eğim

# ============================================================
#  3. YARDIMCI MATEMATİK & YAĞIŞ
# ============================================================
def clamp(v, vmin=0.0, vmax=1.0):
    return max(vmin, min(vmax, v))

def get_rain_data(lat, lon):
    # Yağış verisi geniş ölçekli olduğu için nokta bazlı kalabilir
    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            "&start_date=2023-01-01&end_date=2023-12-31"
            "&daily=precipitation_sum&timezone=UTC"
        )
        r = requests.get(url, timeout=5)
        daily = r.json()["daily"]["precipitation_sum"]
        clean = [d for d in daily if d is not None]
        if not clean: return 0.5, 0.5, 50.0
        
        meanA = sum(clean)
        maxD = max(clean)
        
        W_star = clamp(meanA / 1000.0)
        R_extreme = clamp(maxD / 120.0) # 120mm üstü sel kabul
        
        return W_star, R_extreme, maxD
    except:
        return 0.5, 0.5, 50.0

# ============================================================
#  4. ANA ANALİZ ENDPOINT
# ============================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    lat = float(data["lat"])
    lon = float(data["lon"])

    # --- 100m ÇAPLI ALAN VERİLERİ ---
    slope_percent = get_area_avg_slope(lat, lon)
    K_avg, land_type = get_area_weighted_K(lat, lon) # Ortalama K
    W_star, R_extreme, maxDailyRain = get_rain_data(lat, lon)
    
    # --- DEĞİŞKENLER ---
    # Eğim Skoru (S)
    S = clamp(slope_percent / 25.0) # %25 eğim üzeri 1.0 (Dik)
    
    # Kentsel Etki (C)
    # Artık D (Bina yoğunluğu) yerine doğrudan (1 - K_avg) kullanıyoruz.
    # Çünkü K_avg zaten alandaki betonlaşma oranını içeriyor.
    C = 1.0 - K_avg 
    
    # Bloklar
    W_block = 0.6 * W_star + 0.4 * R_extreme
    S_flat = 1.0 - S
    
    # --- SEL RİSKİ (FloodRisk) ---
    FloodRisk_lin = 0.36 * W_block + 0.34 * C + 0.30 * S_flat
    Boost = max(0.0, R_extreme - 0.75) * 0.4
    FloodRisk = clamp(FloodRisk_lin + Boost)
    
    # --- SİSTEM SEÇİMİ (AHP - Alan Bazlı Veriyle) ---
    # Formüllerde D parametresi yerine C'yi (veya C türevini) kullanarak sadeleştirdik
    
    # 1. Dendritik: Eğimli ve geçirimsiz alanlar
    Score_DEN = 0.50 * S + 0.30 * FloodRisk + 0.20 * C
    
    # 2. Paralel: Düz ve yüksek geçirgenlikli alanlar
    Score_PAR = 0.45 * (1.0 - S) + 0.35 * K_avg + 0.20 * (1.0 - FloodRisk)
    
    # 3. Retiküler: Çok yoğun betonlaşma (C yüksek) ve yüksek risk
    Score_RET = 0.60 * C + 0.40 * FloodRisk
    
    # 4. Hibrit: Arazi karmaşık ise
    S_mid = 1.0 - abs(2.0 * S - 1.0)
    Score_HYB = 0.35 * FloodRisk + 0.35 * C + 0.30 * S_mid
    
    scores = {
        "dendritic": Score_DEN,
        "parallel": Score_PAR,
        "reticular": Score_RET,
        "hybrid": Score_HYB
    }
    selected_system = max(scores, key=scores.get)
    
    # --- HİDROLİK VE BORU ÇAPI ---
    # Sokak bazlı havza varsayımı (100m çaplı dairenin alanı ≈ 7850 m2 ≈ 0.78 hektar)
    # Ancak yolun devamını da topladığı için biraz daha büyük kabul edelim:
    A_ha = 1.5 
    
    # Konsantrasyon Süresi (t_c)
    # Eğim arttıkça su hızlanır, süre kısalır
    t_c = 20.0 - (S * 10.0) # Min 10 dk, Max 20 dk
    
    # IDF (Yağış Şiddeti)
    i_mm_h = (maxDailyRain * 1.5) / ((t_c/60.0 + 0.15) ** 0.7)
    
    # Debi (Q) - Rasyonel Metot
    # Q = C * i * A / 360 (Metrik dönüşüm katsayısı ile 0.278)
    Q = 0.278 * C * i_mm_h * A_ha
    
    # Manning
    n = 0.013
    S_bed = max(0.005, slope_percent / 100.0) # Boru eğimi
    
    D_m = (( (4**(5/3)) * n * Q ) / (math.pi * math.sqrt(S_bed)) )**(3/8)
    D_mm = D_m * 1000.0
    
    # Malzeme
    if D_mm >= 1000: material = "Betonarme / GRP"
    elif D_mm >= 400: material = "Betonarme"
    elif D_mm >= 200: material = "HDPE (Koruge)"
    else: material = "PVC"

    # Risk Seviyesi Metni
    if FloodRisk < 0.25: risk_txt = "Çok Düşük"
    elif FloodRisk < 0.45: risk_txt = "Düşük"
    elif FloodRisk < 0.65: risk_txt = "Orta"
    elif FloodRisk < 0.80: risk_txt = "Yüksek"
    else: risk_txt = "Kritik / Çok Yüksek"

    return jsonify({
        "location_type": f"{land_type} (100m Çap Ortalaması)",
        "K_value": round(K_avg, 3),
        "slope_percent": round(slope_percent, 2),
        "FloodRisk": round(FloodRisk, 3),
        "FloodRiskLevel": risk_txt,
        "selected_system": selected_system,
        "pipe_diameter_mm": round(D_mm, 1),
        "material": material,
        "Q_flow": round(Q, 4),
        "scores": scores
    })

if __name__ == "__main__":
    app.run()


application = app
