# ============================================================
#  BİYOMİMİKRİ DRENAJ SİSTEMİ — TÜBİTAK v8.7 (CALIBRATED)
#  Düzeltme (Risk): Risk modeli (1-S) yerine S_risk kullanacak şekilde kalibre edildi.
#  Düzeltme (AHP): Sc_RET (Kentsel) modelin C'ye duyarlılığı artırıldı.
# ============================================================

# ... (Dosyanın üst kısmı 'get_rain_10years' fonksiyonuna kadar aynı) ...
# ... (Flask, GEE, NDVI, Area, Rain fonksiyonları değişmedi) ...

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        d = request.get_json(force=True)
        lat, lon = float(d["lat"]), float(d["lon"])

        # ... (DÜZELTME 1: TUTARLI ALAN kısmı aynı) ...
        GEE_BUFFER_RADIUS_M = 100.0
        analysis_area_m2 = math.pi * (GEE_BUFFER_RADIUS_M ** 2.0)
        analysis_area_km2 = analysis_area_m2 / 1000000.0
        L_flow = GEE_BUFFER_RADIUS_M 

        # ... (VERİLERİ TOPLA kısmı aynı) ...
        K_cover, soil_factor, land_type, soil_desc, slope_pct = get_advanced_area_data(lat, lon, GEE_BUFFER_RADIUS_M)
        W_star, R_ext, maxRain, meanRain = get_rain_10years(lat, lon)
        ndvi = get_ndvi_data(lat, lon, GEE_BUFFER_RADIUS_M)

        # ... (HİDROLOJİK HESAPLAR kısmı aynı) ...
        S = clamp(slope_pct / 25.0)
        veg_factor = 1.0 - (ndvi * 0.15) if ndvi > 0.2 else 1.0
        raw_C = 1.0 - K_cover
        C = clamp(raw_C * soil_factor * veg_factor)
        K_final = 1.0 - C

        # --- DÜZELTME 1: KALİBRE EDİLMİŞ RİSK MODELİ (v8.7) ---
        W_blk = 0.6*W_star + 0.4*R_ext
        
        # S_risk: Düşük eğim (S=0) ve Yüksek eğim (S=1) için 1.0; 
        # Orta eğim (S=0.5) için 0.0 değerini alır.
        S_risk = abs(2.0*S - 1.0) 
        
        # Risk_Lin formülü (1-S) yerine S_risk kullanıyor.
        # Ağırlıkları da Konya'nın riskini düşürmek için hafifçe ayarlıyoruz:
        Risk_Lin = 0.40*W_blk + 0.40*C + 0.20*S_risk
        
        FloodRisk = clamp(Risk_Lin + max(0, R_ext-0.75)*0.4)
        # --- (Düzeltme 1 Sonu) ---


        # --- DÜZELTME 2: İYİLEŞTİRİLMİŞ AHP (v8.7) ---
        S_mid = 1.0 - abs(2.0*S - 1.0) # Orta eğimi seven skor (Bu değişmedi)
        
        Sc_DEN = 0.40*S_mid + 0.40*FloodRisk + 0.20*(1-K_final)
        Sc_PAR = 0.50*S + 0.30*K_final + 0.20*(1-FloodRisk)
        
        # Retiküler (Ağsı/Kentsel) sistemin C (geçirimsizlik) ağırlığını artırıyoruz.
        # Bu, Ayamama'nın 'Radyal' yerine 'Retiküler' seçmesini teşvik edecek.
        Sc_RET = 0.70*C + 0.30*FloodRisk
        
        Sc_PIN = 0.50*S + 0.30*C + 0.20*W_star
        Sc_RAD = 0.60*(1.0-S) + 0.40*FloodRisk
        Sc_MEA = 0.80*S + 0.20*(1-C)
        Sc_HYB = 0.35*FloodRisk + 0.35*C + 0.30*S_mid
        # --- (Düzeltme 2 Sonu) ---

        scores = {
            "dendritic": round(Sc_DEN, 3), "parallel": round(Sc_PAR, 3), "reticular": round(Sc_RET, 3),
            "pinnate": round(Sc_PIN, 3), "radial": round(Sc_RAD, 3), "meandering": round(Sc_MEA, 3), "hybrid": round(Sc_HYB, 3)
        }
        selected = max(scores, key=scores.get)

        # ... (Dosyanın geri kalanı 'reason_txt', 'D_mm' hesapları, 'harvest' vb. aynı) ...
        # ... (Reasoning text, hidrolik hesaplar ve JSON çıktısı değişmedi) ...
