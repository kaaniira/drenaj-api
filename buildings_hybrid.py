import requests
import math

# -------------------------------
# 1) OSM BİNA FOOTPRINTLERİ
# -------------------------------

def fetch_osm_buildings(lat, lon, radius=200):
    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius},{lat},{lon})["building"];
      relation(around:{radius},{lat},{lon})["building"];
    );
    out geom;
    """
    try:
        r = requests.post("https://overpass-api.de/api/interpreter",
                          data={"data": query}, timeout=25)
        r.raise_for_status()
        data = r.json().get("elements", [])
    except Exception as e:
        return 0, f"OSM error: {e}"

    count = 0
    for el in data:
        if el.get("type") in ["way", "relation"]:
            count += 1

    return count, None


# -------------------------------
# 2) GOOGLE PLACES BİNA YOĞUNLUĞU
# -------------------------------

def fetch_google_places(lat, lon, radius, key):
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lon}&radius={radius}"
        "&type=establishment"
        f"&key={key}"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        return len(results), None
    except Exception as e:
        return 0, f"Google error: {e}"


# -------------------------------
# 3) HİBRİT BİNA SAYIM ALGORİTMASI
# -------------------------------

def hybrid_building_count(lat, lon, radius, google_key):
    osm_count, osm_err = fetch_osm_buildings(lat, lon, radius)
    poi_count, poi_err = fetch_google_places(lat, lon, radius, google_key)

    # Weighted fusion (kalibre edilmiş)
    fused = osm_count + poi_count * 0.65

    errors = []
    if osm_err:
        errors.append(osm_err)
    if poi_err:
        errors.append(poi_err)

    return fused, ", ".join(errors) if errors else None


# -------------------------------
# 4) YOĞUNLUĞA (bin/km²) ÇEVİRME
# -------------------------------

def density_per_km2(building_count, radius):
    area_km2 = math.pi * (radius/1000)**2
    return building_count / area_km2 if area_km2 > 0 else 0


# -------------------------------
# 5) NORMALİZE YOĞUNLUK (D)
# -------------------------------

def normalize_D(dens):
    if dens <= 500:
        return 0
    if dens >= 5000:
        return 1
    return (dens - 500) / (4500)
