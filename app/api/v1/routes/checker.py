import requests
import os
import google.generativeai as genai
from app.core.config import settings
import joblib
import pandas as pd
import pickle


# --- Load ML Models ---
def load_models():
    """Load the pre-trained spirulina model and scaler from pickle files."""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../../../', 'spirulina_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), '../../../', 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

# Load models at startup
SPIRULINA_MODEL, SCALER = load_models()


# --- Biomass Prediction Function ---
def predict_biomass(site_profile):
    """
    Predict spirulina biomass based on site profile data using the ML model.
    
    Expected input: site_profile dict with climate and water_profile data
    Returns: Predicted biomass value and confidence metrics
    """
    if SPIRULINA_MODEL is None or SCALER is None:
        return {"error": "ML models not loaded"}
    
    try:
        # Extract features from site profile
        if "error" in site_profile:
            return {"error": "Cannot predict: Invalid site profile"}
        
        climate = site_profile.get("climate", {})
        water_profile = site_profile.get("water_profile", {})
        
        # Prepare features in the order expected by the model
        # Adjust these feature names based on your model's training features
        features = {
            "temperature": climate.get("temperature", 0),
            "solar_radiation": climate.get("solar_radiation", 0),
            "pH": water_profile.get("initial_pH", 7.6),
            "salinity": water_profile.get("salinity", 0),
            "NaNO3": water_profile.get("NaNO3", 0),
            "NaHCO3": water_profile.get("NaHCO3", 0),
            "K2HPO4": water_profile.get("K2HPO4", 0),
            "light_time": water_profile.get("light_time", 0),
            "aeration": water_profile.get("aeration", 0)
        }
        
        # Create DataFrame for scaler transformation
        df = pd.DataFrame([features])
        
        # Scale features
        scaled_features = SCALER.transform(df)
        
        # Make prediction
        biomass_prediction = SPIRULINA_MODEL.predict(scaled_features)[0]
        
        # Get confidence (if available from model)
        confidence = None
        if hasattr(SPIRULINA_MODEL, 'predict_proba'):
            confidence = SPIRULINA_MODEL.predict_proba(scaled_features)[0]
        
        return {
            "biomass_prediction": round(float(biomass_prediction), 2),
            "unit": "g/L or kg/hectare",  # Adjust based on your model
            "confidence": confidence,
            "input_features": features
        }
    
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


# --- Geocoding Function ---
def geocode_place(place_name):
    """
    Fetch latitude and longitude for a place name using OpenStreetMap Nominatim.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place_name,
        "format": "json",
        "limit": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10, headers={"User-Agent": "SpirulinaApp/1.0"})
        data = response.json()
        if not data:
            return None, None, "Place not found."
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon, None
    except Exception:
        return None, None, "Error fetching coordinates."

def validate_coordinates(latitude, longitude):
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return False, "Latitude and longitude must be numeric."
    if latitude < -90 or latitude > 90:
        return False, "Latitude must be between -90 and 90."
    if longitude < -180 or longitude > 180:
        return False, "Longitude must be between -180 and 180."
    return True, "Valid coordinates"

def fetch_nasa_power_data(latitude, longitude):
    """
    Fetch monthly averaged climate data from NASA POWER API.
    Returns average temperature (°C) and solar radiation (MJ/m²/day).
    """

    base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

    params = {
        "parameters": "T2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": "2020",
        "end": "2020",
        "format": "JSON"
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        climate = data["properties"]["parameter"]

        avg_temp = sum(climate["T2M"].values()) / len(climate["T2M"])
        avg_radiation = sum(climate["ALLSKY_SFC_SW_DWN"].values()) / len(
            climate["ALLSKY_SFC_SW_DWN"]
        )

        return {
            "temperature": round(avg_temp, 2),
            "solar_radiation": round(avg_radiation, 2)
        }

    except Exception:
        return None

def check_cultivation_validity(climate):
    """
    Decide whether a location is suitable for Spirulina cultivation
    using NON-CORRECTABLE climatic constraints.
    """

    temp = climate["temperature"]
    radiation = climate["solar_radiation"]

    reasons = []

    # Temperature rules
    if temp < 20:
        reasons.append("Average temperature is too low for Spirulina growth.")
        return "INVALID", reasons
    elif 20 <= temp < 25:
        reasons.append("Temperature is marginal for optimal Spirulina growth.")

    # Solar radiation rules
    if radiation < 12:
        reasons.append("Solar radiation is insufficient for photosynthesis.")
        return "INVALID", reasons
    elif 12 <= radiation < 16:
        reasons.append("Solar radiation is marginal for high productivity.")

    if reasons:
        return "MARGINAL", reasons

    return "VALID", ["Climatic conditions are suitable for Spirulina cultivation."]

def classify_region(latitude, temperature):
    if abs(latitude) < 23.5:
        return "Tropical / Coastal"
    elif temperature > 30:
        return "Arid / Semi-Arid"
    else:
        return "Inland / Temperate"

def infer_water_profile(latitude, climate):

    region = classify_region(latitude, climate["temperature"])

    if region == "Tropical / Coastal":
        pH = 8.2
        salinity = 2.8
    elif region == "Arid / Semi-Arid":
        pH = 8.8
        salinity = 3.0
    else:
        pH = 7.6
        salinity = 0.9

    return {
        "region": region,
        "initial_pH": pH,
        "salinity": salinity,
        "NaNO3": 1.6,
        "NaHCO3": 15,
        "K2HPO4": 0.7,
        "light_time": 14,
        "aeration": 1
    }

def generate_site_profile(latitude, longitude):

    valid, msg = validate_coordinates(latitude, longitude)
    if not valid:
        return {"error": msg}

    climate = fetch_nasa_power_data(latitude, longitude)
    if climate is None:
        return {"error": "Unable to fetch climate data from NASA POWER API."}

    status, reasons = check_cultivation_validity(climate)

    result = {
        "location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "climate": climate,
        "cultivation_status": status,
        "reasons": reasons
    }
    if status != "INVALID":
        result["water_profile"] = infer_water_profile(latitude, climate)

    return result

# --- Real LLM Function (Google AI Studio) ---
def llm_generate_summary(site_profile):
    """
    Expert Consultant: Compares site data with biological standards 
    and generates corrective actions and market strategies.
    """
    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        return "LLM API key not set."

    genai.configure(api_key=api_key)

    # We provide the "Ideal Standards" from your research paper to the LLM
    # This replaces your 'if-else' logic with dynamic reasoning.
    optimal_standards = """
    SPIRULINA BIOLOGICAL STANDARDS:
    - Temperature: Optimal 30°C - 37°C. (Below 20°C growth stops; Above 40°C cells die).
    - pH: Optimal 8.5 - 11.0. (Requires alkaline environment).
    - Solar Radiation: Minimum 12 MJ/m²/day for photosynthesis.
    - Salinity: 0.5 - 3.5%.
    - Essential Nutrients: NaNO3 (Nitrogen source), NaHCO3 (Carbon/Alkalinity), K2HPO4 (Phosphorus).
    """

    if "error" in site_profile:
        return f"Analysis Error: {site_profile['error']}"

    # REFINED PROMPT: Focuses on "How to improve" and "Business Strategy"
    prompt = f"""
    You are a PhD specialist in Algal Biotechnology.
    
    SYSTEM CONTEXT:
    {optimal_standards}

    SITE DATA TO ANALYZE:
    {site_profile}

    TASK:
    1. EVALUATE: Compare the Site Data against the Biological Standards.
    2. CORRECTIVE ACTIONS: If any parameter (Temperature, pH, Nutrients) is sub-optimal, 
       provide specific technical suggestions (e.g., Greenhouse installation, 
       adding NaHCO3 for pH, or using water heaters). DO NOT use generic advice.
    3. MARKET STRATEGY: Based on the cultivation status, suggest which industry should 
       this spirulina be sold to (Pharmaceutical, Food Industry, or Animal Feed).
    4. FORMAT: Use professional headings. Make it actionable for a farmer.
    """

    try:
        # Use the correct free tier model
        model = genai.GenerativeModel('gemini-3-flash-preview')  # Free tier model
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"LLM error: {str(e)}"

# --- Updated Main Orchestrator ---
def analyze_location(place_name: str):
    lat, lon, error = geocode_place(place_name)
    if error: return {"error": error}
    
    site_profile = generate_site_profile(lat, lon)
    
    # Predict biomass based on site profile
    biomass_result = predict_biomass(site_profile)
    
    # The LLM now handles all "Suggestions" and "Improvements"
    summary = llm_generate_summary(site_profile)
    
    return {
        "place": place_name,
        "coordinates": {"lat": lat, "lon": lon},
        "biomass_prediction": biomass_result,
        "report": summary
    }