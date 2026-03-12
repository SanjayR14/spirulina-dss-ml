from fastapi import APIRouter, Query
from app.api.v1.routes.checker import analyze_location

router = APIRouter()

@router.get("/analyze")
def analyze_site(location: str = Query(..., description="Place name to analyze for Spirulina cultivation")):
    """
    Analyze a location for Spirulina cultivation suitability.

    This endpoint orchestrates:
    - Geocoding of the place name
    - Fetching NASA POWER climate data
    - Inferring a water profile & cultivation status
    - Predicting Spirulina biomass using the ML model
    - Generating a PhD-style narrative report via LLM

    The response is structured so that frontends can:
    - Drive radar charts from `climate` and `water_profile`
    - Drive protein/biomass bands from `biomass_prediction`
    - Render the narrative report from `summary` / `formatted_text`
    """
    result = analyze_location(location)

    # If orchestration failed, surface the error clearly
    if "error" in result:
        return {"status": "error", "message": result["error"]}

    # Clean and format the summary text from the LLM
    summary = result["report"]

    # Split into lines for better readability on the frontend
    lines = summary.split("\n")
    formatted_lines = [line.strip() for line in lines if line.strip()]

    # Unpack structured components for charts / dashboard
    site_profile = result.get("site_profile", {}) or {}
    climate = site_profile.get("climate", {}) or {}
    water_profile = site_profile.get("water_profile", {}) or {}
    cultivation_status = site_profile.get("cultivation_status")
    cultivation_reasons = site_profile.get("reasons", [])

    biomass_prediction = result.get("biomass_prediction", {}) or {}
    coordinates = result.get("coordinates", {}) or {}

    return {
        "status": "success",
        "location": location,
        "analysis": {
            # Primary narrative fields
            "summary": summary,
            "formatted_text": formatted_lines,
            # Site & environmental structure for charts
            "coordinates": coordinates,
            "climate": climate,
            "water_profile": water_profile,
            "cultivation_status": cultivation_status,
            "cultivation_reasons": cultivation_reasons,
            # ML output that can be mapped to protein/biomass bands
            "biomass_prediction": biomass_prediction,
        },
    }