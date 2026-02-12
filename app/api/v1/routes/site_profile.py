from fastapi import APIRouter, Query
from app.api.v1.routes.checker import analyze_location

router = APIRouter()

@router.get("/analyze")
def analyze_site(location: str = Query(..., description="Place name to analyze for Spirulina cultivation")):
    """
    Analyze a location for Spirulina cultivation suitability.
    Returns an LLM-generated summary report.
    """
    result = analyze_location(location)
    
    # Format the response better
    if "error" in result:
        return {"status": "error", "message": result["error"]}
    
    # Clean and format the summary
    summary = result["report"]
    
    # Split into lines for better readability
    lines = summary.split('\n')
    formatted_lines = [line.strip() for line in lines if line.strip()]
    
    return {
        "status": "success",
        "location": location,
        "analysis": {
            "summary": summary,
            "formatted_text": formatted_lines
        }
    }