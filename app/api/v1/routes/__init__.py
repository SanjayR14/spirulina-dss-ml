from fastapi import APIRouter
from .site_profile import router as site_profile_router

router = APIRouter()

# Include your site_profile router
router.include_router(site_profile_router, prefix="/site", tags=["Site Analysis"])

@router.get("/")
async def api_root():
    """API v1 root endpoint"""
    return {"message": "API v1", "status": "active"}
