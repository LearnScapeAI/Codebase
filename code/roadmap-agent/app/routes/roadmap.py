from fastapi import APIRouter
from app.models import RoadmapRequest
from app.services.roadmap_service import generate_roadmap

router = APIRouter(tags=["Roadmap Generator"])

@router.post("/generate_roadmap")
async def create_roadmap(request: RoadmapRequest):
    roadmap = await generate_roadmap(
        request.learning_goals,
        request.months,
        request.days_per_week
    )
    return {"roadmap": roadmap}