from fastapi import APIRouter, HTTPException
from app.models import RoadmapRequest
from app.services.roadmap_service import generate_roadmap
import json

router = APIRouter(tags=["Roadmap Generator"])

@router.post("/generate_roadmap")
async def create_roadmap(request: RoadmapRequest):
    roadmap_json = await generate_roadmap(
        request.learning_goals,
        request.months,
        request.days_per_week
    )
    
    # Parse the roadmap to ensure it's valid JSON before returning
    try:
        # First ensure it's a JSON string (not a Python dict)
        if isinstance(roadmap_json, dict):
            roadmap_json = json.dumps(roadmap_json)
            
        # Parse the string to validate it's proper JSON
        parsed_roadmap = json.loads(roadmap_json)
        
        # Return the parsed dictionary directly (FastAPI will convert to JSON)
        return {"roadmap": parsed_roadmap}
    except json.JSONDecodeError:
        # If JSON is invalid, return the string version as a fallback
        return {"roadmap": roadmap_json}