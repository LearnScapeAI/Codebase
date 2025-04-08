from fastapi import APIRouter, HTTPException
from app.models import RoadmapRequest
from app.services.roadmap_service import generate_roadmap
import json

router = APIRouter(tags=["Roadmap Generator"])

@router.post("/generate_roadmap")
async def create_roadmap(request: RoadmapRequest):
    try:
        roadmap_json = await generate_roadmap(
            request.learning_goals,
            request.months,
            request.days_per_week
        )
        
        # Parse the roadmap to ensure it's valid JSON
        if isinstance(roadmap_json, str):
            parsed_roadmap = json.loads(roadmap_json)
        else:
            parsed_roadmap = roadmap_json
            
        # Verify that all weeks and days are present
        expected_weeks = request.months * 4
        expected_days = request.days_per_week
        
        for week_num in range(1, expected_weeks + 1):
            week_key = f"week{week_num}"
            if week_key not in parsed_roadmap:
                raise ValueError(f"Missing week {week_num} in roadmap")
                
            week_data = parsed_roadmap[week_key]
            for day_num in range(1, expected_days + 1):
                day_key = f"day{day_num}"
                if day_key not in week_data or not week_data[day_key]:
                    raise ValueError(f"Missing or empty day {day_num} in week {week_num}")
        
        # Return the validated roadmap
        return {"roadmap": parsed_roadmap}
    
    except Exception as e:
        # Log the error
        print(f"Error generating roadmap: {str(e)}")
        
        # Return a more graceful error
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating roadmap: {str(e)}"
        )