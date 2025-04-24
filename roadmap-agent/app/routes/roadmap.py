from fastapi import APIRouter, HTTPException, Depends
from app.models import RoadmapRequest, User
from app.services.roadmap_service import generate_roadmap, generate_roadmap_streaming, save_roadmap
from app.services.auth_service import get_current_user
from app.database import get_db
from sqlalchemy.orm import Session
import json
import logging
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("learnscape.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Roadmap Generator"])

# Generate a personalized learning roadmap
@router.post("/generate_roadmap")
async def create_roadmap(
    request: RoadmapRequest, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    logger.info(f"Received roadmap request: {request.learning_goals}, {request.months} months, "
               f"{request.days_per_week} days/week, {request.hours_per_day} hours/day")
    
    try:
        roadmap_json = await generate_roadmap(
            request.learning_goals,
            request.months,
            request.days_per_week,
            request.hours_per_day
        )

        if isinstance(roadmap_json, str):
            try:
                parsed_roadmap = json.loads(roadmap_json)
                logger.info("Successfully parsed roadmap JSON")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON returned from generate_roadmap: {str(e)}")
                raise ValueError(f"Invalid roadmap JSON: {str(e)}")
        else:
            parsed_roadmap = roadmap_json

        expected_weeks = request.months * 4
        expected_days = request.days_per_week

        for week_num in range(1, expected_weeks + 1):
            week_key = f"week{week_num}"
            if week_key not in parsed_roadmap:
                logger.error(f"Missing week {week_num} in roadmap")
                raise ValueError(f"Missing week {week_num} in roadmap")

            week_data = parsed_roadmap[week_key]
            for day_num in range(1, expected_days + 1):
                day_key = f"day{day_num}"
                if day_key not in week_data or not week_data[day_key]:
                    logger.error(f"Missing or empty day {day_num} in week {week_num}")
                    raise ValueError(f"Missing or empty day {day_num} in week {week_num}")
                
                for topic in week_data[day_key]:
                    if 'hours' not in topic:
                        logger.warning(f"Topic missing hours in {day_key}, {week_key}")
                        topic['hours'] = 1.0
                    
                total_hours = sum(float(topic.get('hours', 0)) for topic in week_data[day_key])
                if abs(total_hours - request.hours_per_day) > 0.1:
                    logger.warning(f"Hours don't add up in {day_key}, {week_key}: {total_hours} vs {request.hours_per_day}")

        # Save the roadmap to the database
        db_roadmap = await save_roadmap(
            db,
            current_user.id,
            request.learning_goals,
            request.months,
            request.days_per_week,
            request.hours_per_day,
            parsed_roadmap
        )

        logger.info(f"Roadmap saved with ID: {db_roadmap.id}")
        
        return {
            "roadmap_id": db_roadmap.id,
            "roadmap": parsed_roadmap,
            "message": "Your personalized learning roadmap has been generated and saved successfully. Each day includes an hour-by-hour breakdown to help you manage your study time effectively."
        }

    except Exception as e:
        logger.error(f"Error generating roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating roadmap: {str(e)}")

# Generate a personalized learning roadmap with streaming response
@router.post("/generate_roadmap_stream")
async def create_roadmap_stream(
    request: RoadmapRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate a personalized learning roadmap with a streaming response.
    Instead of returning chunks, this collects and returns a complete JSON response.
    """
    logger.info(f"Received streaming roadmap request: {request.learning_goals}, {request.months} months")

    try:
        roadmap_dict = {}
        async for chunk in generate_roadmap_streaming(
            request.learning_goals,
            request.months,
            request.days_per_week,
            request.hours_per_day
        ):
            # Expect each chunk to be a (week_key, week_data) pair
            # e.g., ("week1", {"day1": [...], "day2": [...]})
            if isinstance(chunk, str):
                try:
                    chunk_data = json.loads(chunk)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid chunk format: {str(e)}")
                    continue
            else:
                chunk_data = chunk

            if isinstance(chunk_data, dict):
                roadmap_dict.update(chunk_data)
            else:
                logger.warning("Chunk not in expected format (dict)")

        # ✅ Save the roadmap using authenticated user
        db_roadmap = await save_roadmap(
            db,
            current_user.id,  # <-- user ID from token
            request.learning_goals,
            request.months,
            request.days_per_week,
            request.hours_per_day,
            roadmap_dict
        )

        logger.info(f"Streaming roadmap saved with ID: {db_roadmap.id}")
        
        # ✅ Include roadmap_id in response
        return JSONResponse(
            content={
                "roadmap_id": db_roadmap.id,
                "roadmap": roadmap_dict,
                "message": "Your personalized learning roadmap has been generated and saved using the streaming process."
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Error generating streaming roadmap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating streaming roadmap: {str(e)}")