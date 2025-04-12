# app/routes/roadmap.py
from fastapi import APIRouter, HTTPException
from app.models import RoadmapRequest
from app.services.roadmap_service import generate_roadmap
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("learnscape.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Roadmap Generator"])

@router.post("/generate_roadmap")
async def create_roadmap(request: RoadmapRequest):
    """
    Generate a personalized learning roadmap based on learning goals and time commitment.
    
    Each day's content will be broken down into hour-by-hour segments to help the user
    manage their study time efficiently.
    """
    logger.info(f"Received roadmap request: {request.learning_goals}, {request.months} months, "
               f"{request.days_per_week} days/week, {request.hours_per_day} hours/day")
    
    try:
        roadmap_json = await generate_roadmap(
            request.learning_goals,
            request.months,
            request.days_per_week,
            request.hours_per_day
        )

        # Parse the roadmap to ensure it's valid JSON
        if isinstance(roadmap_json, str):
            try:
                parsed_roadmap = json.loads(roadmap_json)
                logger.info("Successfully parsed roadmap JSON")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON returned from generate_roadmap: {str(e)}")
                raise ValueError(f"Invalid roadmap JSON: {str(e)}")
        else:
            parsed_roadmap = roadmap_json

        # Verify that all weeks and days are present
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
                
                # Verify that each topic has hours assigned
                for topic in week_data[day_key]:
                    if 'hours' not in topic:
                        logger.warning(f"Topic missing hours in {day_key}, {week_key}")
                        # Add a default hour value rather than failing
                        topic['hours'] = 1.0
                    
                # Verify that total hours match expected
                total_hours = sum(float(topic.get('hours', 0)) for topic in week_data[day_key])
                if abs(total_hours - request.hours_per_day) > 0.1:
                    logger.warning(f"Hours don't add up in {day_key}, {week_key}: {total_hours} vs {request.hours_per_day}")
                    # This is just a warning, we won't fail the request

        logger.info("Roadmap validation complete, returning response")
        # Return the validated roadmap
        return {
            "roadmap": parsed_roadmap,
            "message": "Your personalized learning roadmap has been generated successfully. Each day includes an hour-by-hour breakdown to help you manage your study time effectively."
        }

    except Exception as e:
        # Log the error
        logger.error(f"Error generating roadmap: {str(e)}")

        # Return a more graceful error
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating roadmap: {str(e)}"
        )