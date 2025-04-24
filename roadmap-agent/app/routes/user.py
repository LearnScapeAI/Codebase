from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, Roadmap, Progress, RoadmapResponse, ProgressUpdate
from app.services.auth_service import get_current_user
from app.services.roadmap_service import get_user_roadmaps, get_roadmap_with_progress, update_progress
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["User Dashboard"])

@router.get("/roadmaps")
async def list_user_roadmaps(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all roadmaps for the current user"""
    roadmaps = await get_user_roadmaps(db, current_user.id)
    return {
        "roadmaps": [
            {
                "id": roadmap.id,
                "learning_goals": roadmap.learning_goals,
                "created_at": roadmap.created_at.isoformat()
            } for roadmap in roadmaps
        ]
    }

@router.get("/roadmaps/{roadmap_id}")
async def get_roadmap_details(
    roadmap_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific roadmap with progress information"""
    roadmap_data = await get_roadmap_with_progress(db, roadmap_id, current_user.id)
    
    if not roadmap_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Roadmap not found or access denied"
        )
    
    return roadmap_data

@router.post("/roadmaps/{roadmap_id}/progress")
async def mark_topic_progress(
    roadmap_id: str,
    progress: ProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update progress for a specific topic in a roadmap"""
    result = await update_progress(
        db, 
        roadmap_id, 
        current_user.id,
        progress.week_number,
        progress.day_number,
        progress.topic_index,
        progress.completed
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Roadmap or topic not found"
        )
    
    return {"message": "Progress updated successfully"}