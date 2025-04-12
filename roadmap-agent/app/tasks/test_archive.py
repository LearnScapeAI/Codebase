# app/tasks/test_archive.py
import asyncio
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.cache_service import RoadmapCache
from app.services.roadmap_service import generate_roadmap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_archive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def create_test_roadmaps():
    """Create several roadmaps to be used for testing archiving"""
    topics = [
        "Python Programming",
        "Data Science",
        "Machine Learning",
        "Web Development",
        "Mobile App Development"
    ]
    
    cache = RoadmapCache()
    logger.info("Creating test roadmaps for archiving...")
    
    for topic in topics:
        logger.info(f"Generating roadmap for {topic}")
        try:
            roadmap_json = await generate_roadmap(
                learning_goals=topic,
                months=1,
                days_per_week=3,
                hours_per_day=2.0
            )
            logger.info(f"Successfully generated roadmap for {topic}")
        except Exception as e:
            logger.error(f"Error generating roadmap for {topic}: {str(e)}")

async def test_archive():
    """Test the archiving functionality"""
    logger.info("Testing archive functionality")
    cache = RoadmapCache()
    
    try:
        # Set threshold to 1 and days to 0 to ensure all data is considered "cold"
        archived_count = cache.archive_cold_data(access_threshold=1, days_threshold=0)
        logger.info(f"Archived {archived_count} items to S3")
        
        if archived_count == 0:
            logger.warning("No items were archived. This could be because:")
            logger.warning("1. No roadmaps were found in Pinecone")
            logger.warning("2. All roadmaps have been accessed more than the threshold")
            logger.warning("3. There was an error connecting to S3")
            
            # Test S3 connection explicitly
            if cache.s3_available:
                logger.info("S3 connection appears to be working")
            else:
                logger.error("S3 connection is not available")
        
        return archived_count
    except Exception as e:
        logger.error(f"Error testing archive: {str(e)}")
        return 0

async def main():
    # First create some test roadmaps
    await create_test_roadmaps()
    
    # Wait a bit to ensure roadmaps are stored
    await asyncio.sleep(2)
    
    # Then test archiving
    await test_archive()

if __name__ == "__main__":
    asyncio.run(main())