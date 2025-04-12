# app/tasks/archive_task.py
import schedule
import time
import threading
import logging
from app.services.roadmap_service import archive_cold_data

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

async def archive_cold_data_job():
    """Archive cold data from Pinecone to S3"""
    logger.info("Running scheduled archiving job...")
    try:
        archived_count = await archive_cold_data(access_threshold=5, days_threshold=30)
        logger.info(f"Archived {archived_count} items to S3")
    except Exception as e:
        logger.error(f"Error in scheduled archive job: {str(e)}")

def start_scheduler():
    """Start the scheduler in a separate thread"""
    logger.info("Starting scheduler for archiving cold data")
    schedule.every().day.at("03:00").do(run_archive_job)  # Run at 3 AM daily
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("Scheduler started successfully")

def run_archive_job():
    """Run the archive job asynchronously"""
    import asyncio
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the archive job
        loop.run_until_complete(archive_cold_data_job())
    except Exception as e:
        logger.error(f"Error in run_archive_job: {str(e)}")
    finally:
        loop.close()

def run_scheduler():
    """Run pending scheduled tasks"""
    logger.info("Scheduler thread started")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in scheduler: {str(e)}")
            time.sleep(60)  # Continue despite errors