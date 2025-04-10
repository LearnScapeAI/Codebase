import schedule
import time
import threading
from app.services.cache_service import RoadmapCache

def archive_cold_data_job():
    """Archive cold data from Pinecone to S3"""
    print("Running scheduled archiving job...")
    
    cache = RoadmapCache()
    archived_count = cache.archive_cold_data(access_threshold=5, days_threshold=30)
    
    print(f"Archived {archived_count} items to S3")

def start_scheduler():
    """Start the scheduler in a separate thread"""
    schedule.every().day.at("03:00").do(archive_cold_data_job)  # Run at 3 AM daily
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()