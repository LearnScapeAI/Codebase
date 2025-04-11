from app.services.cache_service import RoadmapCache

# Create several cached items
cache = RoadmapCache()

# This will trigger the archiving process manually
archived_count = cache.archive_cold_data(access_threshold=1, days_threshold=0)
print(f"Archived {archived_count} items to S3")