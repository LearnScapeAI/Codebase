# app/services/cache_service.py
import os
import json
import time
import logging
from datetime import datetime
from pinecone import Pinecone
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

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

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize S3
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION
    )
    logger.info("S3 client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {str(e)}")
    s3_client = None

def ensure_pinecone_index():
    """Create index if it doesn't exist and return the index object"""
    try:
        # List all indexes
        indexes = pc.list_indexes()
        index_names = [idx.name for idx in indexes]

        if PINECONE_INDEX_NAME not in index_names:
            # Create index with dimension 1024 to match existing index
            logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,
                metric="cosine"
            )
            logger.info(f"Pinecone index {PINECONE_INDEX_NAME} created successfully")
        else:
            logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

        # Return the index
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        logger.error(f"Error in ensure_pinecone_index: {str(e)}")
        raise

# Create S3 bucket if it doesn't exist
def ensure_s3_bucket():
    if not s3_client:
        logger.error("S3 client not initialized, cannot ensure bucket exists")
        return False
        
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        logger.info(f"S3 bucket {S3_BUCKET_NAME} exists")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        # 404 means bucket doesn't exist, other errors indicate different issues
        if error_code == '404':
            try:
                # Bucket doesn't exist, create it
                logger.info(f"Creating S3 bucket: {S3_BUCKET_NAME}")
                if S3_REGION != 'us-east-1':
                    s3_client.create_bucket(
                        Bucket=S3_BUCKET_NAME,
                        CreateBucketConfiguration={'LocationConstraint': S3_REGION}
                    )
                else:
                    s3_client.create_bucket(Bucket=S3_BUCKET_NAME)

                # Set lifecycle policy for auto-deletion after 6 months
                lifecycle_config = {
                    'Rules': [{
                        'ID': 'auto-delete',
                        'Status': 'Enabled',
                        'Expiration': {'Days': 180}
                    }]
                }
                s3_client.put_bucket_lifecycle_configuration(
                    Bucket=S3_BUCKET_NAME,
                    LifecycleConfiguration=lifecycle_config
                )
                logger.info(f"S3 bucket {S3_BUCKET_NAME} created successfully with lifecycle policy")
                return True
            except Exception as create_error:
                logger.error(f"Failed to create S3 bucket: {str(create_error)}")
                return False
        else:
            logger.error(f"Error accessing S3 bucket: {str(e)}")
            return False

def get_embedding(text):
    """Get vector embedding for a query"""
    # TODO: Replace with a real embedding service
    # This is a placeholder - in a real implementation, 
    # you would use OpenAI or another embedding service
    logger.info(f"Generating embedding for text: {text[:50]}...")
    import hashlib
    # Create a deterministic vector based on text hash for demo purposes
    hash_value = hashlib.md5(text.encode()).hexdigest()
    # Convert hash to a list of float values between -1 and 1
    vector = []
    for i in range(0, len(hash_value), 2):
        if i < len(hash_value) - 1:
            value = float(int(hash_value[i:i+2], 16)) / 255.0 * 2 - 1
            vector.append(value)

    # Pad to 1024 dimensions with float zeros
    vector = vector + [0.0] * (1024 - len(vector))

    # Ensure we have exactly 1024 dimensions
    return vector[:1024]

class RoadmapCache:
    def __init__(self):
        try:
            logger.info("Initializing RoadmapCache")
            self.index = ensure_pinecone_index()
            self.s3_available = ensure_s3_bucket()
            logger.info(f"RoadmapCache initialized. S3 available: {self.s3_available}")
        except Exception as e:
            logger.error(f"Failed to initialize RoadmapCache: {str(e)}")
            raise

    def create_id(self, learning_goals, months, days_per_week, hours_per_day):
        """Create a unique ID for the roadmap parameters"""
        # Fix: corrected variable name from learninggoals to learning_goals
        normalized_goals = learning_goals.lower().replace(' ', '')
        roadmap_id = f"{normalized_goals}-{months}-{days_per_week}-{hours_per_day}"
        logger.info(f"Created cache ID: {roadmap_id}")
        return roadmap_id

    def get_cached_roadmap(self, learning_goals, months, days_per_week, hours_per_day):
        """Try to get roadmap from cache (Pinecone first, then S3 if needed)"""
        logger.info(f"Checking cache for roadmap: {learning_goals}, {months} months, {days_per_week} days/week")
        roadmap_id = self.create_id(learning_goals, months, days_per_week, hours_per_day)
        query_vector = get_embedding(f"{learning_goals} {months} months {days_per_week} days per week {hours_per_day} hours per day")

        # Try Pinecone first
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )
            
            logger.info(f"Pinecone query returned {len(results['matches'])} matches")

            if results['matches'] and len(results['matches']) > 0:
                # Update access statistics
                match = results['matches'][0]
                score = match.get('score', 0)
                logger.info(f"Found match with score: {score}")
                
                # Only use cache if similarity is high enough
                if score < 0.8:  # Threshold for considering a match valid
                    logger.info(f"Match score too low ({score}), regenerating roadmap")
                    return None
                
                new_count = match['metadata'].get('access_count', 0) + 1

                # Updated code for Pinecone update method
                self.index.update(
                    id=match['id'],
                    set_metadata={
                        'access_count': new_count,
                        'last_accessed': datetime.now().isoformat()
                    }
                )

                roadmap_json = match['metadata'].get('roadmap_json', '{}')
                try:
                    roadmap_data = json.loads(roadmap_json)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in cached roadmap: {roadmap_json[:100]}...")
                    return None

                # Validate roadmap structure
                if not self.validate_roadmap(roadmap_data, months, days_per_week):
                    logger.warning("Cached roadmap validation failed, regenerating")
                    return None

                logger.info("Successfully retrieved and validated roadmap from Pinecone cache")
                return roadmap_data

        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return None

        # If not in Pinecone, try S3
        if not self.s3_available or not s3_client:
            logger.warning("S3 not available, skipping S3 cache check")
            return None
            
        try:
            logger.info(f"Checking S3 for roadmap: {roadmap_id}")
            s3_object = s3_client.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{roadmap_id}.json"
            )

            data = json.loads(s3_object['Body'].read())
            roadmap_json = data.get('roadmap_json')

            # Check for all expected weeks before rehydrating
            try:
                roadmap_data = json.loads(roadmap_json)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in S3 roadmap: {roadmap_json[:100]}...")
                return None

            # Validate roadmap structure
            if not self.validate_roadmap(roadmap_data, months, days_per_week):
                logger.warning("S3 roadmap validation failed, regenerating")
                return None

            # Rehydrate to Pinecone
            logger.info(f"Rehydrating roadmap from S3 to Pinecone: {roadmap_id}")
            self.cache_roadmap(
                learning_goals, 
                months, 
                days_per_week,
                hours_per_day,  # Fixed: Add hours_per_day parameter
                roadmap_data,
                rehydrated=True
            )

            logger.info("Successfully retrieved and rehydrated roadmap from S3 cache")
            return roadmap_data

        except ClientError as e:
            logger.error(f"Error retrieving from S3: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error with S3: {str(e)}")
            return None

    def validate_roadmap(self, roadmap_data, months, days_per_week):
        """Validate that roadmap has all expected weeks and days"""
        expected_weeks = months * 4
        
        # Check all weeks exist
        for week_num in range(1, expected_weeks + 1):
            week_key = f"week{week_num}"
            if week_key not in roadmap_data:
                logger.warning(f"Missing {week_key} in roadmap")
                return False
                
            # Check all days exist in this week
            week_data = roadmap_data[week_key]
            for day_num in range(1, days_per_week + 1):
                day_key = f"day{day_num}"
                if day_key not in week_data:
                    logger.warning(f"Missing {day_key} in {week_key}")
                    return False
                    
                # Check day has content
                if not week_data[day_key] or not isinstance(week_data[day_key], list):
                    logger.warning(f"Empty or invalid content for {day_key} in {week_key}")
                    return False
                    
                # Check each topic has required fields
                for topic in week_data[day_key]:
                    if not isinstance(topic, dict) or 'topic' not in topic or 'resource' not in topic:
                        logger.warning(f"Invalid topic structure in {day_key}, {week_key}")
                        return False
        
        return True

    def cache_roadmap(self, learning_goals, months, days_per_week, hours_per_day, roadmap_json, rehydrated=False):
        """Store a roadmap in Pinecone"""
        logger.info(f"Caching roadmap for: {learning_goals}, {months} months, {days_per_week} days/week")
        roadmap_id = self.create_id(learning_goals, months, days_per_week, hours_per_day)
        vector = get_embedding(f"{learning_goals} {months} months {days_per_week} days per week {hours_per_day} hours per day")

        # Convert roadmap_json to string if it's a dict
        if isinstance(roadmap_json, dict):
            roadmap_json_str = json.dumps(roadmap_json)
        else:
            roadmap_json_str = roadmap_json

        # Set initial metadata
        metadata = {
            'learning_goals': learning_goals,
            'months': months,
            'days_per_week': days_per_week,
            'hours_per_day': hours_per_day,
            'roadmap_json': roadmap_json_str,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1 if rehydrated else 0
        }

        try:
            # Upsert to Pinecone - updated syntax
            self.index.upsert(
                vectors=[(roadmap_id, vector, metadata)]
            )
            logger.info(f"Roadmap cached successfully with ID: {roadmap_id}")
        except Exception as e:
            logger.error(f"Error caching roadmap in Pinecone: {str(e)}")

        return roadmap_id

    def archive_cold_data(self, access_threshold=5, days_threshold=30):
        """Move cold data from Pinecone to S3"""
        if not self.s3_available or not s3_client:
            logger.error("S3 not available, cannot archive data")
            return 0
            
        logger.info(f"Archiving cold data: threshold={access_threshold} accesses, {days_threshold} days")
        # Calculate the date threshold
        from datetime import datetime, timedelta
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()

        try:
            # Query for cold data
            # Fetch all vectors - in a production environment with many vectors,
            # you might need to implement pagination
            fetch_response = self.index.fetch(ids=[])  # Fetch all vectors
            logger.info(f"Fetched {len(fetch_response['vectors'].keys())} vectors from Pinecone")

            archived_count = 0
            for vector_id, vector_data in fetch_response['vectors'].items():
                metadata = vector_data.get('metadata', {})
                access_count = metadata.get('access_count', 0)
                last_accessed = metadata.get('last_accessed', '')

                # Check if it's cold data
                if access_count < access_threshold and last_accessed < threshold_date:
                    logger.info(f"Found cold data: ID={vector_id}, access_count={access_count}, last_accessed={last_accessed}")
                    try:
                        # Convert vector values to float if they're not already
                        if 'values' in vector_data:
                            vector_values = [float(value) for value in vector_data['values']]
                        else:
                            vector_values = []
                            logger.warning(f"No vector values found for ID={vector_id}")

                        # Move to S3
                        s3_object = {
                            'vector': vector_values,
                            'metadata': metadata,
                            'roadmap_json': metadata.get('roadmap_json', '{}')
                        }
                        
                        s3_client.put_object(
                            Bucket=S3_BUCKET_NAME,
                            Key=f"{vector_id}.json",
                            Body=json.dumps(s3_object)
                        )
                        logger.info(f"Archived {vector_id} to S3 successfully")

                        # Delete from Pinecone
                        self.index.delete(ids=[vector_id])
                        logger.info(f"Deleted {vector_id} from Pinecone")
                        
                        archived_count += 1
                    except Exception as archive_error:
                        logger.error(f"Error archiving {vector_id}: {str(archive_error)}")
            
            logger.info(f"Archived {archived_count} items to S3")
            return archived_count
            
        except Exception as e:
            logger.error(f"Error in archive_cold_data: {str(e)}")
            return 0