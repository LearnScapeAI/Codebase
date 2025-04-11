import os
import json
import time
from datetime import datetime
from pinecone import Pinecone
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

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
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION
)

def ensure_pinecone_index():
    """Create index if it doesn't exist and return the index object"""
    # List all indexes
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    if PINECONE_INDEX_NAME not in index_names:
        # Create index with dimension 1024 to match existing index
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,  # Changed from 1536 to 1024
            metric="cosine"
        )
    
    # Return the index
    return pc.Index(PINECONE_INDEX_NAME)

# Create S3 bucket if it doesn't exist
def ensure_s3_bucket():
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError:
        # Bucket doesn't exist, create it
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

def get_embedding(text):
    """Get vector embedding for a query"""
    # This is a placeholder - in a real implementation, 
    # you would use OpenAI or another embedding service
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
        self.index = ensure_pinecone_index()
        ensure_s3_bucket()
        
    def create_id(self, learning_goals, months, days_per_week):
        """Create a unique ID for the roadmap parameters"""
        return f"{learning_goals.lower().replace(' ', '_')}_{months}_{days_per_week}"
    
    def get_cached_roadmap(self, learning_goals, months, days_per_week):
        """Try to get roadmap from cache (Pinecone first, then S3 if needed)"""
        roadmap_id = self.create_id(learning_goals, months, days_per_week)
        query_vector = get_embedding(f"{learning_goals} {months} months {days_per_week} days per week")
        
        # Try Pinecone first
        results = self.index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True
        )
        
        if results['matches'] and len(results['matches']) > 0:
            # Update access statistics
            match = results['matches'][0]
            new_count = match['metadata'].get('access_count', 0) + 1
            
            # Updated code for Pinecone update method
            self.index.update(
                id=match['id'],
                set_metadata={
                    'access_count': new_count,
                    'last_accessed': datetime.now().isoformat()
                }
            )
            
            roadmap_data = json.loads(match['metadata']['roadmap_json'])

            # Check if the roadmap has all the expected weeks
            expected_weeks = months * 4
            for week_num in range(1, expected_weeks + 1):
                week_key = f"week{week_num}"
                if week_key not in roadmap_data:
                    # If any week is missing, return None to force regeneration
                    print(f"Cached roadmap missing {week_key}, regenerating...")
                    return None

            return roadmap_data

        # If not in Pinecone, try S3
        try:
            s3_object = s3_client.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{roadmap_id}.json"
            )
            
            data = json.loads(s3_object['Body'].read())
            roadmap_json = data.get('roadmap_json')

            # Check for all expected weeks before rehydrating
            roadmap_data = json.loads(roadmap_json)
            expected_weeks = months * 4
            for week_num in range(1, expected_weeks + 1):
                week_key = f"week{week_num}"
                if week_key not in roadmap_data:
                    print(f"S3 roadmap missing {week_key}, regenerating...")
                    return None

            # Rehydrate to Pinecone
            self.cache_roadmap(
                learning_goals, 
                months, 
                days_per_week, 
                roadmap_json,
                rehydrated=True
            )

            return roadmap_data

        except ClientError:
            # Not found in S3 either
            return None

    
    def cache_roadmap(self, learning_goals, months, days_per_week, roadmap_json, rehydrated=False):
        """Store a roadmap in Pinecone"""
        roadmap_id = self.create_id(learning_goals, months, days_per_week)
        vector = get_embedding(f"{learning_goals} {months} months {days_per_week} days per week")
        
        # Set initial metadata
        metadata = {
            'learning_goals': learning_goals,
            'months': months,
            'days_per_week': days_per_week,
            'roadmap_json': json.dumps(roadmap_json) if isinstance(roadmap_json, dict) else roadmap_json,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1 if rehydrated else 0
        }
        
        # Upsert to Pinecone - updated syntax
        self.index.upsert(
            vectors=[(roadmap_id, vector, metadata)]
        )
        
        return roadmap_id
    
    def archive_cold_data(self, access_threshold=5, days_threshold=30):
        """Move cold data from Pinecone to S3"""
        # Calculate the date threshold
        from datetime import datetime, timedelta
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        # Query for cold data
        # Note: This is a simplified approach since Pinecone doesn't directly support
        # the complex filtering shown in the requirements. In practice, you'd
        # need to fetch all vectors and filter in application code
        fetch_response = self.index.fetch(ids=[])  # Fetch all vectors
        
        archived_count = 0
        for vector_id, vector_data in fetch_response['vectors'].items():
            metadata = vector_data.get('metadata', {})
            access_count = metadata.get('access_count', 0)
            last_accessed = metadata.get('last_accessed', '')
            
            # Check if it's cold data
            if access_count < access_threshold and last_accessed < threshold_date:
                # Move to S3
                # Check if it's cold data
                # Convert vector values to float
                vector_data['values'] = [float(value) for value in vector_data['values']]

                # Move to S3
                s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=f"{vector_id}.json",
                Body=json.dumps({
                    'vector': vector_data['values'],
                    'metadata': metadata,
                    'roadmap_json': metadata.get('roadmap_json', '{}')
                    })
                )
                
                # Delete from Pinecone
                self.index.delete(ids=[vector_id])
                archived_count += 1
        
        return archived_count