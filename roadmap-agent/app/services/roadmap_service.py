# app/services/roadmap_service.py
from app.utils.prompt_utils import filter_prompt, verify_prompt, format_json_prompt, generate_initial_prompt
from app.utils.resource_fetcher import fetch_real_resources
from app.services.cache_service import RoadmapCache, get_cached_roadmap, cache_roadmap
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from app.models import Roadmap, Progress, User
from sqlalchemy.orm import Session
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import os
import json
import re
import logging
import asyncio
from datetime import datetime
import traceback

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

# Initialize the LLM
try:
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )
    
    # Initialize a separate LLM for batch processing
    batch_llm = ChatOpenAI(
        model_name="gpt-4o",  # Use larger context model for batch processing
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True
    )
    logger.info("LLMs initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Initialize the cache
try:
    roadmap_cache = RoadmapCache()
    logger.info("Roadmap cache initialized")
except Exception as e:
    logger.error(f"Failed to initialize roadmap cache: {str(e)}")
    raise

# Constants for batching
MAX_WEEKS_PER_BATCH = 4  # Number of weeks to process in a single batch
CONCURRENT_BATCHES = 3   # Number of batches to process concurrently

# Add this callback handler class at an appropriate place in the file
class StreamingJSONCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming JSON chunks."""
    
    def __init__(self):
        self.tokens = []
        self.json_buffer = ""
        self.streaming_queue = asyncio.Queue()
        
    async def on_llm_new_token(self, token: str, **kwargs):
        """Process new tokens as they're generated."""
        self.tokens.append(token)
        self.json_buffer += token
        
        # Try to find valid JSON objects or arrays
        if self._is_valid_json_chunk(self.json_buffer):
            # Put the new chunk in the queue
            await self.streaming_queue.put(token)
    
    def _is_valid_json_chunk(self, text: str) -> bool:
        """Check if we have received a valid chunk of JSON."""
        # For simplicity, we'll just check if we have a complete week or day object
        # This is a basic implementation - you might want to improve it
        if ('{"week' in text or '"day' in text) and ('"topic":' in text):
            return True
        return False
    
    async def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when LLM generation ends."""
        # Signal that we're done
        await self.streaming_queue.put(None)
    
    async def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when LLM encounters an error."""
        await self.streaming_queue.put({"error": str(error)})
        await self.streaming_queue.put(None)  # Signal completion

# Add this new streaming function to roadmap_service.py
async def generate_roadmap_streaming(learning_goals: str, months: int, days_per_week: int, hours_per_day: float):
    """Generate a roadmap with streaming responses to show progress in real-time."""
    logger.info(f"Generating streaming roadmap for: {learning_goals}, {months} months")
    
    # Check cache first (Pinecone, then S3 if needed)
    cached_roadmap = roadmap_cache.get_cached_roadmap(learning_goals, months, days_per_week, hours_per_day)
    if cached_roadmap:
        logger.info(f"Cache hit for: {learning_goals}")
        # For cached responses, we'll still stream but all at once
        yield json.dumps({"status": "cached", "message": "Using cached roadmap"}) + "\n"
        yield json.dumps(cached_roadmap) + "\n"
        return

    logger.info(f"Cache miss for: {learning_goals}, generating with streaming LLM...")
    
    # Calculate total weeks for better prompting
    total_weeks = months * 4
    
    # Create the streaming callback handler
    streaming_handler = StreamingJSONCallbackHandler()
    
    # Initialize the roadmap structure and metadata
    yield json.dumps({
        "status": "initializing",
        "message": "Starting roadmap generation",
        "total_weeks": total_weeks,
        "days_per_week": days_per_week
    }) + "\n"
    
    """
    # 1. First LLM call — get outline of all topics
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week, hours_per_day)
    yield json.dumps({
        "status": "planning",
        "message": "Creating high-level roadmap plan"
    }) + "\n"
    
    try:
        day_topic_plan = await llm.ainvoke(
            [HumanMessage(content=prompt1)],
            callbacks=[streaming_handler]
        )
        logger.info("Received initial topic plan from LLM")
        
        yield json.dumps({
            "status": "plan_complete",
            "message": "High-level plan created, now generating detailed content"
        }) + "\n"
    except Exception as e:
        logger.error(f"Error from LLM when generating initial plan: {str(e)}")
        yield json.dumps({
            "status": "error",
            "message": f"Failed to generate initial roadmap plan: {str(e)}"
        }) + "\n"
        return
    """
    
    # In roadmap_service.py, find and modify this part of the generate_roadmap function:

    # 1. First LLM call — get complete outline of all topics
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week, hours_per_day)
    yield json.dumps({
        "status": "planning",
        "message": "Creating high-level roadmap plan"
    }) + "\n"
    logger.info("Calling LLM for initial topic plan")
    # In generate_roadmap_streaming function, make sure the LLM call is handling streaming properly:

    try:
        # Create a separate streaming LLM instance to be extra safe
        streaming_llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True
        )
        
        # Create the streaming callback handler
        streaming_handler = StreamingJSONCallbackHandler()
        
        day_topic_plan = await streaming_llm.ainvoke(
            [HumanMessage(content=prompt1)]
        )

        logger.info("Received initial topic plan from LLM")
        
        yield json.dumps({
            "status": "plan_complete",
            "message": "High-level plan created, now generating detailed content"
        }) + "\n"
    except Exception as e:
        logger.error(f"Error from LLM when generating initial plan: {str(e)}")
        yield json.dumps({
            "status": "error",
            "message": f"Failed to generate initial roadmap plan: {str(e)}"
        }) + "\n"
        return
    
    # 2. Process the roadmap with real-time updates
    complete_roadmap = {}
    
    # Process weeks in smaller batches for better streaming
    batch_size = min(MAX_WEEKS_PER_BATCH, total_weeks)  # If <4 weeks total, do fewer
    
    # Create batch processing tasks
    batches = []
    for start_week in range(1, total_weeks + 1, batch_size):
        end_week = min(start_week + batch_size - 1, total_weeks)
        batches.append((start_week, end_week))
    
    # Stream progress as each batch completes
    current_batch = 0
    total_batches = len(batches)
    
    # Process batches in sequence for more predictable streaming
    for start_week, end_week in batches:
        current_batch += 1
        
        yield json.dumps({
            "status": "processing_batch",
            "message": f"Processing weeks {start_week}-{end_week}",
            "progress": {
                "current_batch": current_batch,
                "total_batches": total_batches,
                "start_week": start_week,
                "end_week": end_week
            }
        }) + "\n"
        
        try:
            # Process this batch
            batch_result = await process_week_batch_streaming(
                start_week, end_week, days_per_week, hours_per_day,
                learning_goals, day_topic_plan.content, streaming_handler
            )
            
            # Update the complete roadmap
            complete_roadmap.update(batch_result)
            
            # Stream the batch result
            yield json.dumps({
                "status": "batch_complete",
                "batch_data": batch_result,
                "progress": {
                    "completed_weeks": end_week,
                    "total_weeks": total_weeks,
                    "percent_complete": round((end_week / total_weeks) * 100)
                }
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Error processing batch for weeks {start_week}-{end_week}: {str(e)}")
            yield json.dumps({
                "status": "batch_error",
                "message": f"Error processing weeks {start_week}-{end_week}: {str(e)}",
                "will_retry": True
            }) + "\n"
            
            # Try to regenerate with a simplified approach
            try:
                simplified_result = {}
                for week_num in range(start_week, end_week + 1):
                    week_result = await process_single_week(
                        week_num, days_per_week, hours_per_day, learning_goals, day_topic_plan.content
                    )
                    simplified_result[f"week{week_num}"] = week_result
                    
                    # Stream each week as it completes
                    yield json.dumps({
                        "status": "week_recovery_complete",
                        #"week_data": {f"week{week_num}": week_result},
                        "progress": {
                            "completed_weeks": week_num,
                            "total_weeks": total_weeks,
                            "percent_complete": round((week_num / total_weeks) * 100)
                        }
                    }) + "\n"
                
                # Update the complete roadmap
                complete_roadmap.update(simplified_result)
                
            except Exception as recovery_error:
                logger.error(f"Recovery attempt failed: {str(recovery_error)}")
                yield json.dumps({
                    "status": "recovery_failed",
                    "message": f"Failed to recover batch {current_batch}: {str(recovery_error)}"
                }) + "\n"
    
    # Cache the result in Pinecone
    logger.info("Caching generated roadmap")
    try:
        roadmap_cache.cache_roadmap(learning_goals, months, days_per_week, hours_per_day, complete_roadmap)
        yield json.dumps({
            "status": "caching_complete",
            "message": "Roadmap saved to cache for future use"
        }) + "\n"
    except Exception as cache_error:
        logger.error(f"Error caching roadmap: {str(cache_error)}")
        yield json.dumps({
            "status": "cache_error",
            "message": f"Could not cache roadmap: {str(cache_error)}"
        }) + "\n"
    
    # Final complete roadmap
    yield json.dumps({
        "status": "complete",
        "roadmap": complete_roadmap,
        "message": "Your personalized learning roadmap has been generated successfully."
    }) + "\n"

async def process_week_batch_streaming(start_week, end_week, days_per_week, hours_per_day, learning_goals, full_plan, streaming_handler):
    """Process a batch of weeks with streaming updates."""
    # This is a modified version of process_week_batch that supports streaming
    
    logger.info(f"Processing streaming batch for weeks {start_week}-{end_week}")
    batch_result = {}
    
    # Extract topics for all weeks in this batch
    batch_topics = ""
    for week_num in range(start_week, end_week + 1):
        week_topics = extract_week_from_plan(full_plan, week_num)
        batch_topics += f"\nWEEK {week_num} TOPICS:\n{week_topics}\n"
    
    # Generate a prompt for the entire batch
    batch_prompt = f"""
    You are creating a detailed learning roadmap for weeks {start_week} through {end_week} of a {learning_goals} course.

    For EACH WEEK and EACH DAY, provide a natural and realistic distribution of topics that makes sense for {learning_goals}.
    
    IMPORTANT - TIME ALLOCATION: Each day has {hours_per_day} total study hours available.
    Break down each day's content into segments, specifying how much time to spend on each topic.
    The sum of hours for all topics must equal exactly {hours_per_day} hours for each day.

    Consider these topics as guidance: 
    {batch_topics}

    For each topic, find one specific, high-quality resource (YouTube video, article, tutorial).
    Each resource must include both a descriptive title AND a URL.

    Your response should be a JSON object with this exact structure:
    {{
      "week{start_week}": {{
        "day1": [
          {{
            "topic": "Topic Name", 
            "resource": "Resource Title - https://link",
            "hours": 1.5
          }},
          {{
            "topic": "Another Topic", 
            "resource": "Resource Title - https://link",
            "hours": 0.5
          }},
          ...
        ],
        "day2": [...],
        ...
        "day{days_per_week}": [...]
      }},
      "week{start_week+1}": {{
        ...
      }},
      ...
      "week{end_week}": {{
        ...
      }}
    }}

    REQUIREMENTS:
    1. Include ALL weeks from week{start_week} to week{end_week}
    2. For each week, include ALL {days_per_week} days from day1 to day{days_per_week}
    3. Each day should have a realistic number of topics
    4. Each day MUST have topics that SUM UP to exactly {hours_per_day} total hours
    5. Resources must be high-quality, relevant to {learning_goals}
    6. Return ONLY valid JSON - no explanations or text before/after

    Focus on appropriate {learning_goals} topics that would be covered at each stage of the learning journey.
    """

    # Call batch LLM to generate content with streaming
    try:
        logger.info(f"Calling LLM for streaming batch processing weeks {start_week}-{end_week}")
        batch_response = await batch_llm.ainvoke(
            [HumanMessage(content=batch_prompt)],
            callbacks=[streaming_handler]
        )
        logger.info(f"Received batch content for weeks {start_week}-{end_week}")
        
        # Extract and parse JSON
        batch_json_str = extract_json(batch_response.content)
        batch_data = json.loads(batch_json_str)
        
        # Validate and fix each week in the batch (same as before)
        for week_num in range(start_week, end_week + 1):
            week_key = f"week{week_num}"
            
            # Check if week exists in response
            if week_key not in batch_data:
                logger.warning(f"Missing {week_key} in batch response, generating separately")
                batch_data[week_key] = await process_single_week(
                    week_num, 
                    days_per_week, 
                    hours_per_day, 
                    learning_goals, 
                    full_plan
                )
                continue
                
            # Quick validation of week structure
            week_data = batch_data[week_key]
            for day in range(1, days_per_week + 1):
                day_key = f"day{day}"
                
                # Check if day exists and has content
                if day_key not in week_data or not week_data[day_key]:
                    logger.warning(f"Missing {day_key} in {week_key}, generating content")
                    week_data[day_key] = await generate_day_content(
                        week_num, 
                        day, 
                        hours_per_day,
                        learning_goals
                    )
                    continue
                    
                # Quick validation of hour allocations - only fix if seriously wrong
                day_content = week_data[day_key]
                try:
                    total_hours = sum(float(topic.get('hours', 0)) for topic in day_content)
                    if abs(total_hours - hours_per_day) > 0.5:  # More tolerant threshold
                        logger.warning(f"Major hour allocation issue in {week_key}/{day_key}: {total_hours} vs {hours_per_day}")
                        # Fix hour allocation in place rather than regenerating
                        adjust_hours(day_content, hours_per_day)
                except Exception as e:
                    logger.error(f"Error validating hours for {week_key}/{day_key}: {str(e)}")
                    week_data[day_key] = await generate_day_content(
                        week_num, 
                        day, 
                        hours_per_day,
                        learning_goals
                    )
        
        # Add the batch results to overall result
        batch_result.update(batch_data)
        logger.info(f"Successfully processed batch for weeks {start_week}-{end_week}")
        return batch_result
        
    except Exception as e:
        logger.error(f"Error processing batch for weeks {start_week}-{end_week}: {str(e)}")
        raise

async def generate_roadmap(learning_goals: str, months: int, days_per_week: int, hours_per_day: float):
    """Generate a learning roadmap with tiered caching (Pinecone for hot data, S3 for cold data)"""
    logger.info(f"Generating roadmap for: {learning_goals}, {months} months, {days_per_week} days/week, {hours_per_day} hours/day")

    # Check cache first (Pinecone, then S3 if needed)
    cache_key = f"{learning_goals}_{months}_{days_per_week}_{hours_per_day}"
    cached_roadmap = get_cached_roadmap(cache_key)
    if cached_roadmap:
        logger.info(f"Cache hit for: {learning_goals}")
        return json.dumps(cached_roadmap, indent=2)

    logger.info(f"Cache miss for: {learning_goals}, generating with LLM...")

    # Calculate total weeks for better prompting
    total_weeks = months * 4

    # 1. First LLM call — get complete outline of all topics
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week, hours_per_day)
    logger.info("Calling LLM for initial topic plan")
    try:
        # Create a non-streaming LLM for this specific call
        non_streaming_llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            streaming=False  # Important: set streaming to False
        )
        day_topic_plan = await non_streaming_llm.ainvoke([HumanMessage(content=prompt1)])
        logger.info("Received initial topic plan from LLM")
    except Exception as e:
        logger.error(f"Error from LLM when generating initial plan: {str(e)}")
        raise RuntimeError(f"Failed to generate initial roadmap plan: {str(e)}")

    # The rest of your function remains the same...
    # 2. Process the entire roadmap using batch processing and concurrency
    logger.info(f"Processing roadmap for {total_weeks} weeks using batched approach")
    complete_roadmap = {}

    # Determine batching strategy based on total weeks
    batch_size = min(MAX_WEEKS_PER_BATCH, max(1, total_weeks // 6))  # Adjust batch size dynamically
    
    # Create batch processing tasks
    batches = []
    for start_week in range(1, total_weeks + 1, batch_size):
        end_week = min(start_week + batch_size - 1, total_weeks)
        batches.append((start_week, end_week))
    
    # Process batches concurrently in groups to avoid API rate limits
    for i in range(0, len(batches), CONCURRENT_BATCHES):
        batch_group = batches[i:i+CONCURRENT_BATCHES]
        batch_tasks = [
            process_week_batch(
                start_week, end_week, days_per_week, hours_per_day, 
                learning_goals, day_topic_plan.content
            ) 
            for start_week, end_week in batch_group
        ]
        
        # Run batch tasks concurrently
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Merge results into complete roadmap
        for result in batch_results:
            complete_roadmap.update(result)
        
        logger.info(f"Completed batch group {i//CONCURRENT_BATCHES + 1}/{(len(batches) + CONCURRENT_BATCHES - 1)//CONCURRENT_BATCHES}")

    # Cache the result in Pinecone
    logger.info("Caching generated roadmap")
    cache_roadmap(cache_key, complete_roadmap)

    # Final validation and formatting
    logger.info("Roadmap generation complete")
    return json.dumps(complete_roadmap, indent=2)

# Save roadmap to database
async def save_roadmap(db: Session, user_id: str, learning_goals: str, 
                      months: int, days_per_week: int, hours_per_day: float, 
                      content: dict):
    # Create new roadmap
    db_roadmap = Roadmap(
        user_id=user_id,
        learning_goals=learning_goals,
        months=months,
        days_per_week=days_per_week,
        hours_per_day=hours_per_day,
        content=content
    )
    
    db.add(db_roadmap)
    db.commit()
    db.refresh(db_roadmap)
    
    # Create progress items for each topic in the roadmap
    for week_num in range(1, months * 4 + 1):
        week_key = f"week{week_num}"
        if week_key in content:
            for day_num in range(1, days_per_week + 1):
                day_key = f"day{day_num}"
                if day_key in content[week_key]:
                    for topic_idx, _ in enumerate(content[week_key][day_key]):
                        progress_item = Progress(
                            roadmap_id=db_roadmap.id,
                            week_number=week_num,
                            day_number=day_num,
                            topic_index=topic_idx,
                            completed=False
                        )
                        db.add(progress_item)
    
    db.commit()
    logger.info(f"Saved roadmap to database: {db_roadmap.id}")
    return db_roadmap

# Get user roadmaps
async def get_user_roadmaps(db: Session, user_id: str):
    roadmaps = db.query(Roadmap).filter(Roadmap.user_id == user_id).all()
    return roadmaps

# Get roadmap with progress
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_roadmap_with_progress(db: Session, roadmap_id: str, user_id: str):
    try:
        logger.info(f"Fetching roadmap {roadmap_id} for user {user_id}")
        
        # Get roadmap and verify it belongs to the user
        roadmap = db.query(Roadmap).filter(
            Roadmap.id == roadmap_id,
            Roadmap.user_id == user_id
        ).first()
        
        if not roadmap:
            logger.warning(f"Roadmap {roadmap_id} not found or doesn't belong to user {user_id}")
            return None
        
        logger.info(f"Found roadmap: {roadmap.id}")
        
        # Safely handle content - ensure it's a dict even if None
        content = roadmap.content if roadmap.content is not None else {}
        
        # Get progress items
        progress_items = db.query(Progress).filter(
            Progress.roadmap_id == roadmap_id
        ).all()
        
        logger.info(f"Found {len(progress_items)} progress items")
        
        # Convert to dictionary for easy access
        progress_dict = {}
        for item in progress_items:
            week_key = f"week{item.week_number}"
            day_key = f"day{item.day_number}"
            
            if week_key not in progress_dict:
                progress_dict[week_key] = {}
            
            if day_key not in progress_dict[week_key]:
                progress_dict[week_key][day_key] = []
            
            # Handle completed_at safely
            completed_at = None
            if item.completed_at:
                try:
                    completed_at = item.completed_at.isoformat()
                except AttributeError:
                    # In case completed_at is not a datetime object
                    completed_at = str(item.completed_at)
            
            progress_dict[week_key][day_key].append({
                "topic_index": item.topic_index,
                "completed": item.completed,
                "completed_at": completed_at
            })
        
        # Create result with roadmap and progress
        result = {
            "id": roadmap.id,
            "learning_goals": roadmap.learning_goals,
            "months": roadmap.months,
            "days_per_week": roadmap.days_per_week,
            "hours_per_day": float(roadmap.hours_per_day),  # Ensure this is a float
            "content": content,
            "progress": progress_dict,
            "created_at": roadmap.created_at.isoformat() if isinstance(roadmap.created_at, datetime) else str(roadmap.created_at)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in get_roadmap_with_progress: {str(e)}")
        logger.error(traceback.format_exc())
        # Re-raise with more context
        raise Exception(f"Failed to retrieve roadmap: {str(e)}")

# Update progress
async def update_progress(db: Session, roadmap_id: str, user_id: str, 
                         week_number: int, day_number: int, 
                         topic_index: int, completed: bool):
    # Verify roadmap belongs to the user
    roadmap = db.query(Roadmap).filter(
        Roadmap.id == roadmap_id,
        Roadmap.user_id == user_id
    ).first()
    
    if not roadmap:
        return None
    
    # Get the progress item
    progress_item = db.query(Progress).filter(
        Progress.roadmap_id == roadmap_id,
        Progress.week_number == week_number,
        Progress.day_number == day_number,
        Progress.topic_index == topic_index
    ).first()
    
    from datetime import datetime

    if not progress_item:
        # 👉 create new progress item if not exists
        progress_item = Progress(
            user_id=user_id,
            roadmap_id=roadmap_id,
            week_number=week_number,
            day_number=day_number,
            topic_index=topic_index,
            completed=completed,
            completed_at=datetime.utcnow() if completed else None
        )
        db.add(progress_item)
    else:
        # 👉 update if exists
        progress_item.completed = completed
        progress_item.completed_at = datetime.utcnow() if completed else None

    db.commit()
    db.refresh(progress_item)
    
    return progress_item

async def process_week_batch(start_week, end_week, days_per_week, hours_per_day, learning_goals, full_plan):
    """Process a batch of consecutive weeks at once."""
    logger.info(f"Processing week batch {start_week}-{end_week}")
    batch_result = {}
    
    # Extract topics for all weeks in this batch
    batch_topics = ""
    for week_num in range(start_week, end_week + 1):
        week_topics = extract_week_from_plan(full_plan, week_num)
        batch_topics += f"\nWEEK {week_num} TOPICS:\n{week_topics}\n"
    
    # Generate a prompt for the entire batch
    batch_prompt = f"""
    You are creating a detailed learning roadmap for weeks {start_week} through {end_week} of a {learning_goals} course.

    For EACH WEEK and EACH DAY, provide a natural and realistic distribution of topics that makes sense for {learning_goals}.
    
    IMPORTANT - TIME ALLOCATION: Each day has {hours_per_day} total study hours available.
    Break down each day's content into segments, specifying how much time to spend on each topic.
    The sum of hours for all topics must equal exactly {hours_per_day} hours for each day.

    Consider these topics as guidance: 
    {batch_topics}

    For each topic, find one specific, high-quality resource (YouTube video, article, tutorial).
    Each resource must include both a descriptive title AND a URL.

    Your response should be a JSON object with this exact structure:
    {{
      "week{start_week}": {{
        "day1": [
          {{
            "topic": "Topic Name", 
            "resource": "Resource Title - https://link",
            "hours": 1.5
          }},
          {{
            "topic": "Another Topic", 
            "resource": "Resource Title - https://link",
            "hours": 0.5
          }},
          ...
        ],
        "day2": [...],
        ...
        "day{days_per_week}": [...]
      }},
      "week{start_week+1}": {{
        ...
      }},
      ...
      "week{end_week}": {{
        ...
      }}
    }}

    REQUIREMENTS:
    1. Include ALL weeks from week{start_week} to week{end_week}
    2. For each week, include ALL {days_per_week} days from day1 to day{days_per_week}
    3. Each day should have a realistic number of topics
    4. Each day MUST have topics that SUM UP to exactly {hours_per_day} total hours
    5. Resources must be high-quality, relevant to {learning_goals}
    6. Return ONLY valid JSON - no explanations or text before/after

    Focus on appropriate {learning_goals} topics that would be covered at each stage of the learning journey.
    """

    # Call batch LLM to generate content for multiple weeks at once
    try:
        logger.info(f"Calling LLM for batch processing weeks {start_week}-{end_week}")
        batch_response = await batch_llm.ainvoke([HumanMessage(content=batch_prompt)])
        logger.info(f"Received batch content for weeks {start_week}-{end_week}")
        
        # Extract and parse JSON
        batch_json_str = extract_json(batch_response.content)
        batch_data = json.loads(batch_json_str)
        
        # Validate and fix each week in the batch
        for week_num in range(start_week, end_week + 1):
            week_key = f"week{week_num}"
            
            # Check if week exists in response
            if week_key not in batch_data:
                logger.warning(f"Missing {week_key} in batch response, generating separately")
                batch_data[week_key] = await process_single_week(
                    week_num, 
                    days_per_week, 
                    hours_per_day, 
                    learning_goals, 
                    full_plan
                )
                continue
                
            # Quick validation of week structure
            week_data = batch_data[week_key]
            for day in range(1, days_per_week + 1):
                day_key = f"day{day}"
                
                # Check if day exists and has content
                if day_key not in week_data or not week_data[day_key]:
                    logger.warning(f"Missing {day_key} in {week_key}, generating content")
                    week_data[day_key] = await generate_day_content(
                        week_num, 
                        day, 
                        hours_per_day,
                        learning_goals
                    )
                    continue
                    
                # Quick validation of hour allocations - only fix if seriously wrong
                day_content = week_data[day_key]
                try:
                    total_hours = sum(float(topic.get('hours', 0)) for topic in day_content)
                    if abs(total_hours - hours_per_day) > 0.5:  # More tolerant threshold
                        logger.warning(f"Major hour allocation issue in {week_key}/{day_key}: {total_hours} vs {hours_per_day}")
                        # Fix hour allocation in place rather than regenerating
                        adjust_hours(day_content, hours_per_day)
                except Exception as e:
                    logger.error(f"Error validating hours for {week_key}/{day_key}: {str(e)}")
                    week_data[day_key] = await generate_day_content(
                        week_num, 
                        day, 
                        hours_per_day,
                        learning_goals
                    )
        
        # Add the batch results to overall result
        batch_result.update(batch_data)
        logger.info(f"Successfully processed batch for weeks {start_week}-{end_week}")
        
    except Exception as e:
        logger.error(f"Error processing batch for weeks {start_week}-{end_week}: {str(e)}")
        
        # Fall back to processing each week individually
        logger.warning(f"Falling back to individual week processing for weeks {start_week}-{end_week}")
        week_tasks = [
            process_single_week(
                week_num, 
                days_per_week, 
                hours_per_day, 
                learning_goals, 
                full_plan
            ) 
            for week_num in range(start_week, end_week + 1)
        ]
        
        week_results = await asyncio.gather(*week_tasks)
        
        for week_num, week_data in zip(range(start_week, end_week + 1), week_results):
            batch_result[f"week{week_num}"] = week_data
    
    return batch_result

def adjust_hours(day_content, target_hours):
    """Adjust hours in a day's content to match the target total."""
    if not day_content:
        return day_content
        
    # Calculate current total
    current_total = sum(float(topic.get('hours', 0)) for topic in day_content)
    
    # If current total is zero, distribute hours evenly
    if current_total == 0:
        hours_per_topic = round(target_hours / len(day_content), 1)
        remaining = target_hours
        
        for i in range(len(day_content) - 1):
            day_content[i]['hours'] = hours_per_topic
            remaining -= hours_per_topic
            
        day_content[-1]['hours'] = round(remaining, 1)
    else:
        # Adjust proportionally
        scale_factor = target_hours / current_total
        remaining = target_hours
        
        for i in range(len(day_content) - 1):
            adjusted = round(float(day_content[i].get('hours', 0)) * scale_factor, 1)
            day_content[i]['hours'] = adjusted
            remaining -= adjusted
            
        day_content[-1]['hours'] = round(remaining, 1)
    
    return day_content

async def process_single_week(week_num, days_per_week, hours_per_day, learning_goals, full_plan):
    """Process a single week to ensure all days are populated with content."""
    logger.info(f"Processing content for week {week_num}")

    # Extract topics for this week if they exist in the plan
    week_topics = extract_week_from_plan(full_plan, week_num)
    logger.info(f"Extracted topics for week {week_num}")

    # Generate a prompt specifically for this week with hour-based scheduling
    week_prompt = f"""
    You are creating a detailed learning roadmap for week {week_num} of a {learning_goals} course.

    Create content for ALL {days_per_week} days of week {week_num}. 
    For EACH DAY, provide a natural and realistic distribution of topics that makes sense for {learning_goals}.
    
    IMPORTANT - TIME ALLOCATION: Each day has {hours_per_day} total study hours available.
    Break down each day's content into segments, specifying how much time to spend on each topic.
    The sum of hours for all topics must equal exactly {hours_per_day} hours for each day.

    If available, use these topics as guidance: 
    {week_topics}

    For each topic, find one specific, high-quality resource (YouTube video, article, tutorial).
    Each resource must include both a descriptive title AND a URL.

    Your response should be a JSON object with this exact structure:
    {{
      "day1": [
        {{
          "topic": "Topic Name", 
          "resource": "Resource Title - https://link",
          "hours": 1.5  // Time to spend on this topic
        }},
        {{
          "topic": "Another Topic", 
          "resource": "Resource Title - https://link",
          "hours": 0.5  // Time to spend on this topic
        }},
        ...
      ],
      "day2": [...],
      ...
      "day{days_per_week}": [...]
    }}

    REQUIREMENTS:
    1. Include ALL {days_per_week} days from day1 to day{days_per_week}
    2. Each day should have a realistic number of topics - don't force any specific number
    3. Each day MUST have topics that SUM UP to exactly {hours_per_day} total hours
    4. Resources must be high-quality, relevant to {learning_goals}
    5. Return ONLY valid JSON - no explanations or text before/after

    For week {week_num}, focus on appropriate {learning_goals} topics that would be covered at this stage.
    """

    # Call LLM to generate content for this week
    try:
        logger.info(f"Calling LLM for week {week_num} content")
        week_response = await llm.ainvoke([HumanMessage(content=week_prompt)])
        logger.info(f"Received week {week_num} content from LLM")
    except Exception as e:
        logger.error(f"Error generating content for week {week_num}: {str(e)}")
        return await generate_fallback_week(week_num, days_per_week, hours_per_day, learning_goals)

    # Extract and parse JSON
    try:
        week_json_str = extract_json(week_response.content)
        week_data = json.loads(week_json_str)
        logger.info(f"Successfully parsed JSON for week {week_num}")

        # Quick validation that all days are present
        for day in range(1, days_per_week + 1):
            day_key = f"day{day}"
            
            # Check if day exists
            if day_key not in week_data or not week_data[day_key]:
                logger.warning(f"Missing {day_key} in week {week_num}, generating content")
                week_data[day_key] = await generate_day_content(
                    week_num, 
                    day, 
                    hours_per_day,
                    learning_goals
                )
                continue
                
            # Quick validate hour allocations
            day_content = week_data[day_key]
            try:
                total_hours = sum(float(topic.get('hours', 0)) for topic in day_content)
                if abs(total_hours - hours_per_day) > 0.5:  # More tolerant threshold
                    logger.warning(f"Major hour allocation issue: {total_hours} vs {hours_per_day}")
                    # Fix in place
                    adjust_hours(day_content, hours_per_day)
            except Exception as e:
                logger.error(f"Error validating hours: {str(e)}")
                week_data[day_key] = await generate_day_content(
                    week_num, 
                    day, 
                    hours_per_day,
                    learning_goals
                )

        return week_data

    except Exception as e:
        logger.error(f"Error processing week {week_num}: {str(e)}")
        # If parsing fails, generate a simpler structure
        return await generate_fallback_week(week_num, days_per_week, hours_per_day, learning_goals)

async def generate_day_content(week_num, day_num, hours_per_day, learning_goals):
    """Generate content for a specific day if it's missing."""
    logger.info(f"Generating content for week {week_num}, day {day_num}")
    day_prompt = f"""
    Generate a natural and realistic study plan for day {day_num} of week {week_num} 
    for a learning roadmap on {learning_goals}.
    
    IMPORTANT: 
    1. There are {hours_per_day} total study hours available for this day.
    2. Create a NATURAL distribution of topics - don't force exactly 2-4 topics if that's not appropriate.
    3. For complex topics that need focus, it's better to have fewer topics with more time allocated.
    4. For simpler topics, having more topics with less time each might make sense.
    5. The topics should be directly relevant to {learning_goals} and appropriate for week {week_num}.
    
    Specify how many hours to spend on each topic, ensuring the total equals exactly {hours_per_day}.

    Return ONLY a JSON array like this:
    [
      {{
        "topic": "Topic Name", 
        "resource": "Resource Title - https://link",
        "hours": 1.5
      }},
      {{
        "topic": "Topic Name", 
        "resource": "Resource Title - https://link",
        "hours": 0.5
      }},
      ...
    ]
    
    Make sure the hours for all topics sum up to exactly {hours_per_day}.
    """

    try:
        day_response = await llm.ainvoke([HumanMessage(content=day_prompt)])
        day_json_str = extract_json(day_response.content)
        day_data = json.loads(day_json_str)
        
        # Verify and adjust hour allocation if needed
        adjust_hours(day_data, hours_per_day)
        logger.info(f"Successfully generated content for week {week_num}, day {day_num}")
        return day_data
    except Exception as e:
        logger.error(f"Error generating day content: {str(e)}")
        # Emergency fallback
        return generate_emergency_day_content(week_num, day_num, hours_per_day, learning_goals)

def generate_emergency_day_content(week_num, day_num, hours_per_day, learning_goals):
    """Generate emergency fallback content that's relevant to the learning goals"""
    logger.warning(f"Using emergency fallback content for week {week_num}, day {day_num}")
    
    # Make another attempt with a simpler prompt
    try:
        simplified_prompt = f"""
        For a learning path on '{learning_goals}', provide 2-3 relevant topics for week {week_num}, day {day_num}.
        For each topic include:
        1. A specific topic name relevant to {learning_goals}
        2. A real resource (title and link) 
        3. Hours to allocate (total must be {hours_per_day})
        
        Return as JSON array:
        [
          {{"topic": "Topic name", "resource": "Resource title - https://link", "hours": X}}
        ]
        """
        
        # Use synchronous call to avoid nested awaits
        fallback_response = llm.invoke([HumanMessage(content=simplified_prompt)])
        fallback_json_str = extract_json(fallback_response.content)
        fallback_data = json.loads(fallback_json_str)
        
        # Verify hours total
        adjust_hours(fallback_data, hours_per_day)
        return fallback_data
        
    except Exception as e:
        logger.error(f"Emergency fallback failed again: {str(e)}")
        # Last resort - generate content based on learning goals with no LLM call
        
        # Define topic templates based on common learning domains
        templates = {
            "python": [
                {"topic": "Python Fundamentals", 
                 "resource": "Python Crash Course - https://nostarch.com/pythoncrashcourse2e"},
                {"topic": "Python Data Structures", 
                 "resource": "Python Data Structures Tutorial - https://realpython.com/python-data-structures/"}
            ],
            "javascript": [
                {"topic": "JavaScript Basics", 
                 "resource": "JavaScript.info - https://javascript.info/"},
                {"topic": "DOM Manipulation", 
                 "resource": "MDN Web Docs - https://developer.mozilla.org/en-US/docs/Web/API/Document_Object_Model"}
            ],
            "machine learning": [
                {"topic": "ML Fundamentals", 
                 "resource": "Machine Learning Crash Course - https://developers.google.com/machine-learning/crash-course"},
                {"topic": "Supervised Learning Basics", 
                 "resource": "Introduction to Statistical Learning - https://www.statlearning.com/"}
            ],
            "data science": [
                {"topic": "Data Analysis with Pandas", 
                 "resource": "Python for Data Analysis - https://wesmckinney.com/book/"},
                {"topic": "Data Visualization Techniques", 
                 "resource": "Data Visualization with Python - https://plotly.com/python/"}
            ],
            "web development": [
                {"topic": "HTML & CSS Fundamentals", 
                 "resource": "MDN Web Docs - https://developer.mozilla.org/en-US/docs/Web"},
                {"topic": "Responsive Design", 
                 "resource": "Responsive Web Design - https://www.w3schools.com/css/css_rwd_intro.asp"}
            ],
            "mobile development": [
                {"topic": "Mobile App Architecture", 
                 "resource": "Mobile App Development Guide - https://developer.android.com/guide"},
                {"topic": "UI/UX for Mobile", 
                 "resource": "Mobile UI Design Patterns - https://www.smashingmagazine.com/2021/05/ux-design-mobile-apps/"}
            ],
            "devops": [
                {"topic": "CI/CD Fundamentals", 
                 "resource": "CI/CD Pipeline Tutorial - https://www.atlassian.com/continuous-delivery/principles/continuous-integration-vs-delivery-vs-deployment"},
                {"topic": "Docker Containerization", 
                 "resource": "Docker Documentation - https://docs.docker.com/get-started/"}
            ],
            "cloud computing": [
                {"topic": "Cloud Architecture", 
                 "resource": "AWS Architecture Center - https://aws.amazon.com/architecture/"},
                {"topic": "Serverless Computing", 
                 "resource": "Serverless Framework - https://www.serverless.com/"}
            ],
            "blockchain": [
                {"topic": "Blockchain Fundamentals", 
                 "resource": "Blockchain Basics - https://www.coursera.org/learn/blockchain-basics"},
                {"topic": "Smart Contracts", 
                 "resource": "Solidity Documentation - https://docs.soliditylang.org/"}
            ],
            "cybersecurity": [
                {"topic": "Security Fundamentals", 
                 "resource": "OWASP Top 10 - https://owasp.org/www-project-top-ten/"},
                {"topic": "Penetration Testing", 
                 "resource": "Penetration Testing: A Hands-On Introduction - https://nostarch.com/pentesting"}
            ],
            "artificial intelligence": [
                {"topic": "AI Fundamentals", 
                 "resource": "AI For Everyone - https://www.coursera.org/learn/ai-for-everyone"},
                {"topic": "Neural Networks", 
                 "resource": "Neural Networks and Deep Learning - https://www.coursera.org/learn/neural-networks-deep-learning"}
            ]
        }
        
        # Find best matching template for the learning goals
        best_match = None
        highest_score = 0
        
        # Try to find the most relevant template based on keyword matching
        for key in templates:
            # Calculate similarity score (simple word overlap for now)
            if key.lower() in learning_goals.lower():
                # Direct match gets high score
                score = len(key)
                if score > highest_score:
                    highest_score = score
                    best_match = key
        
        # Use default if no match found
        if not best_match:
            # Try to use "data science" as a reasonable default
            if "data" in learning_goals.lower() or "analytics" in learning_goals.lower():
                best_match = "data science"
            elif "web" in learning_goals.lower() or "html" in learning_goals.lower() or "css" in learning_goals.lower():
                best_match = "web development"
            elif "python" in learning_goals.lower():
                best_match = "python"
            elif "javascript" in learning_goals.lower() or "js" in learning_goals.lower():
                best_match = "javascript"
            elif "machine" in learning_goals.lower() or "ml" in learning_goals.lower():
                best_match = "machine learning"
            elif "ai" in learning_goals.lower() or "intelligent" in learning_goals.lower():
                best_match = "artificial intelligence"
            else:
                best_match = "data science"  # Ultimate fallback
            
        # Get template topics and adjust hours
        topics = templates[best_match]
        
        # Make topic names more specific to the learning goals
        for topic in topics:
            # Add week and day context to topic names to make them more relevant
            if week_num <= 4:
                topic["topic"] = f"Introduction to {topic['topic']}"
            elif week_num <= 8:
                topic["topic"] = f"Intermediate {topic['topic']}"
            else:
                topic["topic"] = f"Advanced {topic['topic']}"
        
        # Distribute hours more naturally
        if hours_per_day <= 2:
            # For short study days, focus on one topic
            return [{"topic": topics[0]["topic"], "resource": topics[0]["resource"], "hours": hours_per_day}]
        else:
            # For longer study days, distribute between topics
            hours1 = round(hours_per_day * 0.6, 1)
            hours2 = round(hours_per_day - hours1, 1)
            
            # Create fallback content
            fallback_content = [
                {**topics[0], "hours": hours1},
                {**topics[1], "hours": hours2}
            ]
            
            return fallback_content

async def generate_fallback_week(week_num, days_per_week, hours_per_day, learning_goals):
    """Generate a fallback week structure with content for all days."""
    logger.warning(f"Generating fallback content for week {week_num}")
    week_data = {}

    # Process days in parallel for fallback
    day_tasks = []
    for day in range(1, days_per_week + 1):
        day_tasks.append(generate_day_content(week_num, day, hours_per_day, learning_goals))
    
    day_contents = await asyncio.gather(*day_tasks)
    
    for day, content in enumerate(day_contents, 1):
        day_key = f"day{day}"
        week_data[day_key] = content

    return week_data

def extract_week_from_plan(plan_text, week_num):
    """Extract topics for a specific week from the full plan."""
    logger.info(f"Extracting week {week_num} topics from plan")
    result = []
    current_week = None
    lines = plan_text.strip().split('\n')

    week_pattern = re.compile(r'Week\s+(\d+):', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        week_match = week_pattern.match(line)
        if week_match:
            week_num_found = int(week_match.group(1))
            if week_num_found == week_num:
                current_week = week_num
                result.append(line)
            elif current_week == week_num and week_num_found > week_num:
                # We've moved past our target week
                break
        elif current_week == week_num:
            result.append(line)

    extracted_text = '\n'.join(result)
    logger.info(f"Extracted {len(result)} lines for week {week_num}")
    return extracted_text

def extract_json(text):
    """Extract JSON content from text that might contain explanations."""
    logger.info("Extracting JSON from LLM response")

    # Try to find content between triple backticks (```json ... ```)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        logger.info("Found JSON between backticks")
        return json_match.group(1)

    # If no backticks, look for content that starts with { and ends with }
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        logger.info("Found JSON between curly braces")
        return json_match.group(1)

    # If searching for object failed, try looking for an array
    json_match = re.search(r'(\[[\s\S]*\])', text)
    if json_match:
        logger.info("Found JSON array")
        return json_match.group(1)

    # If all else fails, return the original text
    logger.warning("Could not extract JSON with regex, returning full text")
    return text

async def archive_cold_data(access_threshold=5, days_threshold=30):
    """Run the archiving process to move cold data from Pinecone to S3"""
    logger.info(f"Running archiving process: threshold={access_threshold} accesses, {days_threshold} days")
    try:
        archived_count = roadmap_cache.archive_cold_data(
            access_threshold=access_threshold,
            days_threshold=days_threshold
        )
        logger.info(f"Archived {archived_count} items to S3")
        return archived_count
    except Exception as e:
        logger.error(f"Error in archive_cold_data: {str(e)}")
        return 0
