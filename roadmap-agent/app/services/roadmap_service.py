# app/services/roadmap_service.py
from app.utils.prompt_utils import filter_prompt, verify_prompt, format_json_prompt, generate_initial_prompt
from app.utils.resource_fetcher import fetch_real_resources
from app.services.cache_service import RoadmapCache
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
import json
import re
import logging
import asyncio
from datetime import datetime

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
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    logger.info("LLM initialized successfully")
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

async def generate_roadmap(learning_goals: str, months: int, days_per_week: int, hours_per_day: float):
    """Generate a learning roadmap with tiered caching (Pinecone for hot data, S3 for cold data)"""
    logger.info(f"Generating roadmap for: {learning_goals}, {months} months, {days_per_week} days/week, {hours_per_day} hours/day")

    # Check cache first (Pinecone, then S3 if needed)
    cached_roadmap = roadmap_cache.get_cached_roadmap(learning_goals, months, days_per_week, hours_per_day)
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
        day_topic_plan = await llm.ainvoke([HumanMessage(content=prompt1)])
        logger.info("Received initial topic plan from LLM")
    except Exception as e:
        logger.error(f"Error from LLM when generating initial plan: {str(e)}")
        raise RuntimeError(f"Failed to generate initial roadmap plan: {str(e)}")

    # 2. Process the entire roadmap week by week to ensure completeness
    logger.info(f"Processing roadmap for {total_weeks} weeks")
    complete_roadmap = {}

    for week_num in range(1, total_weeks + 1):
        logger.info(f"Processing week {week_num}/{total_weeks}")
        # Process each week individually to ensure proper handling
        week_content = await process_single_week(
            week_num, 
            days_per_week,
            hours_per_day,  
            learning_goals, 
            day_topic_plan.content
        )

        # Add the week to our complete roadmap
        complete_roadmap[f"week{week_num}"] = week_content
        logger.info(f"Completed week {week_num}")

    # Cache the result in Pinecone
    logger.info("Caching generated roadmap")
    roadmap_cache.cache_roadmap(learning_goals, months, days_per_week, hours_per_day, complete_roadmap)

    # Final validation and formatting
    logger.info("Roadmap generation complete")
    return json.dumps(complete_roadmap, indent=2)

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
    For EACH DAY, provide 2-4 topics with specific, high-quality resources. 
    
    IMPORTANT - TIME ALLOCATION: Each day has {hours_per_day} total study hours available.
    Break down each day's content into HOURLY segments, specifying how much time to spend on each topic.
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
    2. Each day MUST have 2-4 topics with specific resources
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

        # Validate that all days are present and have correct hour allocations
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
                
            # Validate hour allocations
            day_content = week_data[day_key]
            total_hours = 0
            hours_missing = False
            
            for topic in day_content:
                if 'hours' not in topic:
                    hours_missing = True
                    break
                total_hours += float(topic['hours'])
            
            # If hours are missing or don't add up to the required total, regenerate
            if hours_missing or abs(total_hours - hours_per_day) > 0.1:
                logger.warning(f"Incorrect hour allocation for {day_key} in week {week_num}: {total_hours} vs {hours_per_day}, regenerating")
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
    Generate 2-4 relevant topics with resources for day {day_num} of week {week_num} 
    for a learning roadmap on {learning_goals}.
    
    IMPORTANT: There are {hours_per_day} total study hours available for this day.
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
        
        # Verify hour allocation
        total_hours = sum(float(topic.get('hours', 0)) for topic in day_data)
        
        if abs(total_hours - hours_per_day) > 0.1:
            logger.warning(f"Generated day content has incorrect hours: {total_hours} vs {hours_per_day}, adjusting")
            # Adjust hours to match exactly
            adjustment_factor = hours_per_day / total_hours if total_hours > 0 else 1
            remaining_hours = hours_per_day
            
            for i in range(len(day_data) - 1):
                adjusted_hours = round(float(day_data[i]['hours']) * adjustment_factor, 1)
                day_data[i]['hours'] = adjusted_hours
                remaining_hours -= adjusted_hours
                
            # Assign remaining hours to last topic
            day_data[-1]['hours'] = round(remaining_hours, 1)
            
        logger.info(f"Successfully generated content for week {week_num}, day {day_num}")
        return day_data
    except Exception as e:
        logger.error(f"Error generating day content: {str(e)}")
        # Emergency fallback
        return generate_emergency_day_content(week_num, day_num, hours_per_day)

def generate_emergency_day_content(week_num, day_num, hours_per_day):
    """Generate emergency fallback content when all else fails"""
    logger.warning(f"Using emergency fallback content for week {week_num}, day {day_num}")
    
    # Split hours between two topics
    hours1 = round(hours_per_day * 0.6, 1)
    hours2 = round(hours_per_day - hours1, 1)
    
    return [
        {"topic": f"Week {week_num} Day {day_num} Topic 1", 
         "resource": "Python for Data Science Handbook - https://jakevdp.github.io/PythonDataScienceHandbook/",
         "hours": hours1},
        {"topic": f"Week {week_num} Day {day_num} Topic 2", 
         "resource": "Machine Learning Crash Course - https://developers.google.com/machine-learning/crash-course",
         "hours": hours2}
    ]

async def generate_fallback_week(week_num, days_per_week, hours_per_day, learning_goals):
    """Generate a fallback week structure with content for all days."""
    logger.warning(f"Generating fallback content for week {week_num}")
    week_data = {}

    for day in range(1, days_per_week + 1):
        day_key = f"day{day}"
        week_data[day_key] = await generate_day_content(week_num, day, hours_per_day, learning_goals)

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