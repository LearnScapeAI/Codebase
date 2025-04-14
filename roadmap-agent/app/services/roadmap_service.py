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
        total_hours = sum(float(topic.get('hours', 0)) for topic in fallback_data)
        if abs(total_hours - hours_per_day) > 0.1:
            # Simple fix - distribute hours evenly
            hours_per_topic = round(hours_per_day / len(fallback_data), 1)
            remaining = round(hours_per_day, 1)
            
            for i in range(len(fallback_data)-1):
                fallback_data[i]['hours'] = hours_per_topic
                remaining -= hours_per_topic
                
            fallback_data[-1]['hours'] = remaining
            
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