from app.utils.prompt_utils import filter_prompt, verify_prompt, format_json_prompt, generate_initial_prompt
from app.utils.resource_fetcher import fetch_real_resources
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

async def generate_roadmap(learning_goals: str, months: int, days_per_week: int):
    # Calculate total weeks for better prompting
    total_weeks = months * 4
    
    # 1. First LLM call — get complete outline of all topics
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week)
    day_topic_plan = await llm.ainvoke([HumanMessage(content=prompt1)])
    
    # 2. Process the entire roadmap week by week to ensure completeness
    complete_roadmap = {}
    
    for week_num in range(1, total_weeks + 1):
        # Process each week individually to ensure proper handling
        week_content = await process_single_week(
            week_num, 
            days_per_week, 
            learning_goals, 
            day_topic_plan.content
        )
        
        # Add the week to our complete roadmap
        complete_roadmap[f"week{week_num}"] = week_content
    
    # Final validation and formatting
    return json.dumps(complete_roadmap, indent=2)

async def process_single_week(week_num, days_per_week, learning_goals, full_plan):
    """Process a single week to ensure all days are populated with content."""
    
    # Extract topics for this week if they exist in the plan
    week_topics = extract_week_from_plan(full_plan, week_num)
    
    # Generate a prompt specifically for this week
    week_prompt = f"""
    You are creating a detailed learning roadmap for week {week_num} of a {learning_goals} course.
    
    Create content for ALL {days_per_week} days of week {week_num}. For EACH DAY, provide 2-4 topics
    with specific, high-quality resources.
    
    If available, use these topics as guidance: 
    {week_topics}
    
    For each topic, find one specific, high-quality resource (YouTube video, article, tutorial).
    Each resource must include both a descriptive title AND a URL.
    
    Your response should be a JSON object with this exact structure:
    {{
      "day1": [
        {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
        {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
        ...
      ],
      "day2": [...],
      ...
      "day{days_per_week}": [...]
    }}
    
    REQUIREMENTS:
    1. Include ALL {days_per_week} days from day1 to day{days_per_week}
    2. Each day MUST have 2-4 topics with specific resources
    3. Resources must be high-quality, relevant to {learning_goals}
    4. Return ONLY valid JSON - no explanations or text before/after
    
    For week {week_num}, focus on appropriate {learning_goals} topics that would be covered at this stage.
    """
    
    # Call LLM to generate content for this week
    week_response = await llm.ainvoke([HumanMessage(content=week_prompt)])
    
    # Extract and parse JSON
    try:
        week_json_str = extract_json(week_response.content)
        week_data = json.loads(week_json_str)
        
        # Validate that all days are present
        for day in range(1, days_per_week + 1):
            day_key = f"day{day}"
            if day_key not in week_data or not week_data[day_key]:
                # Generate content for missing day
                week_data[day_key] = await generate_day_content(
                    week_num, 
                    day, 
                    learning_goals
                )
        
        return week_data
        
    except Exception as e:
        # If parsing fails, generate a simpler structure
        return await generate_fallback_week(week_num, days_per_week, learning_goals)

async def generate_day_content(week_num, day_num, learning_goals):
    """Generate content for a specific day if it's missing."""
    day_prompt = f"""
    Generate 2-4 relevant topics with resources for day {day_num} of week {week_num} 
    for a learning roadmap on {learning_goals}.
    
    Return ONLY a JSON array like this:
    [
      {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
      {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
      ...
    ]
    """
    
    day_response = await llm.ainvoke([HumanMessage(content=day_prompt)])
    
    try:
        day_json_str = extract_json(day_response.content)
        day_data = json.loads(day_json_str)
        return day_data
    except:
        # Emergency fallback
        return [
            {"topic": f"Week {week_num} Day {day_num} Topic 1", 
             "resource": "Python for Data Science Handbook - https://jakevdp.github.io/PythonDataScienceHandbook/"},
            {"topic": f"Week {week_num} Day {day_num} Topic 2", 
             "resource": "Machine Learning Crash Course - https://developers.google.com/machine-learning/crash-course"}
        ]

async def generate_fallback_week(week_num, days_per_week, learning_goals):
    """Generate a fallback week structure with content for all days."""
    week_data = {}
    
    for day in range(1, days_per_week + 1):
        day_key = f"day{day}"
        week_data[day_key] = await generate_day_content(week_num, day, learning_goals)
    
    return week_data

def extract_week_from_plan(plan_text, week_num):
    """Extract topics for a specific week from the full plan."""
    result = []
    current_week = None
    lines = plan_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        week_match = re.match(r'Week\s+(\d+):', line)
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
    
    return '\n'.join(result)

def extract_json(text):
    """Extract JSON content from text that might contain explanations."""
    # Try to find content between triple backticks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    
    if json_match:
        return json_match.group(1)
    
    # If no backticks, look for content that starts with { and ends with }
    json_match = re.search(r'(\{[\s\S]*\})', text)
    
    if json_match:
        return json_match.group(1)
    
    # If all else fails, return the original text
    return text