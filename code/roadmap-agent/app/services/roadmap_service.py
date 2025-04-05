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
    
    # 1. First LLM call — break down roadmap into weeks -> days -> topics
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week)
    day_topic_plan = await llm.ainvoke([HumanMessage(content=prompt1)])

    # 2. Break down the roadmap by chunks to handle large roadmaps better
    # Process in batches of 4 weeks at a time
    complete_roadmap = {}
    
    for batch_start in range(1, total_weeks + 1, 4):
        batch_end = min(batch_start + 3, total_weeks)
        
        # Extract just this batch of weeks from the plan
        batch_plan = extract_weeks_from_plan(day_topic_plan.content, batch_start, batch_end)
        
        # Process this batch of weeks
        prompt2 = filter_prompt_for_batch(batch_plan, batch_start, batch_end)
        batch_resources = await llm.ainvoke([HumanMessage(content=prompt2)])
        
        # Structure this batch into JSON
        structure_prompt = f"""
        Generate a JSON with detailed resources for weeks {batch_start} to {batch_end} of the learning roadmap.
        
        The JSON must follow this exact structure:
        {{
          "week{batch_start}": {{
            "day1": [
              {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
              {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
              ...
            ],
            "day2": [...],
            ...
          }},
          "week{batch_start+1}": {{
            ...
          }},
          ...
        }}
        
        REQUIREMENTS:
        1. Include ALL weeks from week{batch_start} to week{batch_end}
        2. Include ALL {days_per_week} days for each week
        3. Each day MUST have 2-4 topics with specific resources
        4. Resources must be high-quality, relevant, and include both a title and URL
        5. Format JSON correctly with quotes, commas, and brackets
        6. ONLY output valid JSON with no explanation text
        
        Content to use:
        {batch_resources.content}
        """
        
        batch_roadmap = await llm.ainvoke([HumanMessage(content=structure_prompt)])
        
        # Parse and validate this batch
        try:
            batch_json_content = extract_json(batch_roadmap.content)
            batch_json = json.loads(batch_json_content)
            
            # Check if all expected weeks are present
            for week_num in range(batch_start, batch_end + 1):
                week_key = f"week{week_num}"
                if week_key not in batch_json:
                    # Create missing week with placeholder structure
                    batch_json[week_key] = generate_placeholder_week(days_per_week)
            
            # Update complete roadmap with this batch
            complete_roadmap.update(batch_json)
            
        except Exception as e:
            # If JSON parsing fails, try to fix it
            fix_prompt = f"""
            The following output should be valid JSON but has formatting issues. 
            Fix it to create valid JSON for weeks {batch_start} to {batch_end}, 
            with {days_per_week} days per week, and return ONLY the fixed JSON:
            
            {batch_roadmap.content}
            """
            fixed_json = await llm.ainvoke([HumanMessage(content=fix_prompt)])
            
            try:
                fixed_content = extract_json(fixed_json.content)
                batch_json = json.loads(fixed_content)
                complete_roadmap.update(batch_json)
            except Exception as e2:
                # Last resort: generate a simple placeholder structure
                for week_num in range(batch_start, batch_end + 1):
                    week_key = f"week{week_num}"
                    complete_roadmap[week_key] = generate_placeholder_week(days_per_week)
    
    # 4. Final validation to ensure complete roadmap
    # Check if any weeks are missing or empty
    for week_num in range(1, total_weeks + 1):
        week_key = f"week{week_num}"
        if week_key not in complete_roadmap or not complete_roadmap[week_key]:
            # Generate content for missing week
            missing_week_prompt = f"""
            Generate content for Week {week_num} of a {total_weeks}-week roadmap for learning {learning_goals}.
            
            Return ONLY a JSON object for week{week_num} with {days_per_week} days, where each day has 2-4 topic-resource pairs.
            
            Format example:
            {{
              "day1": [
                {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
                {{"topic": "Topic Name", "resource": "Resource Title - https://link"}},
                ...
              ],
              "day2": [...],
              ...
            }}
            """
            week_content = await llm.ainvoke([HumanMessage(content=missing_week_prompt)])
            
            try:
                week_json = json.loads(extract_json(week_content.content))
                complete_roadmap[week_key] = week_json
            except:
                complete_roadmap[week_key] = generate_placeholder_week(days_per_week)
    
    # Convert the completed roadmap back to JSON string
    final_json = json.dumps(complete_roadmap, indent=2)
    
    return final_json

def extract_weeks_from_plan(plan_text, start_week, end_week):
    """Extract specific weeks range from the complete plan."""
    result = []
    current_week = None
    lines = plan_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        week_match = re.match(r'Week\s+(\d+):', line)
        if week_match:
            week_num = int(week_match.group(1))
            if start_week <= week_num <= end_week:
                current_week = week_num
                result.append(line)
        elif current_week and start_week <= current_week <= end_week:
            result.append(line)
    
    return '\n'.join(result)

def filter_prompt_for_batch(batch_plan, start_week, end_week):
    """Create a focused prompt for just a batch of weeks."""
    return f"""
    Based on the following breakdown of topics for weeks {start_week} to {end_week}:

    {batch_plan}

    For **each topic**, find **one high-quality, specific and free resource** (like YouTube videos with specific video titles, blog posts with articles names, or free courses with course names).
    
    Guidelines for selecting resources:
    1. Each resource must be specific (e.g., "Introduction to Python Variables by CS Dojo" not just "CS Dojo")
    2. Include predominantly free resources from reputable sources
    3. Mix resource types (videos, articles, interactive tutorials, documentation)
    4. Resources should be current and relevant
    5. Match resource difficulty to the topic's position in the learning journey
    
    Use this format:
    Week X:
      Day Y:
        - Topic: Topic A
          Resource: Resource Title - https://...
        - Topic: Topic B
          Resource: Resource Title - https://...
        - Topic: Topic C
          Resource: Resource Title - https://...
    
    PROVIDE RESOURCES FOR ALL TOPICS in weeks {start_week} to {end_week}.
    """

def generate_placeholder_week(days_per_week):
    """Generate a placeholder structure for a week."""
    week = {}
    for day in range(1, days_per_week + 1):
        day_key = f"day{day}"
        week[day_key] = [
            {"topic": "Topic placeholder", "resource": "Resource placeholder - https://example.com"},
            {"topic": "Topic placeholder", "resource": "Resource placeholder - https://example.com"}
        ]
    return week

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