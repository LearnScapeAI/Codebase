def generate_initial_prompt(goals, months, days_per_week):
    total_weeks = months * 4  # Approximate weeks in months
    total_days = total_weeks * days_per_week
    
    return f"""
    You are an expert AI curriculum planner specializing in creating comprehensive learning roadmaps.

    Create a detailed daily roadmap for learning **{goals}** over a period of {months} month(s), assuming the user studies {days_per_week} day(s) per week.
    
    This equals approximately {total_weeks} weeks or {total_days} study days total.
    
    Guidelines:
    1. Each day MUST include **2 to 4 unique and related topics** that build on each other
    2. Start with foundational concepts and progress to advanced topics
    3. Include both theoretical knowledge and practical applications
    4. Ensure topics follow a logical progression from basics to mastery
    5. Consider industry best practices and in-demand skills for this field
    
    Use this exact format:
    Week 1:
      Day 1: Topic A, Topic B, Topic C
      Day 2: Topic D, Topic E, Topic F
    ...
    Week {total_weeks}:
      Day {days_per_week}: Topic X, Topic Y, Topic Z
    
    IMPORTANT: You MUST create a complete roadmap covering ALL {total_weeks} weeks in full detail.
    Make sure to include content for EVERY week from Week 1 to Week {total_weeks}.
    Do not abbreviate or skip any weeks.
    """


def filter_prompt(day_topic_plan):
    return f"""
    Based on the following breakdown of day-wise topics:

    {day_topic_plan}

    For **each topic**, find **one high-quality, specific and free resource** (like YouTube videos with specific video titles, blog posts with articles names, or free courses with course names).
    
    Guidelines for selecting resources:
    1. Each resource must be specific (e.g., "Introduction to Python Variables by CS Dojo" not just "CS Dojo")
    2. Include predominantly free resources from reputable sources
    3. Mix resource types (videos, articles, interactive tutorials, documentation)
    4. Resources should be current and relevant
    5. Match resource difficulty to the topic's position in the learning journey
    
    Use this format:
    Week 1:
      Day 1:
        - Topic: Topic A
          Resource: Resource Title - https://...
        - Topic: Topic B
          Resource: Resource Title - https://...
        - Topic: Topic C
          Resource: Resource Title - https://...
      
    Continue this pattern for ALL days across ALL weeks. Do not skip any topic or week.
    """


def verify_prompt(resource_mapped_content):
    return f"""
    You are a content validator responsible for ensuring educational roadmap quality.
    
    Review the following topic-resource roadmap JSON carefully:
    
    {resource_mapped_content}
    
    Validation tasks:
    1. Ensure all resources are relevant to their corresponding topics
    2. Verify that resource links follow the format "Resource Title - https://..." 
    3. Check that each day has 2-4 topics as required
    4. Make sure all weeks and days are included with no gaps
    5. Fix any JSON formatting issues (missing quotes, commas, brackets)
    
    If any resources appear generic or low-quality, replace them with specific, high-quality alternatives.
    
    Return the complete, validated roadmap in proper JSON format:
    {{
      "week1": {{
        "day1": [{{"topic": "...", "resource": "..."}}, ...],
        ...
      }},
      ...
    }}
    
    Your output must be valid, parseable JSON with no text before or after.
    """


def format_json_prompt(raw_text: str):
    return f"""
    You previously generated this roadmap text:

    {raw_text}

    Please format it strictly as valid JSON with week and day structure:

    {{
      "week1": {{
        "day1": [...],
        "day2": [...]
      }},
      ...
    }}

    Only return valid, parseable JSON with no text before or after.
    """