def generate_initial_prompt(goals, months, days_per_week):
    total_weeks = months * 4  # Approximate weeks in months
    total_days = total_weeks * days_per_week
    
    return f"""
    You are an expert AI curriculum planner specializing in comprehensive learning roadmaps.

    Create a detailed daily roadmap for learning **{goals}** over a period of {months} month(s), studying {days_per_week} day(s) per week.
    
    This equals {total_weeks} weeks or {total_days} study days total.
    
    INSTRUCTIONS:
    1. Each day MUST include 2-4 unique, related topics
    2. Start with fundamentals and progress to advanced concepts
    3. Include both theory and practical applications
    4. Ensure logical progression from basics to expertise
    5. IMPORTANT: For EVERY week, provide content for ALL {days_per_week} days!
    
    Use this exact format:
    Week 1:
      Day 1: Topic A, Topic B, Topic C
      Day 2: Topic D, Topic E, Topic F
      ...
      Day {days_per_week}: Topic X, Topic Y, Topic Z
    Week 2:
      Day 1: ...
      ...
    
    Create a complete roadmap covering ALL {total_weeks} weeks with ALL {days_per_week} days in each week.
    Do not abbreviate weeks or days - be explicit and detailed for every single day.
    """

def filter_prompt(day_topic_plan):
    return f"""
    Based on the following day-wise topics:

    {day_topic_plan}

    For **each topic**, find **one high-quality, specific and free resource**.
    
    INSTRUCTIONS FOR RESOURCES:
    1. Each resource must be specific
    2. Include predominantly free resources from reputable sources
    3. Mix resource types (videos, articles, interactive tutorials, documentation)
    4. Resources should match the topic's difficulty level
    
    Use this format:
    Week 1:
      Day 1:
        - Topic: Topic A
          Resource: Resource Title - https://...
        - Topic: Topic B
          Resource: Resource Title - https://...
      Day 2:
        - Topic: Topic C
          Resource: Resource Title - https://...
        ...
      
    DO NOT SKIP ANY DAYS. If topics for certain days are missing in the input, create appropriate topics and resources for those days.
    """

def verify_prompt(resource_mapped_content):
    return f"""
    You are a content validator responsible for ensuring educational roadmap quality.
    
    Review this topic-resource roadmap JSON:
    
    {resource_mapped_content}
    
    Validation requirements:
    1. Each resource must match its topic
    2. Each resource must include both title and URL in format "Title - URL"
    3. Each day must have 2-4 topics
    4. All days must be included with no missing days
    5. JSON must be properly formatted with no errors
    
    Return the complete, validated roadmap as valid JSON:
    {{
      "week1": {{
        "day1": [{{"topic": "...", "resource": "..."}}, ...],
        "day2": [...],
        ...
      }},
      ...
    }}
    
    Output must be valid JSON only - no explanations or text before/after.
    """

def format_json_prompt(raw_text: str):
    return f"""
    Fix any formatting issues in this roadmap JSON:

    {raw_text}

    Format it as valid JSON with this structure:
    {{
      "week1": {{
        "day1": [...],
        "day2": [...]
      }},
      ...
    }}

    Return ONLY valid, parseable JSON with no explanations.
    """