def generate_initial_prompt(goals, months, days_per_week):
    return f"""
    You are an AI roadmap planner. Create a daily learning roadmap for "{goals}" spread over {months} month(s), assuming the user studies {days_per_week} day(s) per week.
    
    For each day, list 2–4 **topics** the learner should study (not resources). Don't include links here.
    
    Structure it like:
    Week 1:
      Day 1: Topic A, Topic B, Topic C
      Day 2: Topic D, Topic E
    ...
    """

def filter_prompt(day_topic_plan):
    return f"""
    Based on this day-wise breakdown of topics:

    {day_topic_plan}

    For each topic, find 1 best **free** resource (e.g., YouTube, blogs, courses).
    Output day-wise topic-resource mapping. No JSON formatting yet, just plain text like:
    
    Week 1:
      Day 1:
        - Topic: Topic A
          Resource: https://...
    """

def verify_prompt(resource_mapped_content):
    return f"""
    Review the following topic-resource roadmap. Validate all links. Make sure each link is working, accessible, and relevant to the topic.
    
    If any are broken or poor quality, replace them. Then return the **final roadmap in JSON**, structured as:
    {{
      "week1": {{
        "day1": [{{"topic": "...", "resource": "..."}}, ...],
        ...
      }},
      ...
    }}
    
    Content:
    {resource_mapped_content}
    """

def format_json_prompt(raw_text: str):
    return f"""
You previously generated this roadmap text:

{raw_text}

Please format it strictly as valid JSON with week and day structure:

{{
  "week1": {{
    "day1": "...",
    "day2": "..."
  }},
  ...
}}

Only return JSON.
"""