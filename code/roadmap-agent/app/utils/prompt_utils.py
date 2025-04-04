def generate_initial_prompt(learning_goals: str, months: int, days_per_week: int):
    return f"""
    Create a detailed learning roadmap for the topic: {learning_goals}.
    Timeframe: {months} months
    Study commitment: {days_per_week} days per week.
    Break it into weekly and daily tasks. For each day, suggest 3-5 free online resources.
    Output in format:
    Week 1:
      Day 1:
        - Topic: ...
        - Resources:
            1. Resource A (link)
            2. Resource B (link)
    """

def filter_prompt(raw_resources: str):
    return f"""
    Given the roadmap and resources below, filter and retain only the top 5 free high-quality resources per day:
    {raw_resources}
    """

def verify_prompt(top_resources: str):
    return f"""
    Review the selected resources below, verify their availability and clarity, and format them into a clean final roadmap:
    {top_resources}
    Output should be:
    - Clear daily tasks
    - Verified links
    - Well-structured sections per week
    """