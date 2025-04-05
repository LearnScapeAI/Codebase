from app.utils.prompt_utils import filter_prompt, verify_prompt, format_json_prompt, generate_initial_prompt
from app.utils.resource_fetcher import fetch_real_resources
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

async def generate_roadmap(learning_goals: str, months: int, days_per_week: int):
    # 1. First LLM call — break down roadmap into weeks -> days -> topics
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week)
    day_topic_plan = await llm.ainvoke([HumanMessage(content=prompt1)])

    # 2. Second LLM call — find resources per topic
    prompt2 = filter_prompt(day_topic_plan.content)
    topic_resources = await llm.ainvoke([HumanMessage(content=prompt2)])

    # 3. Third LLM call — structure the roadmap into JSON format (week -> day -> topic -> resource)
    structure_prompt = f"""
    Based on the following day-wise topic-resource pairs, generate a JSON with this exact structure:
    {{
      "week1": {{
        "day1": [{{"topic": "...", "resource": "..."}}, ...],
        "day2": [...],
        ...
      }},
      ...
    }}
    Ensure each day has 2–4 topics, and every topic has a working link.

    Content:
    {topic_resources.content}
    """

    structured_roadmap = await llm.ainvoke([HumanMessage(content=structure_prompt)])

    # 4. Final validation call — cross-check for dead links or bad resources
    prompt3 = verify_prompt(structured_roadmap.content)
    validated_roadmap = await llm.ainvoke([HumanMessage(content=prompt3)])

    return validated_roadmap.content