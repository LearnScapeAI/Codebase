from app.utils.prompt_utils import generate_initial_prompt, filter_prompt, verify_prompt
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

async def generate_roadmap(learning_goals: str, months: int, days_per_week: int):
    # First LLM call to fetch resources
    prompt1 = generate_initial_prompt(learning_goals, months, days_per_week)
    raw_resources = await llm.ainvoke([HumanMessage(content=prompt1)])

    # Second LLM call to filter top 5 resources
    prompt2 = filter_prompt(raw_resources.content)
    top_resources = await llm.ainvoke([HumanMessage(content=prompt2)])

    # Third LLM call to verify resources and finalize roadmap
    prompt3 = verify_prompt(top_resources.content)
    final_roadmap = await llm.ainvoke([HumanMessage(content=prompt3)])

    return final_roadmap.content