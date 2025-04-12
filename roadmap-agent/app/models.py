from pydantic import BaseModel

class RoadmapRequest(BaseModel):
    learning_goals: str
    months: int
    days_per_week: int
    hours_per_day: int