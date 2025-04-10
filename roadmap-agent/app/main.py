from fastapi import FastAPI
from app.routes.roadmap import router as roadmap_router
from app.tasks.archive_task import start_scheduler

app = FastAPI(title="LearnScape Roadmap Generator")

app.include_router(roadmap_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    # Start the scheduler for archiving cold data
    start_scheduler()