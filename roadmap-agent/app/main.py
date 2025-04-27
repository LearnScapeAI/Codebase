from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.roadmap import router as roadmap_router
from app.routes.auth import router as auth_router
from app.routes.user import router as user_router
from app.tasks.archive_task import start_scheduler
from app.database import engine, Base
from dotenv import load_dotenv
load_dotenv()

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LearnScape Roadmap Generator")

# Configure CORS
origins = [
    "http://localhost:3000",
    "https://learnscape-frontend.vercel.app",  # Add your frontend URL here
    "*"  # During development, you can allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(roadmap_router, prefix="/api")
app.include_router(auth_router, prefix="/api/auth")
app.include_router(user_router, prefix="/api/user")

@app.on_event("startup")
async def startup_event():
    # Start the scheduler for archiving cold data
    start_scheduler()

@app.get("/")
async def root():
    return {"message": "Welcome to LearnScape AI - Your personalized learning roadmap generator"}