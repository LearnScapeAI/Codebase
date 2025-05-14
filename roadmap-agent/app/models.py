from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, JSON, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid
from datetime import datetime

# Pydantic models for request validation
class RoadmapRequest(BaseModel):
    learning_goals: str
    months: int
    days_per_week: int
    hours_per_day: int

class UserCreate(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

# SQLAlchemy models for database
class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    name = Column(String)
    hashed_password = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationship with roadmaps
    roadmaps = relationship("Roadmap", back_populates="user")
    progress_items = relationship("Progress", back_populates="user")

class Roadmap(Base):
    __tablename__ = "roadmaps"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    learning_goals = Column(String)
    months = Column(Integer)
    days_per_week = Column(Integer)
    hours_per_day = Column(Float)
    content = Column(JSON)  # Stores the entire roadmap JSON
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationship with users and progress
    user = relationship("User", back_populates="roadmaps")
    progress_items = relationship("Progress", back_populates="roadmap")

class Progress(Base):
    __tablename__ = "progress"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))  # Add this line
    roadmap_id = Column(String, ForeignKey("roadmaps.id"))
    week_number = Column(Integer)
    day_number = Column(Integer)
    topic_index = Column(Integer)
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationship with roadmap
    roadmap = relationship("Roadmap", back_populates="progress_items")
    # Add user relationship
    user = relationship("User", back_populates="progress_items")

# Pydantic models for responses
class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime  # Changed from str to datetime
    
    class Config:
        orm_mode = True

class RoadmapResponse(BaseModel):
    id: str
    learning_goals: str
    months: int
    days_per_week: int
    hours_per_day: float
    content: dict
    created_at: datetime  # Changed from str to datetime
    
    class Config:
        orm_mode = True

class ProgressUpdate(BaseModel):
    week_number: int
    day_number: int
    topic_index: int
    completed: bool