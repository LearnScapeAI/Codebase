from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DB URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Log to confirm the DB URL is read correctly
print(f"[DEBUG] Loaded DATABASE_URL: {DATABASE_URL}")

# Create SQLAlchemy engine
try:
    engine = create_engine(DATABASE_URL)
    print("[DEBUG] SQLAlchemy engine created successfully.")
except Exception as e:
    print(f"[ERROR] Failed to create engine: {e}")

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
print("[DEBUG] SessionLocal created.")

# Create base class for models
Base = declarative_base()
print("[DEBUG] Base declarative class created.")

# Dependency to get DB session
def get_db():
    print("[DEBUG] Opening new DB session.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        print("[DEBUG] DB session closed.")