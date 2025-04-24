from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DB URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Check if DATABASE_URL is loaded properly
if DATABASE_URL is None:
    print("[ERROR] DATABASE_URL is not set in the environment variables.")
    raise ValueError("DATABASE_URL is not set in the environment variables.")
else:
    print(f"[DEBUG] Loaded DATABASE_URL: {DATABASE_URL}")

# Create SQLAlchemy engine
try:
    engine = create_engine(DATABASE_URL)
    print("[DEBUG] SQLAlchemy engine created successfully.")
except Exception as e:
    print(f"[ERROR] Failed to create engine: {e}")
    raise

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