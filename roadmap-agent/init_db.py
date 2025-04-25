from app.database import engine, Base
from app.models import User, Roadmap, Progress
import logging

def init_db():
    """
    Initialize the database by creating all tables defined in the models.
    This function will create tables for User, Roadmap, and Progress models.
    """
    try:
        # Create all tables based on the imported models
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created successfully!")
        print("Database tables created successfully!")
        
        # List the tables that were created
        tables = list(Base.metadata.tables.keys())
        logging.info(f"Created tables: {', '.join(tables)}")
        print(f"Created tables: {', '.join(tables)}")
        
    except Exception as e:
        logging.error(f"Error creating database tables: {str(e)}")
        print(f"Error creating database tables: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the initialization
    init_db()