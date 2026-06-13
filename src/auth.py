#src/auth.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, event
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import bcrypt

# Locate project root directory for the local SQLite database
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(PROJECT_ROOT, 'retinopathy.db')}")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# FIX 1: Enable Write-Ahead Logging (WAL) mode for local SQLite testing.
# This allows multiple Uvicorn worker processes on Hugging Face to safely read 
# and write to the exact same file simultaneously without desynchronizing or locking.
if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User Table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    scans = relationship("ScanHistory", back_populates="owner", cascade="all, delete-orphan")

# Scan History Table
class ScanHistory(Base):
    __tablename__ = "scan_histories"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    diagnosis_class = Column(Integer, nullable=False)  
    diagnosis_text = Column(String, nullable=False)   
    confidence = Column(Float, nullable=False)        
    scanned_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="scans")

# Create tables automatically if missing
Base.metadata.create_all(bind=engine)

# Modernized secure password routines using native bcrypt directly
def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_bytes.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        plain_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(plain_bytes, hashed_bytes)
    except Exception:
        return False

# FIX 2: Automatic Testing Seed Account
# If a Hugging Face container sleeps or restarts during your UI testing, this 
# ensures you have a master credential ready to go instantly without registering.
def seed_test_environment():
    db = SessionLocal()
    try:
        test_email = "test@example.com"
        exists = db.query(User).filter(User.email == test_email).first()
        if not exists:
            user = User(email=test_email, hashed_password=hash_password("test1234"))
            db.add(user)
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

seed_test_environment()

# DB Dependency injection session handler
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()