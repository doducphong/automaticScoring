from pydantic import BaseSettings, PostgresDsn, validator
from typing import Optional, Dict, Any
from pathlib import Path
import os


class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY") 
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    PROJECT_NAME: str = "Automatic Document Scoring API"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]
    
    # Database settings
    # By default, use SQLite for development, but allow override via env var
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")
    
    # SQLite settings
    SQLITE_DB_PATH: str = "sqlite:///./app.db"
    
    # PostgreSQL settings
    POSTGRES_SERVER: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    
    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if values.get("DB_TYPE") == "postgres":
            if isinstance(v, str):
                return v
            return PostgresDsn.build(
                scheme="postgresql",
                user=values.get("POSTGRES_USER"),
                password=values.get("POSTGRES_PASSWORD"),
                host=values.get("POSTGRES_SERVER"),
                path=f"/{values.get('POSTGRES_DB') or ''}",
            )
        # Default to SQLite
        return values.get("SQLITE_DB_PATH")
    
    # File upload settings
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: list[str] = ["docx", "xlsx"]
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB max file size
    
    # Cloudinary settings
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET")
    
    # Scoring settings
    SIMILARITY_THRESHOLD: float = 0.7
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

# Create uploads directory if it doesn't exist
upload_path = Path(settings.UPLOAD_DIR)
upload_path.mkdir(exist_ok=True)
