"""
Secure configuration management for Tribrain
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables (fails silently if .env missing)
load_dotenv()

class SecurityError(Exception):
    """Raised when security configuration is invalid"""
    pass

class Config:
    """Centralized configuration with security validation"""
    
    # Database configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "tribrain")
    DB_USER = os.getenv("DB_USER", "tribrain_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    
    # API keys
    CRITIC_API_KEY = os.getenv("CRITIC_API_KEY", "")
    VLM_API_KEY = os.getenv("VLM_API_KEY", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    
    # Security configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "")
    MAX_SESSION_AGE = int(os.getenv("MAX_SESSION_AGE", "3600"))
    
    # Critical paths
    STORAGE_DIR = os.getenv("STORAGE_DIR", "storage")
    CALIBRATION_DB = os.getenv("CALIBRATION_DB", "storage/calibration.db")
    
    @classmethod
    def validate(cls) -> None:
        """Validate critical security configuration"""
        errors = []
        
        if not cls.DB_PASSWORD:
            errors.append("DB_PASSWORD environment variable is required")
        
        if not cls.SECRET_KEY:
            # Generate a secure default if not provided (for development only)
            import secrets
            cls.SECRET_KEY = secrets.token_hex(32)
            print("WARNING: Generated temporary SECRET_KEY for development. "
                  "Set SECRET_KEY environment variable for production.")
        
        # In production, all API keys should be set
        if os.getenv("ENV", "development") == "production":
            if not cls.CRITIC_API_KEY:
                errors.append("CRITIC_API_KEY required in production")
            if not cls.VLM_API_KEY:
                errors.append("VLM_API_KEY required in production")
            if not cls.LLM_API_KEY:
                errors.append("LLM_API_KEY required in production")
        
        if errors:
            raise SecurityError("\n".join(errors))
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """Get database configuration with security defaults"""
        return {
            "host": cls.DB_HOST,
            "database": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "connect_timeout": 10,
            # Add SSL configuration for production
            "sslmode": "require" if os.getenv("ENV") == "production" else "disable"
        }

# Validate configuration on import
Config.validate()