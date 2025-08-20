import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


@dataclass
class Config:
    # API
    FPL_BASE_URL: str = "https://fantasy.premierleague.com/api/"
    FBREF_BASE_URL: str = "https://fbref.com/en/comps/9/"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///fpl_ai_2025_26.db")

    # Email
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    EMAIL_USER: str | None = os.getenv("EMAIL_USER")
    EMAIL_PASS: str | None = os.getenv("EMAIL_PASS")
    ALERT_EMAIL: str | None = os.getenv("ALERT_EMAIL")

    # ML
    MODEL_RETRAIN_INTERVAL: int = int(os.getenv("MODEL_RETRAIN_INTERVAL", 6))
    PREDICTION_CONFIDENCE_THRESHOLD: float = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", 0.7))

    # Season specifics
    SEASON: str = os.getenv("SEASON", "2025_26")
    AFCON_START_GW: int = int(os.getenv("AFCON_START_GW", 16))
    AFCON_END_GW: int = int(os.getenv("AFCON_END_GW", 21))
    CHIP_SET_1_DEADLINE_GW: int = int(os.getenv("CHIP_SET_1_DEADLINE_GW", 19))

    # Security
    MAX_API_REQUESTS_PER_HOUR: int = int(os.getenv("MAX_API_REQUESTS_PER_HOUR", 100))
    RATE_LIMIT_DELAY: float = float(os.getenv("RATE_LIMIT_DELAY", 1.0))
    DATA_RETENTION_DAYS: int = int(os.getenv("DATA_RETENTION_DAYS", 90))

    # FPL Team
    FPL_TEAM_ID: int = int(os.getenv("FPL_TEAM_ID", 0))
    FPL_EMAIL: str | None = os.getenv("FPL_EMAIL")
    FPL_PASSWORD: str | None = os.getenv("FPL_PASSWORD")

    # Optional APIs
    RAPID_API_KEY: str | None = os.getenv("RAPID_API_KEY")
    FOOTBALL_API_KEY: str | None = os.getenv("FOOTBALL_API_KEY")
    ODDSAPI_KEY: str | None = os.getenv("ODDSAPI_KEY")
    API_FOOTBALL_KEY: str | None = os.getenv("API_FOOTBALL_KEY")

    # Dev
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    ENABLE_WEB_INTERFACE: bool = os.getenv("ENABLE_WEB_INTERFACE", "False").lower() == "true"
    WEB_PORT: int = int(os.getenv("WEB_PORT", 8000))

    # Security token for public run trigger
    RUN_NOW_TOKEN: str | None = os.getenv("RUN_NOW_TOKEN")


config = Config()

