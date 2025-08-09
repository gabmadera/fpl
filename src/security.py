import os
import logging
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SecurityManager:
    def __init__(self) -> None:
        self._setup_secure_logging()
        self.session = self._create_secure_session()
        self.request_count = 0
        self.last_request_reset = datetime.now()

    def _setup_secure_logging(self) -> None:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/fpl_ai.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("FPL_AI_Security")

    def _create_secure_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {
                "User-Agent": "FPL-AI-Personal-Bot/1.0 (Educational Use)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )
        return session

    def validate_environment(self) -> bool:
        required_vars = ["FPL_TEAM_ID", "FPL_EMAIL", "EMAIL_USER", "ALERT_EMAIL"]
        missing: list[str] = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing.append(var)
            elif var == "FPL_TEAM_ID":
                try:
                    team_id = int(value)
                    if team_id <= 0 or team_id > 20000000:
                        missing.append(f"{var} (invalid format)")
                except ValueError:
                    missing.append(f"{var} (not a number)")

        if missing:
            self.logger.error("Missing/invalid environment variables: %s", missing)
            return False

        self.logger.info("All required environment variables validated")
        return True

    def rate_limit_check(self) -> bool:
        max_requests = int(os.getenv("MAX_API_REQUESTS_PER_HOUR", 100))
        if datetime.now() - self.last_request_reset > timedelta(hours=1):
            self.request_count = 0
            self.last_request_reset = datetime.now()
        if self.request_count >= max_requests:
            self.logger.warning("Rate limit reached, delaying request")
            return False
        self.request_count += 1
        return True

