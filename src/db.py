from __future__ import annotations

from dataclasses import dataclass
from typing import Generator
from sqlalchemy import create_engine, Integer, String, DECIMAL, Boolean, TIMESTAMP, Column, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from .config import config


Base = declarative_base()


def get_engine():
    return create_engine(config.DATABASE_URL, future=True)


SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    position = Column(String(3))
    team = Column(String(50))
    price = Column(DECIMAL(4, 1))
    availability = Column(DECIMAL(3, 1), nullable=True)
    afcon_risk = Column(Boolean, default=False)
    last_updated = Column(TIMESTAMP, nullable=True)


class GameweekData(Base):
    __tablename__ = "gameweek_data"
    player_id = Column(Integer)
    gameweek = Column(Integer)
    points = Column(Integer, nullable=True)
    minutes = Column(Integer, nullable=True)
    goals = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    xg = Column(DECIMAL(4, 2), nullable=True)
    xa = Column(DECIMAL(4, 2), nullable=True)
    cbit_count = Column(Integer, nullable=True)
    cbirt_count = Column(Integer, nullable=True)
    defensive_contribution_points = Column(Integer, default=0)
    fixture_difficulty = Column(Integer, nullable=True)
    __table_args__ = (PrimaryKeyConstraint("player_id", "gameweek", name="pk_gameweek_data"),)


class Prediction(Base):
    __tablename__ = "predictions"
    player_id = Column(Integer, primary_key=True)
    gameweek = Column(Integer, primary_key=True)
    predicted_points = Column(DECIMAL(5, 2))
    confidence = Column(DECIMAL(3, 2))
    includes_defensive_contributions = Column(Boolean, default=True)
    afcon_adjusted = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP)


class ChipUsageLog(Base):
    __tablename__ = "chip_usage_log"
    gameweek = Column(Integer, primary_key=True)
    chip_type = Column(String(20))
    chip_set = Column(Integer)  # 1 or 2
    points_impact = Column(DECIMAL(6, 2))
    success_rating = Column(Integer)


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)

