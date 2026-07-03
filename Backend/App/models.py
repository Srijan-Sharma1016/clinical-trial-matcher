# models.py
"""
SQLModel table definitions.
Responsibility: Define all database tables — nothing else.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import JSON, Text
from sqlmodel import SQLModel, Field, Relationship, Column

__all__ = [
    "PatientProfileTable",
    "TrialMatchTable",
]


class PatientProfileTable(SQLModel, table=True):
    """
    Stores extracted and normalized patient profiles.
    One patient profile can have many trial matches.
    """
    __tablename__ = "patient_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)

    age: Optional[int] = Field(default=None, nullable=True)
    gender: Optional[str] = Field(default=None, nullable=True)
    cancer_type: Optional[str] = Field(default=None, nullable=True)
    cancer_stage: Optional[str] = Field(default=None, nullable=True)

    biomarkers: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    previous_treatments: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    matches: List["TrialMatchTable"] = Relationship(
        back_populates="patient_profile"
    )


class TrialMatchTable(SQLModel, table=True):
    """
    Stores individual trial match results per patient profile.
    Each row = one trial evaluated for one patient.
    """
    __tablename__ = "trial_matches"

    id: Optional[int] = Field(default=None, primary_key=True)

    patient_profile_id: int = Field(
        foreign_key="patient_profiles.id",
        nullable=False,
        index=True,
    )

    nct_id: Optional[str] = Field(default=None, index=True)
    title: Optional[str] = Field(default=None, nullable=True)

    match_explanation: str = Field(
        sa_column=Column(Text, nullable=False)
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    patient_profile: Optional[PatientProfileTable] = Relationship(
        back_populates="matches"
    )
