from sqlmodel import Session
from sqlalchemy import text

from database import engine


def reset_tables():
    statements = [
        "DROP TABLE IF EXISTS trial_match_results CASCADE;",
        "DROP TABLE IF EXISTS trial_match_runs CASCADE;",
        "DROP TABLE IF EXISTS patient_profiles CASCADE;",
    ]

    with Session(engine) as session:
        for stmt in statements:
            session.exec(text(stmt))
        session.commit()

    print("Old tables dropped successfully.")


if __name__ == "__main__":
    reset_tables()
