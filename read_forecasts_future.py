import sqlite3
import pandas as pd
from sqlalchemy import create_engine, inspect

DB_PATH = "db/bread_forecasts.db"  # adjust if needed

def main():
    # Connect using SQLAlchemy for flexibility
    engine = create_engine(f"sqlite:///{DB_PATH}")

    # Inspect tables
    insp = inspect(engine)
    tables = insp.get_table_names()
    print(f"\nTables in database: {tables}")

    if "forecasts_future" not in tables:
        print("❌ Table 'forecasts_future' not found in database.")
        return

    # Load the table into a DataFrame
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM forecasts_future", conn)

    print(f"\n✅ Loaded {len(df)} rows from forecasts_future\n")
    print(df)  # Show first 20 rows

    # Optionally, print summary
    print("\nColumns:")
    print(list(df.columns))

    df.to_csv("forecasts_future.csv", index=False)

    print("\nDate range:")
    print(f"{df['date'].min()} → {df['date'].max()}")

if __name__ == "__main__":
    main()
