# db_utils.py
import os
import datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text

# ------------------------------------------------------------
# Engine / Backend helpers
# ------------------------------------------------------------
def get_engine(uri: str):
    """
    Create a SQLAlchemy engine.
    - If given a local SQLite path (./... or *.db), ensure folder exists.
    - Otherwise, treat as full SQLAlchemy URI (e.g., Postgres).
    """
    if uri.startswith("./") or uri.endswith(".db"):
        # SQLite file path
        os.makedirs(os.path.dirname(uri), exist_ok=True)
        uri = "sqlite:///" + os.path.abspath(uri)
    return create_engine(uri, future=True)

def _backend_name(con_or_engine) -> str:
    """Return 'sqlite' or 'postgresql' (or other) from an engine/connection."""
    eng = con_or_engine.engine if hasattr(con_or_engine, "engine") else con_or_engine
    return eng.url.get_backend_name()

# ------------------------------------------------------------
# Schemas
# ------------------------------------------------------------
def init_actuals_raw_schema(engine):
    """
    Create the raw actuals schema that mirrors your source columns plus totals.
    Tables:
      - products(product_id, product)
      - stores(store_id, store)
      - actuals_raw (one row per date x store x product with raw fields)
    """
    create_products = """
    CREATE TABLE IF NOT EXISTS products (
      product_id INTEGER PRIMARY KEY,
      product TEXT UNIQUE NOT NULL
    )
    """
    create_stores = """
    CREATE TABLE IF NOT EXISTS stores (
      store_id INTEGER PRIMARY KEY,
      store TEXT UNIQUE NOT NULL
    )
    """
    create_actuals_raw = """
    CREATE TABLE IF NOT EXISTS actuals_raw (
      date DATE NOT NULL,
      store_id INTEGER NOT NULL,
      product_id INTEGER NOT NULL,
      category TEXT,

      START_OF_DAY_WHOLE REAL,
      MADE_TODAY_WHOLE REAL,
      AVAILABLE_TO_SELL_WHOLE REAL,
      OUT_THE_DOOR_WHOLE REAL,
      ORDERS_TODAY_WHOLE REAL,
      REMOVED_AS_DAY_OLD_WHOLE REAL,
      LEFT_WHOLE REAL,

      START_OF_DAY_HALF REAL,
      MADE_TODAY_HALF REAL,
      AVAILABLE_TO_SELL_HALF REAL,
      OUT_THE_DOOR_HALF REAL,
      ORDERS_TODAY_HALF REAL,
      REMOVED_AS_DAY_OLD_HALF REAL,
      LEFT_HALF REAL,

      TOTAL_OUT_THE_DOOR REAL,
      TOTAL_AVAILABLE_TO_SELL REAL,

      PRIMARY KEY (date, store_id, product_id),
      FOREIGN KEY (store_id) REFERENCES stores(store_id),
      FOREIGN KEY (product_id) REFERENCES products(product_id)
    )
    """
    create_idx = """
    CREATE INDEX IF NOT EXISTS idx_actuals_raw_prod_date
      ON actuals_raw(product_id, date)
    """

    with engine.begin() as con:
        con.execute(text(create_products))
        con.execute(text(create_stores))
        con.execute(text(create_actuals_raw))
        con.execute(text(create_idx))


def init_forecasts_schema(engine):
    """
    Create the forecasts schema (separate DB recommended).
    Tables:
      - products, stores (same dimension design)
      - forecasts_backtest: backtest rows (includes y)
      - forecasts_future: true future forecast rows (no y)
      - run_metrics: JSON metrics blob per run
    """
    create_products = """
    CREATE TABLE IF NOT EXISTS products (
      product_id INTEGER PRIMARY KEY,
      product TEXT UNIQUE NOT NULL
    )
    """
    create_stores = """
    CREATE TABLE IF NOT EXISTS stores (
      store_id INTEGER PRIMARY KEY,
      store TEXT UNIQUE NOT NULL
    )
    """
    create_backtest = """
    CREATE TABLE IF NOT EXISTS forecasts_backtest (
      run_ts TIMESTAMP NOT NULL,
      model_version TEXT,
      date DATE NOT NULL,
      store_id INTEGER NOT NULL,
      product_id INTEGER NOT NULL,
      y REAL,
      pred_prob REAL,
      pred_flag INTEGER,
      pred_qty_cond REAL,
      pred_two_stage REAL,
      PRIMARY KEY (run_ts, date, store_id, product_id)
    )
    """
    create_backtest_idx = """
    CREATE INDEX IF NOT EXISTS idx_fb_product_date
      ON forecasts_backtest(product_id, date)
    """
    create_future = """
    CREATE TABLE IF NOT EXISTS forecasts_future (
      run_ts TIMESTAMP NOT NULL,
      model_version TEXT,
      horizon INTEGER NOT NULL,
      date DATE NOT NULL,
      store_id INTEGER NOT NULL,
      product_id INTEGER NOT NULL,
      pred_prob REAL,
      pred_flag INTEGER,
      pred_qty_cond REAL,
      pred_two_stage REAL,
      PRIMARY KEY (run_ts, date, store_id, product_id)
    )
    """
    create_future_idx = """
    CREATE INDEX IF NOT EXISTS idx_ff_product_date
      ON forecasts_future(product_id, date)
    """
    create_metrics = """
    CREATE TABLE IF NOT EXISTS run_metrics (
      run_ts TIMESTAMP PRIMARY KEY,
      model_version TEXT,
      payload_json TEXT
    )
    """

    create_future_uq = """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_ff_date_store_product_h
    ON forecasts_future(date, store_id, product_id, horizon)
    """

    with engine.begin() as con:
        con.execute(text(create_products))
        con.execute(text(create_stores))
        con.execute(text(create_backtest))
        con.execute(text(create_backtest_idx))
        con.execute(text(create_future))
        con.execute(text(create_future_idx))
        con.execute(text(create_future_uq))
        con.execute(text(create_metrics))

# ------------------------------------------------------------
# Dimension upsert + mapping
# ------------------------------------------------------------
def upsert_dim(con, table: str, key_col: str, values):
    """
    Ensure the given natural keys exist in a dimension table and return a mapping.
    - Backend-aware UPSERT (SQLite vs Postgres)
    - Returns dict: { natural_key_value -> surrogate_id }
    Assumes PK column is named like '<singular>_id' (products->product_id, stores->store_id).
    """
    vals = sorted({v for v in values if v is not None})
    if not vals:
        return {}

    backend = _backend_name(con)
    if backend == "sqlite":
        insert_sql = f"INSERT OR IGNORE INTO {table} ({key_col}) VALUES (:v)"
    else:
        # Postgres and others that support ON CONFLICT DO NOTHING
        insert_sql = f"INSERT INTO {table} ({key_col}) VALUES (:v) ON CONFLICT ({key_col}) DO NOTHING"

    for v in vals:
        con.execute(text(insert_sql), {"v": v})

    id_col = f"{table[:-1]}_id"  # crude but matches our schema (products->product_id)
    rows = con.execute(text(f"SELECT {id_col}, {key_col} FROM {table}")).all()
    return {r._mapping[key_col]: r._mapping[id_col] for r in rows}


def map_dims(con, df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert any missing store/product keys and attach store_id/product_id to df.
    Expects df with columns: store, product
    Returns df with additional columns: store_id, product_id
    """
    df2 = df.copy()
    store_map = upsert_dim(con, "stores", "store", df2["store"].dropna().unique())
    prod_map  = upsert_dim(con, "products", "product", df2["product"].dropna().unique())
    df2["store_id"] = df2["store"].map(store_map)
    df2["product_id"] = df2["product"].map(prod_map)
    return df2


def ensure_dims_and_map(con, df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as map_dims, but keeps the entire input df intact and just adds id columns.
    Useful for forecast loaders where we have flexible columns.
    """
    df2 = df.copy()
    # tolerate missing store/product (but usually they should exist)
    if "store" not in df2.columns:
        df2["store"] = "store_all"
    if "product" not in df2.columns:
        raise ValueError("ensure_dims_and_map: 'product' column is required")

    store_map = upsert_dim(con, "stores", "store", df2["store"].dropna().unique())
    prod_map  = upsert_dim(con, "products", "product", df2["product"].dropna().unique())
    df2["store_id"] = df2["store"].map(store_map)
    df2["product_id"] = df2["product"].map(prod_map)
    return df2