# load_actuals_raw.py
import re, glob, yaml
import pandas as pd, numpy as np
from sqlalchemy import text
from db_utils import get_engine, init_actuals_raw_schema, map_dims

def _try_read(path):
    encodings = ["utf-8","utf-8-sig","latin-1"]
    seps = [",",";","\t","|"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    raise RuntimeError(f"Could not parse {path}")

def _find_col(df, name):
    # exact → substring match (case-insensitive)
    for col in df.columns:
        if col.strip().lower() == name.strip().lower():
            return col
    for col in df.columns:
        if name.strip().lower() in col.strip().lower():
            return col
    return None

RAW_COLS = [
 "Category","Product",
 "START_OF_DAY_WHOLE","MADE_TODAY_WHOLE","AVAILABLE_TO_SELL_WHOLE","OUT_THE_DOOR_WHOLE","ORDERS_TODAY_WHOLE","REMOVED_AS_DAY_OLD_WHOLE","LEFT_WHOLE",
 "START_OF_DAY_HALF","MADE_TODAY_HALF","AVAILABLE_TO_SELL_HALF","OUT_THE_DOOR_HALF","ORDERS_TODAY_HALF","REMOVED_AS_DAY_OLD_HALF","LEFT_HALF"
]

def normalize_product_name(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9\s\-\/]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _ensure_product_column(con):
    """
    Make sure actuals_raw has a TEXT column named `product`.
    SQLite: PRAGMA + ALTER TABLE if missing.
    Postgres: ALTER TABLE ... ADD COLUMN IF NOT EXISTS.
    Others: best-effort ADD COLUMN in try/except.
    """
    backend = con.engine.url.get_backend_name()
    if backend == "sqlite":
        cols = con.execute(text("PRAGMA table_info(actuals_raw)")).fetchall()
        have = {c[1] for c in cols}  # column name at index 1
        if "product" not in have:
            con.execute(text("ALTER TABLE actuals_raw ADD COLUMN product TEXT"))
        return

    if backend == "postgresql":
        con.execute(text("ALTER TABLE actuals_raw ADD COLUMN IF NOT EXISTS product TEXT"))
        return

    # Fallback: try once; ignore error if exists
    try:
        con.execute(text("ALTER TABLE actuals_raw ADD COLUMN product TEXT"))
    except Exception:
        pass

def _ensure_unique_key(con):
    """
    Enforce uniqueness on (date, store_id, product_id) so duplicates cannot be inserted.
    """
    backend = con.engine.url.get_backend_name()

    if backend == "sqlite":
        # SQLite: create a unique index (safe with IF NOT EXISTS)
        con.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_actuals_raw_dsp
            ON actuals_raw(date, store_id, product_id)
        """))
        return

    if backend == "postgresql":
        # Postgres: prefer a table constraint (no-op if already there)
        # Using a DO block would need superuser; this simple IF NOT EXISTS is enough on recent PG.
        con.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM   pg_constraint
                    WHERE  conrelid = 'actuals_raw'::regclass
                    AND    conname = 'ux_actuals_raw_dsp'
                ) THEN
                    ALTER TABLE actuals_raw
                    ADD CONSTRAINT ux_actuals_raw_dsp UNIQUE (date, store_id, product_id);
                END IF;
            END$$;
        """))
        return

    # Fallback: attempt to create a unique index
    try:
        con.execute(text("""
            CREATE UNIQUE INDEX ux_actuals_raw_dsp
            ON actuals_raw(date, store_id, product_id)
        """))
    except Exception:
        pass

def _drop_in_batch_dupes(df):
    """
    Remove duplicate rows within the current DataFrame batch based on
    (date, store_id, product_id). Keep the last occurrence.
    """
    keys = ["date", "store_id", "product_id"]
    if not all(k in df.columns for k in keys):
        return df
    # Avoid pandas warning with None by temporarily filling for dedupe purpose
    tmp = df.copy()
    tmp["__date_str__"] = tmp["date"].astype(str)
    tmp = tmp.drop_duplicates(subset=["__date_str__", "store_id", "product_id"], keep="last")
    tmp = tmp.drop(columns=["__date_str__"])
    return tmp

def main():
    cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    files = sorted(glob.glob(cfg["data_glob"]))
    if not files:
        raise FileNotFoundError(f"No files matched: {cfg['data_glob']}")

    eng = get_engine(yaml.safe_load(open("db_config.yaml","r",encoding="utf-8"))["actuals_db"])
    init_actuals_raw_schema(eng)

    with eng.begin() as con:
        # ensure we physically have a `product` column in actuals_raw
        _ensure_product_column(con)
        # ensure uniqueness so true duplicates are blocked at DB level
        _ensure_unique_key(con)

        backend = con.engine.url.get_backend_name()

        for p in files:
            df = _try_read(p)
            df.columns = [c.strip() for c in df.columns]

            # Date
            date_col = _find_col(df, "Date") or _find_col(df, "date")
            if not date_col:
                print(f"Skipping {p}: no Date column")
                continue

            df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
            df = df.dropna(subset=["date"]).copy()

            # --- Ensure SQLite/Postgres-friendly types ---
            # 1) dates as python 'date'
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

            # 2) convert NaN -> None for DB NULLs
            df = df.where(pd.notna(df), None)

            # Numerics → floats
            num_cols = [
                "START_OF_DAY_WHOLE","MADE_TODAY_WHOLE","AVAILABLE_TO_SELL_WHOLE","OUT_THE_DOOR_WHOLE","ORDERS_TODAY_WHOLE","REMOVED_AS_DAY_OLD_WHOLE","LEFT_WHOLE",
                "START_OF_DAY_HALF","MADE_TODAY_HALF","AVAILABLE_TO_SELL_HALF","OUT_THE_DOOR_HALF","ORDERS_TODAY_HALF","REMOVED_AS_DAY_OLD_HALF","LEFT_HALF",
                "TOTAL_OUT_THE_DOOR","TOTAL_AVAILABLE_TO_SELL"
            ]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(float).where(pd.notna(df[c]), None)

            # Store (single store setup)
            store = "store_all"
            df["store"] = store

            # Product / Category
            prod_col = _find_col(df, "Product")
            if not prod_col:
                print(f"Skipping {p}: no Product column")
                continue
            df["product"] = df[prod_col].apply(normalize_product_name)

            cat_col = _find_col(df, "Category")
            df["category"] = df[cat_col] if cat_col else None

            # Bring over raw numeric fields if present; else NA
            for c in RAW_COLS:
                if c in ["Category","Product"]:
                    continue
                col = _find_col(df, c)  # tolerant to header variants
                df[c] = pd.to_numeric(df[col], errors="coerce") if col else np.nan

            # Totals (HALF counts as 1.0 here)
            df["TOTAL_OUT_THE_DOOR"] = (df["OUT_THE_DOOR_WHOLE"].fillna(0) + df["OUT_THE_DOOR_HALF"].fillna(0))
            df["TOTAL_AVAILABLE_TO_SELL"] = (df["AVAILABLE_TO_SELL_WHOLE"].fillna(0) + df["AVAILABLE_TO_SELL_HALF"].fillna(0))

            # Map to IDs
            dim_df = map_dims(con, df[["store","product"]].drop_duplicates())
            df = df.merge(dim_df[["store","product","store_id","product_id"]], on=["store","product"], how="left")

            # Filter out rows missing keys (prevents dup leakage via NULL semantics)
            before = len(df)
            df = df.dropna(subset=["date", "store_id", "product_id"]).copy()
            after = len(df)
            if after < before:
                print(f"Skipping {before - after} rows with null keys in {p}")

            # Build records including *product* text now
            keep = ["date","store_id","product_id","product","category"] + [
                "START_OF_DAY_WHOLE","MADE_TODAY_WHOLE","AVAILABLE_TO_SELL_WHOLE","OUT_THE_DOOR_WHOLE","ORDERS_TODAY_WHOLE","REMOVED_AS_DAY_OLD_WHOLE","LEFT_WHOLE",
                "START_OF_DAY_HALF","MADE_TODAY_HALF","AVAILABLE_TO_SELL_HALF","OUT_THE_DOOR_HALF","ORDERS_TODAY_HALF","REMOVED_AS_DAY_OLD_HALF","LEFT_HALF",
                "TOTAL_OUT_THE_DOOR","TOTAL_AVAILABLE_TO_SELL"
            ]
            # ensure missing numeric cols are present as None
            for c in keep:
                if c not in df.columns:
                    df[c] = None

            # In-batch de-duplication on the natural key (date, store_id, product_id)
            df = _drop_in_batch_dupes(df)

            # convert to native Python types where needed
            df = df.where(pd.notna(df), None)
            recs = df[keep].to_dict(orient="records")

            # Use consistent UPSERT on both backends
            sql_upsert = """
            INSERT INTO actuals_raw
              (date, store_id, product_id, product, category,
               START_OF_DAY_WHOLE, MADE_TODAY_WHOLE, AVAILABLE_TO_SELL_WHOLE, OUT_THE_DOOR_WHOLE, ORDERS_TODAY_WHOLE, REMOVED_AS_DAY_OLD_WHOLE, LEFT_WHOLE,
               START_OF_DAY_HALF, MADE_TODAY_HALF, AVAILABLE_TO_SELL_HALF, OUT_THE_DOOR_HALF, ORDERS_TODAY_HALF, REMOVED_AS_DAY_OLD_HALF, LEFT_HALF,
               TOTAL_OUT_THE_DOOR, TOTAL_AVAILABLE_TO_SELL)
            VALUES
              (:date, :store_id, :product_id, :product, :category,
               :START_OF_DAY_WHOLE, :MADE_TODAY_WHOLE, :AVAILABLE_TO_SELL_WHOLE, :OUT_THE_DOOR_WHOLE, :ORDERS_TODAY_WHOLE, :REMOVED_AS_DAY_OLD_WHOLE, :LEFT_WHOLE,
               :START_OF_DAY_HALF, :MADE_TODAY_HALF, :AVAILABLE_TO_SELL_HALF, :OUT_THE_DOOR_HALF, :ORDERS_TODAY_HALF, :REMOVED_AS_DAY_OLD_HALF, :LEFT_HALF,
               :TOTAL_OUT_THE_DOOR, :TOTAL_AVAILABLE_TO_SELL)
            ON CONFLICT (date, store_id, product_id)
            DO UPDATE SET
               product = EXCLUDED.product,
               category = EXCLUDED.category,
               START_OF_DAY_WHOLE = EXCLUDED.START_OF_DAY_WHOLE,
               MADE_TODAY_WHOLE = EXCLUDED.MADE_TODAY_WHOLE,
               AVAILABLE_TO_SELL_WHOLE = EXCLUDED.AVAILABLE_TO_SELL_WHOLE,
               OUT_THE_DOOR_WHOLE = EXCLUDED.OUT_THE_DOOR_WHOLE,
               ORDERS_TODAY_WHOLE = EXCLUDED.ORDERS_TODAY_WHOLE,
               REMOVED_AS_DAY_OLD_WHOLE = EXCLUDED.REMOVED_AS_DAY_OLD_WHOLE,
               LEFT_WHOLE = EXCLUDED.LEFT_WHOLE,
               START_OF_DAY_HALF = EXCLUDED.START_OF_DAY_HALF,
               MADE_TODAY_HALF = EXCLUDED.MADE_TODAY_HALF,
               AVAILABLE_TO_SELL_HALF = EXCLUDED.AVAILABLE_TO_SELL_HALF,
               OUT_THE_DOOR_HALF = EXCLUDED.OUT_THE_DOOR_HALF,
               ORDERS_TODAY_HALF = EXCLUDED.ORDERS_TODAY_HALF,
               REMOVED_AS_DAY_OLD_HALF = EXCLUDED.REMOVED_AS_DAY_OLD_HALF,
               LEFT_HALF = EXCLUDED.LEFT_HALF,
               TOTAL_OUT_THE_DOOR = EXCLUDED.TOTAL_OUT_THE_DOOR,
               TOTAL_AVAILABLE_TO_SELL = EXCLUDED.TOTAL_AVAILABLE_TO_SELL
            """
            # SQLite supports ON CONFLICT(..) DO UPDATE in modern versions; if very old, the unique index still blocks dup inserts.
            con.execute(text(sql_upsert), recs)

    print("Loaded raw actuals into DB with duplicate protection (unique key + upsert).")

if __name__ == "__main__":
    main()