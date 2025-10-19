# load_forecasts.py
import json, yaml, pandas as pd, datetime as dt
from sqlalchemy import text
from db_utils import get_engine, init_forecasts_schema, ensure_dims_and_map

def _safe_read(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None

def _to_native_dates(df, col="date"):
    """Ensure pandas Timestamps -> python date for SQLite binding."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        df = df.dropna(subset=[col]).copy()
    return df

def _coerce_types_for_db(df, numeric_cols=None, int_cols=None):
    """Convert numeric/int fields cleanly for DB binding."""
    numeric_cols = numeric_cols or []
    int_cols = int_cols or []

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].where(pd.notna(df[c]), None)
            df[c] = df[c].apply(lambda x: int(x) if x is not None else None)

    df = df.where(pd.notna(df), None)
    return df

def _drop_in_batch_dupes(df, key_cols=("date","store_id","product_id")):
    """Remove duplicate rows within the DataFrame based on key_cols; keep last occurrence."""
    if not all(k in df.columns for k in key_cols):
        return df
    tmp = df.copy()
    if "date" in key_cols:
        tmp["__date_str__"] = tmp["date"].astype(str)
        subset = [("__date_str__" if k == "date" else k) for k in key_cols]
        tmp = tmp.drop_duplicates(subset=subset, keep="last").drop(columns=["__date_str__"])
        return tmp
    return df.drop_duplicates(subset=list(key_cols), keep="last")

def _ensure_unique_keys(con):
    """Add unique constraints (date, store_id, product_id) for both forecast tables."""
    backend = con.engine.url.get_backend_name()
    if backend == "sqlite":
        con.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_forecasts_backtest_dsp
            ON forecasts_backtest(date, store_id, product_id)
        """))
        con.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_forecasts_future_dsp
            ON forecasts_future(date, store_id, product_id)
        """))
        return
    if backend == "postgresql":
        con.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'forecasts_backtest'::regclass
                  AND conname = 'ux_forecasts_backtest_dsp'
            ) THEN
                ALTER TABLE forecasts_backtest
                ADD CONSTRAINT ux_forecasts_backtest_dsp UNIQUE (date, store_id, product_id);
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'forecasts_future'::regclass
                  AND conname = 'ux_forecasts_future_dsp'
            ) THEN
                ALTER TABLE forecasts_future
                ADD CONSTRAINT ux_forecasts_future_dsp UNIQUE (date, store_id, product_id);
            END IF;
        END$$;
        """))
        return
    try:
        con.execute(text("CREATE UNIQUE INDEX ux_forecasts_backtest_dsp ON forecasts_backtest(date, store_id, product_id)"))
    except Exception:
        pass
    try:
        con.execute(text("CREATE UNIQUE INDEX ux_forecasts_future_dsp ON forecasts_future(date, store_id, product_id)"))
    except Exception:
        pass

def _ensure_product_column(con):
    """Ensure a TEXT column `product` exists in both forecasts tables."""
    backend = con.engine.url.get_backend_name()
    tables = ["forecasts_backtest", "forecasts_future"]

    for tbl in tables:
        if backend == "sqlite":
            cols = con.execute(text(f"PRAGMA table_info({tbl})")).fetchall()
            have = {c[1] for c in cols}
            if "product" not in have:
                con.execute(text(f"ALTER TABLE {tbl} ADD COLUMN product TEXT"))
        elif backend == "postgresql":
            con.execute(text(f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS product TEXT"))
        else:
            try:
                con.execute(text(f"ALTER TABLE {tbl} ADD COLUMN product TEXT"))
            except Exception:
                pass

def main():
    cfg = yaml.safe_load(open("db_config.yaml", "r", encoding="utf-8"))
    engine = get_engine(cfg["forecasts_db"])
    init_forecasts_schema(engine)

    run_ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    model_version = "lightgbm_two_stage_v1"

    with engine.begin() as con:
        _ensure_unique_keys(con)
        _ensure_product_column(con)

        # --- Backtest predictions ---
        back = _safe_read("outputs/predictions_two_stage_test.csv", parse_dates=["date"])
        if back is not None and len(back):
            need = ["date","store","product","y","pred_prob","pred_flag","pred_qty_cond","pred_two_stage"]
            missing = [c for c in need if c not in back.columns]
            if missing:
                raise ValueError(f"Backtest CSV missing columns: {missing}")

            back = _to_native_dates(back, "date")
            back = _coerce_types_for_db(
                back,
                numeric_cols=["y","pred_prob","pred_qty_cond","pred_two_stage"],
                int_cols=["pred_flag"],
            )

            fact = ensure_dims_and_map(con, back[["date","store","product","y","pred_prob","pred_flag","pred_qty_cond","pred_two_stage"]])
            fact["run_ts"] = run_ts
            fact["model_version"] = model_version

            before = len(fact)
            fact = fact.dropna(subset=["date","store_id","product_id"]).copy()
            if len(fact) < before:
                print(f"Backtest: skipped {before - len(fact)} rows with null keys")
            fact = _drop_in_batch_dupes(fact)

            sql_backtest_upsert = """
            INSERT INTO forecasts_backtest
              (run_ts, model_version, date, store_id, product_id, product,
               y, pred_prob, pred_flag, pred_qty_cond, pred_two_stage)
            VALUES (:run_ts, :model_version, :date, :store_id, :product_id, :product,
                    :y, :pred_prob, :pred_flag, :pred_qty_cond, :pred_two_stage)
            ON CONFLICT (date, store_id, product_id)
            DO UPDATE SET
              run_ts        = EXCLUDED.run_ts,
              model_version = EXCLUDED.model_version,
              product       = EXCLUDED.product,
              y             = EXCLUDED.y,
              pred_prob     = EXCLUDED.pred_prob,
              pred_flag     = EXCLUDED.pred_flag,
              pred_qty_cond = EXCLUDED.pred_qty_cond,
              pred_two_stage= EXCLUDED.pred_two_stage
            """
            con.execute(text(sql_backtest_upsert), fact.to_dict(orient="records"))

        # --- Future multi-step predictions ---
        for path in [
            "outputs/predictions_next_7_days.csv",
            "outputs/predictions_next_14_days.csv",
            "outputs/predictions_next_28_days.csv",
        ]:
            fut = _safe_read(path, parse_dates=["date"])
            if fut is None or len(fut) == 0:
                continue

            fut = _to_native_dates(fut, "date")

            try:
                H = int(path.split("_next_")[1].split("_days")[0])
            except Exception:
                H = None

            base_cols = ["date","store","product"]
            pred_cols = [c for c in ["pred_prob","pred_flag","pred_qty_cond","pred_two_stage"] if c in fut.columns]
            fut = fut[base_cols + pred_cols]

            num_cols = [c for c in ["pred_prob","pred_qty_cond","pred_two_stage"] if c in fut.columns]
            int_cols = [c for c in ["pred_flag"] if c in fut.columns]
            fut = _coerce_types_for_db(fut, numeric_cols=num_cols, int_cols=int_cols)

            fact = ensure_dims_and_map(con, fut.assign(y=None))
            fact["run_ts"] = run_ts
            fact["model_version"] = model_version
            fact["horizon"] = H

            before = len(fact)
            fact = fact.dropna(subset=["date","store_id","product_id"]).copy()
            if len(fact) < before:
                print(f"Future{H}: skipped {before - len(fact)} rows with null keys")
            fact = _drop_in_batch_dupes(fact)

            sql_future_upsert = """
            INSERT INTO forecasts_future
              (run_ts, model_version, horizon, date, store_id, product_id, product,
               pred_prob, pred_flag, pred_qty_cond, pred_two_stage)
            VALUES (:run_ts, :model_version, :horizon, :date, :store_id, :product_id, :product,
                    :pred_prob, :pred_flag, :pred_qty_cond, :pred_two_stage)
            ON CONFLICT (date, store_id, product_id)
            DO UPDATE SET
              run_ts        = EXCLUDED.run_ts,
              model_version = EXCLUDED.model_version,
              product       = EXCLUDED.product,
              horizon       = EXCLUDED.horizon,
              pred_prob     = EXCLUDED.pred_prob,
              pred_flag     = EXCLUDED.pred_flag,
              pred_qty_cond = EXCLUDED.pred_qty_cond,
              pred_two_stage= EXCLUDED.pred_two_stage
            """
            con.execute(text(sql_future_upsert), fact.to_dict(orient="records"))

        # --- Metrics blob ---
        try:
            payload = json.load(open("outputs/metrics.json","r",encoding="utf-8"))
            con.execute(
                text("INSERT OR REPLACE INTO run_metrics (run_ts, model_version, payload_json) VALUES (:run_ts,:mv,:p)"),
                {"run_ts": run_ts, "mv": model_version, "p": json.dumps(payload)},
            )
        except Exception:
            pass

    print("Loaded forecasts (backtest + future) into DB with product column + duplicate protection.")

if __name__ == "__main__":
    main()