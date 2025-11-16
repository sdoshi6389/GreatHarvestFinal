from flask import Flask, jsonify, request, render_template
from sqlalchemy import create_engine, text
import pandas as pd
import os, sys, subprocess
from pathlib import Path

FORECASTS_DB_PATH = os.environ.get("FORECASTS_DB", "db/bread_forecasts.db")
ACTUALS_DB_PATH   = os.environ.get("ACTUALS_DB",   "db/bread_actuals.db")

FORECASTS_URI = f"sqlite:///{FORECASTS_DB_PATH}"
ACTUALS_URI   = f"sqlite:///{ACTUALS_DB_PATH}"

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates",
)

eng_forecasts = create_engine(FORECASTS_URI, future=True)
eng_actuals   = create_engine(ACTUALS_URI,   future=True)

@app.route("/")
def home():
    return render_template("index.html")

# ---------- Data API (now supports pred_flag filter for forecasts) ----------
@app.route("/api/data", methods=["GET"])
def api_get_data():
    source    = (request.args.get("source") or "forecasts").lower()
    req_limit = max(1, request.args.get("limit", default=500, type=int))  # min 1

    if source == "forecasts":
        only1 = 1 if str(request.args.get("only1", "0")).strip() in {"1", "true", "True"} else 0
        with eng_forecasts.connect() as con:
            if only1:
                total_rows = con.execute(text("SELECT COUNT(*) FROM forecasts_future WHERE pred_flag = 1")).scalar_one()
                applied_limit = min(req_limit, total_rows if total_rows is not None else 0)
                q = text("""
                    SELECT
                        date, horizon, model_version, pred_flag, pred_prob,
                        pred_qty_cond, pred_two_stage, product, product_id,
                        run_ts, store_id
                    FROM forecasts_future
                    WHERE pred_flag = 1
                    ORDER BY date DESC
                    LIMIT :limit
                """)
                df = pd.read_sql(q, con, params={"limit": applied_limit})
            else:
                total_rows = con.execute(text("SELECT COUNT(*) FROM forecasts_future")).scalar_one()
                applied_limit = min(req_limit, total_rows if total_rows is not None else 0)
                q = text("""
                    SELECT
                        date, horizon, model_version, pred_flag, pred_prob,
                        pred_qty_cond, pred_two_stage, product, product_id,
                        run_ts, store_id
                    FROM forecasts_future
                    ORDER BY date DESC
                    LIMIT :limit
                """)
                df = pd.read_sql(q, con, params={"limit": applied_limit})

        return jsonify({
            "source": "forecasts",
            "only1": bool(only1),
            "total_rows": int(total_rows),
            "applied_limit": int(applied_limit),
            "rows": df.to_dict(orient="records"),
        })

    if source == "actuals":
        with eng_actuals.connect() as con:
            total_rows = con.execute(text("SELECT COUNT(*) FROM actuals_raw")).scalar_one()
            applied_limit = min(req_limit, total_rows if total_rows is not None else 0)
            q = text("""
                SELECT
                    date, store_id, product_id, product, category,
                    TOTAL_OUT_THE_DOOR, TOTAL_AVAILABLE_TO_SELL
                FROM actuals_raw
                ORDER BY date DESC
                LIMIT :limit
            """)
            df = pd.read_sql(q, con, params={"limit": applied_limit})
        return jsonify({
            "source": "actuals",
            "total_rows": int(total_rows),
            "applied_limit": int(applied_limit),
            "rows": df.to_dict(orient="records"),
        })

    return jsonify({"error": "unknown source"}), 400

# ---------- Actuals upsert ----------
@app.route("/api/actuals", methods=["POST"])
def api_upsert_actuals():
    payload = request.get_json(force=True) or {}
    date = payload.get("date")
    if not date:
        return jsonify({"ok": False, "error": "Missing required field: date"}), 400

    row = {
        "date": date,
        "store_id":      payload.get("store_id"),
        "product_id":    payload.get("product_id"),
        "product":       payload.get("product"),
        "category":      payload.get("category"),
        "TOTAL_OUT_THE_DOOR":      payload.get("TOTAL_OUT_THE_DOOR"),
        "TOTAL_AVAILABLE_TO_SELL": payload.get("TOTAL_AVAILABLE_TO_SELL"),
    }

    with eng_actuals.begin() as con:
        exists = con.execute(
            text("SELECT 1 FROM actuals_raw WHERE date = :date LIMIT 1"),
            {"date": row["date"]}
        ).first()

        if exists:
            con.execute(text("""
                UPDATE actuals_raw
                SET store_id = :store_id,
                    product_id = :product_id,
                    product = :product,
                    category = :category,
                    TOTAL_OUT_THE_DOOR = :TOTAL_OUT_THE_DOOR,
                    TOTAL_AVAILABLE_TO_SELL = :TOTAL_AVAILABLE_TO_SELL
                WHERE date = :date
            """), row)
            action = "updated"
        else:
            con.execute(text("""
                INSERT INTO actuals_raw
                  (date, store_id, product_id, product, category,
                   TOTAL_OUT_THE_DOOR, TOTAL_AVAILABLE_TO_SELL)
                VALUES
                  (:date, :store_id, :product_id, :product, :category,
                   :TOTAL_OUT_THE_DOOR, :TOTAL_AVAILABLE_TO_SELL)
            """), row)
            action = "inserted"

    return jsonify({"ok": True, "action": action, "target": "actuals_raw"})

# ---------- Predict pipeline (unchanged from your last working version) ----------
def _find_pipeline_script() -> str | None:
    env_path = os.environ.get("PIPELINE_SCRIPT")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        if p.exists():
            return str(p.resolve())
    here = Path(__file__).resolve().parent
    candidates = ["main_bread_pipeline.py","bread_forecasting_pipeline.py","bread_forecasting_pipeline_v1.py"]
    for base in [here, here.parent, Path.cwd()]:
        for name in candidates:
            p = base / name
            if p.exists():
                return str(p.resolve())
    for base in [here, here.parent, Path.cwd()]:
        matches = list(base.rglob("*pipeline*.py"))
        if matches:
            matches.sort(key=lambda m: (("bread" not in m.name.lower()), len(str(m))))
            return str(matches[0].resolve())
    return None

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True) or {}
    try:
        days = int(payload.get("days"))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid days"}), 400
    days = max(1, min(days, int(os.environ.get("MAX_HORIZON_DAYS", 60))))

    script_path = _find_pipeline_script()
    if not script_path:
        return jsonify({"ok": False, "error": "Pipeline script not found. Set PIPELINE_SCRIPT env var or place the script next to app.py."}), 500

    script_path = str(Path(script_path).resolve())
    workdir = str(Path(script_path).parent)
    cmd = [sys.executable, script_path, "--horizon", str(days)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir, check=True)
        return jsonify({"ok": True, "days": days, "script": script_path,
                        "logs": (proc.stdout or "")[-50000:], "errors": (proc.stderr or "")[-50000:]})
    except subprocess.CalledProcessError as e:
        return jsonify({"ok": False, "error": "Pipeline failed", "code": e.returncode,
                        "script": script_path,
                        "logs": (e.stdout or "")[-50000:], "errors": (e.stderr or "")[-50000:]}), 500

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)


# from flask import Flask, jsonify, request, render_template
# from sqlalchemy import create_engine, text
# import pandas as pd
# import os, sys, subprocess

# # --- DB paths (override with env vars if you like) ---
# FORECASTS_DB_PATH = os.environ.get("FORECASTS_DB", "db/bread_forecasts.db")
# ACTUALS_DB_PATH   = os.environ.get("ACTUALS_DB",   "db/bread_actuals.db")

# FORECASTS_URI = f"sqlite:///{FORECASTS_DB_PATH}"
# ACTUALS_URI   = f"sqlite:///{ACTUALS_DB_PATH}"

# app = Flask(
#     __name__,
#     static_url_path="/static",
#     static_folder="static",
#     template_folder="templates",
# )

# # Engines for each DB
# from sqlalchemy import create_engine
# eng_forecasts = create_engine(FORECASTS_URI, future=True)
# eng_actuals   = create_engine(ACTUALS_URI,   future=True)

# # ---------- Views ----------
# @app.route("/")
# def home():
#     return render_template("index.html")

# # GET /api/data?source=forecasts|actuals&limit=500
# # Clamps limit to the table's row count to avoid repeats/phantom rows.
# @app.route("/api/data", methods=["GET"])
# def api_get_data():
#     source = (request.args.get("source") or "forecasts").lower()
#     req_limit = max(1, request.args.get("limit", default=500, type=int))  # enforce min 1

#     if source == "forecasts":
#         with eng_forecasts.connect() as con:
#             total_rows = con.execute(text("SELECT COUNT(*) FROM forecasts_future")).scalar_one()
#             applied_limit = min(req_limit, total_rows if total_rows is not None else 0)

#             q = text("""
#                 SELECT
#                     date, horizon, model_version, pred_flag, pred_prob,
#                     pred_qty_cond, pred_two_stage, product, product_id,
#                     run_ts, store_id
#                 FROM forecasts_future
#                 ORDER BY date DESC
#                 LIMIT :limit
#             """)
#             df = pd.read_sql(q, con, params={"limit": applied_limit})
#         return jsonify({
#             "source": "forecasts",
#             "total_rows": int(total_rows),
#             "applied_limit": int(applied_limit),
#             "rows": df.to_dict(orient="records"),
#         })

#     if source == "actuals":
#         with eng_actuals.connect() as con:
#             total_rows = con.execute(text("SELECT COUNT(*) FROM actuals_raw")).scalar_one()
#             applied_limit = min(req_limit, total_rows if total_rows is not None else 0)

#             q = text("""
#                 SELECT
#                     date, store_id, product_id, product, category,
#                     TOTAL_OUT_THE_DOOR, TOTAL_AVAILABLE_TO_SELL
#                 FROM actuals_raw
#                 ORDER BY date DESC
#                 LIMIT :limit
#             """)
#             df = pd.read_sql(q, con, params={"limit": applied_limit})
#         return jsonify({
#             "source": "actuals",
#             "total_rows": int(total_rows),
#             "applied_limit": int(applied_limit),
#             "rows": df.to_dict(orient="records"),
#         })

#     return jsonify({"error": "unknown source"}), 400

# # POST /api/actuals  → upsert-by-date ONLY into actuals_raw
# @app.route("/api/actuals", methods=["POST"])
# def api_upsert_actuals():
#     payload = request.get_json(force=True) or {}
#     date = payload.get("date")
#     if not date:
#         return jsonify({"ok": False, "error": "Missing required field: date"}), 400

#     row = {
#         "date": date,
#         "store_id":      payload.get("store_id"),
#         "product_id":    payload.get("product_id"),
#         "product":       payload.get("product"),
#         "category":      payload.get("category"),
#         "TOTAL_OUT_THE_DOOR":      payload.get("TOTAL_OUT_THE_DOOR"),
#         "TOTAL_AVAILABLE_TO_SELL": payload.get("TOTAL_AVAILABLE_TO_SELL"),
#     }

#     with eng_actuals.begin() as con:
#         exists = con.execute(
#             text("SELECT 1 FROM actuals_raw WHERE date = :date LIMIT 1"),
#             {"date": row["date"]}
#         ).first()

#         if exists:
#             con.execute(text("""
#                 UPDATE actuals_raw
#                 SET store_id = :store_id,
#                     product_id = :product_id,
#                     product = :product,
#                     category = :category,
#                     TOTAL_OUT_THE_DOOR = :TOTAL_OUT_THE_DOOR,
#                     TOTAL_AVAILABLE_TO_SELL = :TOTAL_AVAILABLE_TO_SELL
#                 WHERE date = :date
#             """), row)
#             action = "updated"
#         else:
#             con.execute(text("""
#                 INSERT INTO actuals_raw
#                   (date, store_id, product_id, product, category,
#                    TOTAL_OUT_THE_DOOR, TOTAL_AVAILABLE_TO_SELL)
#                 VALUES
#                   (:date, :store_id, :product_id, :product, :category,
#                    :TOTAL_OUT_THE_DOOR, :TOTAL_AVAILABLE_TO_SELL)
#             """), row)
#             action = "inserted"

#     return jsonify({"ok": True, "action": action, "target": "actuals_raw"})

# # POST /api/predict  → run bread_forecasting_pipeline.py with --horizon X
# @app.route("/api/predict", methods=["POST"])
# def api_predict():
#     payload = request.get_json(force=True) or {}
#     days = payload.get("days")
#     try:
#         days = int(days)
#     except Exception:
#         return jsonify({"ok": False, "error": "Invalid days"}), 400

#     # clamp horizon for safety (adjust as you like)
#     days = max(1, min(days, int(os.environ.get("MAX_HORIZON_DAYS", 60))))

#     # run the pipeline using the same Python interpreter
#     script = os.path.abspath("bread_forecasting_pipeline.py")
#     if not os.path.exists(script):
#         return jsonify({"ok": False, "error": f"Script not found: {script}"}), 500

#     cmd = [sys.executable, script, "--horizon", str(days)]
#     try:
#         proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), check=True)
#         stdout = proc.stdout[-50000:]  # return tail if huge
#         stderr = proc.stderr[-50000:]
#         # After pipeline runs, it already writes forecasts to DB.
#         return jsonify({"ok": True, "days": days, "logs": stdout, "errors": stderr})
#     except subprocess.CalledProcessError as e:
#         return jsonify({"ok": False, "error": "Pipeline failed", "code": e.returncode,
#                         "logs": (e.stdout or "")[-50000:], "errors": (e.stderr or "")[-50000:]}), 500

# # Silence favicon 404
# @app.route("/favicon.ico")
# def favicon():
#     return ("", 204)

# if __name__ == "__main__":
#     app.run(debug=True)