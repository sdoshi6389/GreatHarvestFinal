# main_bread_pipeline.py  (updated to accept --horizon / HORIZON_DAYS)
# imports of standard libraries
import json, yaml, pandas as pd, numpy as np
from pathlib import Path
import datetime as dt
from sqlalchemy import text
import argparse, os

# imports from other files in project
from utils.ingest import ingest_many
from utils.features import add_time_features, add_lags_rolls, train_val_test_split_by_date
from utils.baselines import naive_forecast, seasonal_naive
from utils.metrics import mae, rmse, smape
from utils.models import fit_lightgbm_if_available, fit_lgbm_classifier_if_available

# DB helpers
from db_utils import get_engine, init_forecasts_schema, ensure_dims_and_map

def _to_native_dates(df, col="date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        df = df.dropna(subset=[col]).copy()
    return df

def _coerce_for_db(df, numeric_cols=None, int_cols=None):
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

def main():
    # --- NEW: allow horizon override via CLI or env ---
    parser = argparse.ArgumentParser(description="Bread forecasting pipeline")
    parser.add_argument("--horizon", "-H", type=int, default=None,
                        help="Number of future days to forecast (overrides default)")
    args, _ = parser.parse_known_args()

    # opens the config file and parses it into a python dict
    with open("config.yaml","r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # creates the output directory if it doesn't already exist
    out = Path(cfg["outputs_dir"]); out.mkdir(parents=True, exist_ok=True)

    print(">> Ingesting CSVs…")

    # reads all csvs that are in the data_glob
    big, meta = ingest_many(cfg["data_glob"], cfg)

    # sends to files
    big.to_parquet(out/"tidy.parquet", index=False)
    meta.to_csv(out/"ingest_meta.csv", index=False)

    print(f"Rows={len(big):,} | stores={big['store'].nunique()} | products={big['product'].nunique()}")
    print(big.head())

    print(">> Feature engineering…")

    # adds calendar features (day of week, month, cyclic encodings)
    df = add_time_features(big)

    # adds historical featuers (lags, rolling means, standard windows)
    df = add_lags_rolls(df, cfg["lags"], cfg["roll_windows"])

    # binary classification target: whether bread was sold that day or not
    df["sold_flag"] = (df["y"] > 0).astype(int)

    # drop rows with NA values
    dfm = df.dropna().reset_index(drop=True)

    print(">> Chronological split…")

    # splits data by date into training, validation, and test sets (no shuffling, uses last N days for val/test)
    train, val, test = train_val_test_split_by_date(dfm, cfg["val_size_days"], cfg["test_size_days"])
    
    # choosing features (everything but the target and identifiers)
    feats = [c for c in dfm.columns if c not in ["y","date","store","product","sold_flag"]]
    
    print(f"n_train={len(train)}, n_val={len(val)}, n_test={len(test)}")
    print("Using feature columns:", feats[:10], ("… (+more)" if len(feats)>10 else ""))

    # ---- Baselines ----
    print(">> Baselines (naïve & seasonal-naïve)…")

    # concatenates splits so we can shift within each (store, product) group across the whole period
    base = pd.concat([train, val, test]).sort_values(["store","product","date"]).reset_index(drop=True)

    # naive baseline predicts y_t based on y_(t-1)
    base = naive_forecast(base)

    # seasonal naive baseline predicts y_t based on y(t-7) for weekly seasonality
    base = seasonal_naive(base, season=cfg["seasonal_period"])

    # helper function to compute MAE/RMSE/sMAPE for a predicted column on a subset of the data
    def eval_block(df_sub, pred_col):
        df_sub = df_sub.dropna(subset=[pred_col])
        return {"MAE": mae(df_sub["y"], df_sub[pred_col]),
                "RMSE": rmse(df_sub["y"], df_sub[pred_col]),
                "sMAPE": smape(df_sub["y"], df_sub[pred_col]),
                "n": int(len(df_sub))}

    val_block  = base[base["date"].isin(val["date"].unique())]
    test_block = base[base["date"].isin(test["date"].unique())]
    metrics = {
        "val":  {"naive":  eval_block(val_block,  "pred_naive"),
                 "snaive": eval_block(val_block,  "pred_snaive")},
        "test": {"naive":  eval_block(test_block, "pred_naive"),
                 "snaive": eval_block(test_block, "pred_snaive")},
    }

    # ---- LightGBM (global regressor) on all rows ----
    print(">> LightGBM (global regression)…")

    model_reg, info_reg = fit_lightgbm_if_available(train[feats], train["y"], val[feats], val["y"])
    if model_reg is None:
        print("LightGBM not available for regression:", info_reg.get("error"))
    else:
        import numpy as np
        best_iter_reg = (info_reg.get("best_iteration")
                         or getattr(model_reg, "best_iteration_", None)
                         or getattr(model_reg, "_best_iteration", None))
        val_pred_reg  = model_reg.predict(val[feats],  num_iteration=best_iter_reg) if best_iter_reg else model_reg.predict(val[feats])
        test_pred_reg = model_reg.predict(test[feats], num_iteration=best_iter_reg) if best_iter_reg else model_reg.predict(test[feats])
        metrics["val"]["lightgbm"]  = {"MAE": float(np.mean(np.abs(val["y"]-val_pred_reg))),
                                       "RMSE": float(np.sqrt(np.mean((val["y"]-val_pred_reg)**2))),
                                       "sMAPE": float(100*np.mean(np.abs(val["y"]-val_pred_reg)/((np.abs(val["y"])+np.abs(val_pred_reg))/2))),
                                       "n": int(len(val))}
        metrics["test"]["lightgbm"] = {"MAE": float(np.mean(np.abs(test["y"]-test_pred_reg))),
                                       "RMSE": float(np.sqrt(np.mean((test["y"]-test_pred_reg)**2))),
                                       "sMAPE": float(100*np.mean(np.abs(test["y"]-test_pred_reg)/((np.abs(test["y"])+np.abs(test_pred_reg))/2))),
                                       "n": int(len(test))}

    # ---- TWO-STAGE: Classifier (sold or not) + Regressor on positives ----
    print(">> Two-stage model: classifier + conditional regressor…")
    
    model_clf, info_clf = fit_lgbm_classifier_if_available(train[feats], train["sold_flag"], val[feats], val["sold_flag"])
    if model_clf is None:
        print("LightGBM not available for classification:", info_clf.get("error"))
    else:
        val_prob = model_clf.predict_proba(val[feats])[:, 1]
        test_prob = model_clf.predict_proba(test[feats])[:, 1]

        import numpy as np
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

        cand = np.linspace(0.2, 0.8, 61)
        f1s = [f1_score(val["sold_flag"], (val_prob > t).astype(int)) for t in cand]
        best_t = float(cand[int(np.argmax(f1s))])

        val_flag = (val_prob > best_t).astype(int)
        test_flag = (test_prob > best_t).astype(int)

        train_pos = train[train["sold_flag"] == 1].copy()
        val_pos   = val[val["sold_flag"] == 1].copy()
        model_reg_pos, info_reg_pos = fit_lightgbm_if_available(train_pos[feats], train_pos["y"], val_pos[feats], val_pos["y"])
        if model_reg_pos is None:
            print("LightGBM not available for conditional regression:", info_reg_pos.get("error"))
        else:
            best_iter_pos = (info_reg_pos.get("best_iteration")
                             or getattr(model_reg_pos, "best_iteration_", None)
                             or getattr(model_reg_pos, "_best_iteration", None))
            val_qty  = model_reg_pos.predict(val[feats],  num_iteration=best_iter_pos) if best_iter_pos else model_reg_pos.predict(val[feats])
            test_qty = model_reg_pos.predict(test[feats], num_iteration=best_iter_pos) if best_iter_pos else model_reg_pos.predict(test[feats])

            val_combined  = val_flag  * val_qty
            test_combined = test_flag * test_qty

            metrics.setdefault("val", {})["two_stage"]  = {"MAE": mae(val["y"], val_combined),
                                                           "RMSE": rmse(val["y"], val_combined),
                                                           "sMAPE": smape(val["y"], val_combined),
                                                           "n": int(len(val))}
            metrics.setdefault("test", {})["two_stage"] = {"MAE": mae(test["y"], test_combined),
                                                           "RMSE": rmse(test["y"], test_combined),
                                                           "sMAPE": smape(test["y"], test_combined),
                                                           "n": int(len(test))}
            metrics["classification"] = {
                "val":  {"AUC": float(roc_auc_score(val["sold_flag"], val_prob)),
                         "F1": float(f1_score(val["sold_flag"], val_flag)),
                         "Accuracy": float(accuracy_score(val["sold_flag"], val_flag)),
                         "threshold": best_t},
                "test": {"AUC": float(roc_auc_score(test["sold_flag"], test_prob)),
                         "F1": float(f1_score(test["sold_flag"], test_flag)),
                         "Accuracy": float(accuracy_score(test["sold_flag"], test_flag)),
                         "threshold": best_t}
            }

            pd.DataFrame({
                "date": test["date"], "store": test["store"], "product": test["product"],
                "y": test["y"],
                "pred_prob": test_prob,
                "pred_flag": test_flag,
                "pred_qty_cond": test_qty,
                "pred_two_stage": test_combined
            }).to_csv(out/"predictions_two_stage_test.csv", index=False)

    # ---- Multi-step forecast for the next N days ----
    print(">> Building multi-step forecast…")

    # DEFAULT H, overridden by CLI arg or env var
    default_h = 2
    env_h = os.environ.get("HORIZON_DAYS")
    H = (args.horizon if args.horizon is not None
         else (int(env_h) if env_h else default_h))
    H = max(1, int(H))  # safety clamp
    print(f">> Using horizon H = {H} day(s)")

    work = dfm.copy()
    last_date_all = work["date"].max()

    latest_keys = (
        work.sort_values(["store", "product", "date"])
            .groupby(["store", "product"], as_index=False)
            .tail(1)[["store", "product"]]
            .reset_index(drop=True)
    )

    def _best_iter(m):
        return (getattr(m, "best_iteration_", None)
                or getattr(m, "_best_iteration", None))

    model_feature_cols = [c for c in dfm.columns if c not in ["y","sold_flag","date","store","product"]]
    day_forecasts = []

    for step in range(1, H + 1):
        future_date = last_date_all + pd.Timedelta(days=step)
        print(f"  -> Day +{step} ({future_date.date()})")

        future = latest_keys.copy()
        future["date"] = future_date
        future = add_time_features(future)

        tmp = pd.concat([work, future], ignore_index=True, sort=False)
        tmp = add_lags_rolls(tmp, cfg["lags"], cfg["roll_windows"])
        future_feats = tmp[tmp["date"] == future_date].copy()

        if "stock" in work.columns:
            last_stock = (
                work.sort_values(["store","product","date"])
                    .groupby(["store","product"], as_index=False)
                    .tail(1)[["store","product","stock"]]
            )
            future_feats = (future_feats.drop(columns=["stock"], errors="ignore")
                            .merge(last_stock, on=["store","product"], how="left"))

        Xf = future_feats[model_feature_cols].copy()
        day_out = future_feats[["date","store","product"]].copy()

        if 'model_clf' in locals() and model_clf is not None and 'model_reg_pos' in locals() and model_reg_pos is not None:
            prob = model_clf.predict_proba(Xf)[:, 1]
            best_t_local = locals().get('best_t', 0.5)
            flag = (prob > best_t_local).astype(int)
            best_iter_pos = _best_iter(model_reg_pos)
            qty_cond = (model_reg_pos.predict(Xf, num_iteration=best_iter_pos)
                        if best_iter_pos else model_reg_pos.predict(Xf))
            yhat = flag * qty_cond
            day_out["pred_prob"] = prob
            day_out["pred_flag"] = flag
            day_out["pred_qty_cond"] = qty_cond
            day_out["pred_two_stage"] = yhat
        elif 'model_reg' in locals() and model_reg is not None:
            best_iter_reg_local = _best_iter(model_reg)
            yhat = (model_reg.predict(Xf, num_iteration=best_iter_reg_local)
                    if best_iter_reg_local else model_reg.predict(Xf))
            day_out["pred_qty"] = yhat
        else:
            print("No trained models available; skipping multi-step.")
            break

        new_rows = future_feats[["date","store","product"]].copy()
        new_rows["y"] = (day_out.get("pred_two_stage", day_out.get("pred_qty")))
        if "sold_flag" in work.columns:
            new_rows["sold_flag"] = (new_rows["y"] > 0).astype(int)
        for col in work.columns:
            if col not in new_rows.columns:
                new_rows[col] = np.nan
        work = pd.concat([work, new_rows[work.columns]], ignore_index=True, sort=False)

        day_out["horizon"] = step  # keep horizon per step
        day_forecasts.append(day_out)

    # part 7 - save all future days
    future_df = None
    if day_forecasts:
        future_df = pd.concat(day_forecasts, ignore_index=True)
        future_path = Path(cfg["outputs_dir"]) / f"predictions_next_{H}_days.csv"
        future_df.to_csv(future_path, index=False)
        print(f">> Wrote {future_path.name}")

    # ---- NEW: write future forecasts to DB (upsert by date/store/product/horizon) ----
    if future_df is not None and len(future_df):
        # prepare DB
        dbc = yaml.safe_load(open("db_config.yaml","r",encoding="utf-8"))
        eng = get_engine(dbc["forecasts_db"])
        init_forecasts_schema(eng)

        run_ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        model_version = "lightgbm_two_stage_v1"

        # normalize types
        future_df = _to_native_dates(future_df, "date")
        num_cols = [c for c in ["pred_prob","pred_qty_cond","pred_two_stage","pred_qty"] if c in future_df.columns]
        int_cols = [c for c in ["pred_flag"] if c in future_df.columns]
        future_df = _coerce_for_db(future_df, numeric_cols=num_cols, int_cols=int_cols)

        # pick unified columns for DB (support both two-stage and single-regressor)
        for missing in ["pred_prob","pred_flag","pred_qty_cond","pred_two_stage"]:
            if missing not in future_df.columns:
                future_df[missing] = None

        # map to IDs and write
        with eng.begin() as con:
            fact = ensure_dims_and_map(con, future_df[["date","store","product","horizon","pred_prob","pred_flag","pred_qty_cond","pred_two_stage"]])
            fact["run_ts"] = run_ts
            fact["model_version"] = model_version

            # delete-then-insert "upsert" per (date, store_id, product_id, horizon)
            delete_sql = """
            DELETE FROM forecasts_future
            WHERE date = :date AND store_id = :store_id AND product_id = :product_id
                  AND (:horizon IS NULL OR horizon = :horizon)
            """
            keys = fact[["date","store_id","product_id","horizon"]].drop_duplicates().to_dict(orient="records")
            con.execute(text(delete_sql), keys)

            backend = con.engine.url.get_backend_name()
            if backend == "sqlite":
                insert_sql = """
                INSERT OR REPLACE INTO forecasts_future
                  (run_ts, model_version, horizon, date, store_id, product_id,
                   pred_prob, pred_flag, pred_qty_cond, pred_two_stage)
                VALUES (:run_ts, :model_version, :horizon, :date, :store_id, :product_id,
                        :pred_prob, :pred_flag, :pred_qty_cond, :pred_two_stage)
                """
            else:
                insert_sql = """
                INSERT INTO forecasts_future
                  (run_ts, model_version, horizon, date, store_id, product_id,
                   pred_prob, pred_flag, pred_qty_cond, pred_two_stage)
                VALUES (:run_ts, :model_version, :horizon, :date, :store_id, :product_id,
                        :pred_prob, :pred_flag, :pred_qty_cond, :pred_two_stage)
                """
            con.execute(text(insert_sql), fact.to_dict(orient="records"))

        print(">> Wrote future forecasts to DB (upserted by date/store/product/horizon).")

    # dump metrics into json file
    with open(out/"metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(">> Done. See outputs/metrics.json")

if __name__ == "__main__":
    main()


# # main_bread_pipeline.py
# # imports of standard libraries
# import json, yaml, pandas as pd, numpy as np
# from pathlib import Path
# import datetime as dt
# from sqlalchemy import text

# # imports from other files in project
# from utils.ingest import ingest_many
# from utils.features import add_time_features, add_lags_rolls, train_val_test_split_by_date
# from utils.baselines import naive_forecast, seasonal_naive
# from utils.metrics import mae, rmse, smape
# from utils.models import fit_lightgbm_if_available, fit_lgbm_classifier_if_available

# # DB helpers
# from db_utils import get_engine, init_forecasts_schema, ensure_dims_and_map

# def _to_native_dates(df, col="date"):
#     if col in df.columns:
#         df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
#         df = df.dropna(subset=[col]).copy()
#     return df

# def _coerce_for_db(df, numeric_cols=None, int_cols=None):
#     numeric_cols = numeric_cols or []
#     int_cols = int_cols or []
#     for c in numeric_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
#     for c in int_cols:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
#             df[c] = df[c].where(pd.notna(df[c]), None)
#             df[c] = df[c].apply(lambda x: int(x) if x is not None else None)
#     df = df.where(pd.notna(df), None)
#     return df

# def main():
#     # opens the config file and parses it into a python dict
#     with open("config.yaml","r",encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)

#     # creates the output directory if it doesn't already exist
#     out = Path(cfg["outputs_dir"]); out.mkdir(parents=True, exist_ok=True)

#     print(">> Ingesting CSVs…")

#     # reads all csvs that are in the data_glob
#     big, meta = ingest_many(cfg["data_glob"], cfg)

#     # sends to files
#     big.to_parquet(out/"tidy.parquet", index=False)
#     meta.to_csv(out/"ingest_meta.csv", index=False)

#     print(f"Rows={len(big):,} | stores={big['store'].nunique()} | products={big['product'].nunique()}")
#     print(big.head())

#     print(">> Feature engineering…")

#     # adds calendar features (day of week, month, cyclic encodings)
#     df = add_time_features(big)

#     # adds historical featuers (lags, rolling means, standard windows)
#     df = add_lags_rolls(df, cfg["lags"], cfg["roll_windows"])

#     # binary classification target: whether bread was sold that day or not
#     df["sold_flag"] = (df["y"] > 0).astype(int)

#     # drop rows with NA values
#     dfm = df.dropna().reset_index(drop=True)

#     print(">> Chronological split…")

#     # splits data by date into training, validation, and test sets (no shuffling, uses last N days for val/test)
#     train, val, test = train_val_test_split_by_date(dfm, cfg["val_size_days"], cfg["test_size_days"])
    
#     # choosing features (everything but the target and identifiers)
#     feats = [c for c in dfm.columns if c not in ["y","date","store","product","sold_flag"]]
    
#     print(f"n_train={len(train)}, n_val={len(val)}, n_test={len(test)}")
#     print("Using feature columns:", feats[:10], ("… (+more)" if len(feats)>10 else ""))

#     # ---- Baselines ----
#     print(">> Baselines (naïve & seasonal-naïve)…")

#     # concatenates splits so we can shift within each (store, product) group across the whole period
#     base = pd.concat([train, val, test]).sort_values(["store","product","date"]).reset_index(drop=True)

#     # naive baseline predicts y_t based on y_(t-1)
#     base = naive_forecast(base)

#     # seasonal naive baseline predicts y_t based on y(t-7) for weekly seasonality
#     base = seasonal_naive(base, season=cfg["seasonal_period"])

#     # helper function to compute MAE/RMSE/sMAPE for a predicted column on a subset of the data
#     def eval_block(df_sub, pred_col):
#         df_sub = df_sub.dropna(subset=[pred_col])
#         return {"MAE": mae(df_sub["y"], df_sub[pred_col]),
#                 "RMSE": rmse(df_sub["y"], df_sub[pred_col]),
#                 "sMAPE": smape(df_sub["y"], df_sub[pred_col]),
#                 "n": int(len(df_sub))}

#     val_block  = base[base["date"].isin(val["date"].unique())]
#     test_block = base[base["date"].isin(test["date"].unique())]
#     metrics = {
#         "val":  {"naive":  eval_block(val_block,  "pred_naive"),
#                  "snaive": eval_block(val_block,  "pred_snaive")},
#         "test": {"naive":  eval_block(test_block, "pred_naive"),
#                  "snaive": eval_block(test_block, "pred_snaive")},
#     }

#     # ---- LightGBM (global regressor) on all rows ----
#     print(">> LightGBM (global regression)…")

#     model_reg, info_reg = fit_lightgbm_if_available(train[feats], train["y"], val[feats], val["y"])
#     if model_reg is None:
#         print("LightGBM not available for regression:", info_reg.get("error"))
#     else:
#         import numpy as np
#         best_iter_reg = (info_reg.get("best_iteration")
#                          or getattr(model_reg, "best_iteration_", None)
#                          or getattr(model_reg, "_best_iteration", None))
#         val_pred_reg  = model_reg.predict(val[feats],  num_iteration=best_iter_reg) if best_iter_reg else model_reg.predict(val[feats])
#         test_pred_reg = model_reg.predict(test[feats], num_iteration=best_iter_reg) if best_iter_reg else model_reg.predict(test[feats])
#         metrics["val"]["lightgbm"]  = {"MAE": float(np.mean(np.abs(val["y"]-val_pred_reg))),
#                                        "RMSE": float(np.sqrt(np.mean((val["y"]-val_pred_reg)**2))),
#                                        "sMAPE": float(100*np.mean(np.abs(val["y"]-val_pred_reg)/((np.abs(val["y"])+np.abs(val_pred_reg))/2))),
#                                        "n": int(len(val))}
#         metrics["test"]["lightgbm"] = {"MAE": float(np.mean(np.abs(test["y"]-test_pred_reg))),
#                                        "RMSE": float(np.sqrt(np.mean((test["y"]-test_pred_reg)**2))),
#                                        "sMAPE": float(100*np.mean(np.abs(test["y"]-test_pred_reg)/((np.abs(test["y"])+np.abs(test_pred_reg))/2))),
#                                        "n": int(len(test))}

#     # ---- TWO-STAGE: Classifier (sold or not) + Regressor on positives ----
#     print(">> Two-stage model: classifier + conditional regressor…")
    
#     model_clf, info_clf = fit_lgbm_classifier_if_available(train[feats], train["sold_flag"], val[feats], val["sold_flag"])
#     if model_clf is None:
#         print("LightGBM not available for classification:", info_clf.get("error"))
#     else:
#         val_prob = model_clf.predict_proba(val[feats])[:, 1]
#         test_prob = model_clf.predict_proba(test[feats])[:, 1]

#         import numpy as np
#         from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

#         cand = np.linspace(0.2, 0.8, 61)
#         f1s = [f1_score(val["sold_flag"], (val_prob > t).astype(int)) for t in cand]
#         best_t = float(cand[int(np.argmax(f1s))])

#         val_flag = (val_prob > best_t).astype(int)
#         test_flag = (test_prob > best_t).astype(int)

#         train_pos = train[train["sold_flag"] == 1].copy()
#         val_pos   = val[val["sold_flag"] == 1].copy()
#         model_reg_pos, info_reg_pos = fit_lightgbm_if_available(train_pos[feats], train_pos["y"], val_pos[feats], val_pos["y"])
#         if model_reg_pos is None:
#             print("LightGBM not available for conditional regression:", info_reg_pos.get("error"))
#         else:
#             best_iter_pos = (info_reg_pos.get("best_iteration")
#                              or getattr(model_reg_pos, "best_iteration_", None)
#                              or getattr(model_reg_pos, "_best_iteration", None))
#             val_qty  = model_reg_pos.predict(val[feats],  num_iteration=best_iter_pos) if best_iter_pos else model_reg_pos.predict(val[feats])
#             test_qty = model_reg_pos.predict(test[feats], num_iteration=best_iter_pos) if best_iter_pos else model_reg_pos.predict(test[feats])

#             val_combined  = val_flag  * val_qty
#             test_combined = test_flag * test_qty

#             metrics.setdefault("val", {})["two_stage"]  = {"MAE": mae(val["y"], val_combined),
#                                                            "RMSE": rmse(val["y"], val_combined),
#                                                            "sMAPE": smape(val["y"], val_combined),
#                                                            "n": int(len(val))}
#             metrics.setdefault("test", {})["two_stage"] = {"MAE": mae(test["y"], test_combined),
#                                                            "RMSE": rmse(test["y"], test_combined),
#                                                            "sMAPE": smape(test["y"], test_combined),
#                                                            "n": int(len(test))}
#             metrics["classification"] = {
#                 "val":  {"AUC": float(roc_auc_score(val["sold_flag"], val_prob)),
#                          "F1": float(f1_score(val["sold_flag"], val_flag)),
#                          "Accuracy": float(accuracy_score(val["sold_flag"], val_flag)),
#                          "threshold": best_t},
#                 "test": {"AUC": float(roc_auc_score(test["sold_flag"], test_prob)),
#                          "F1": float(f1_score(test["sold_flag"], test_flag)),
#                          "Accuracy": float(accuracy_score(test["sold_flag"], test_flag)),
#                          "threshold": best_t}
#             }

#             pd.DataFrame({
#                 "date": test["date"], "store": test["store"], "product": test["product"],
#                 "y": test["y"],
#                 "pred_prob": test_prob,
#                 "pred_flag": test_flag,
#                 "pred_qty_cond": test_qty,
#                 "pred_two_stage": test_combined
#             }).to_csv(out/"predictions_two_stage_test.csv", index=False)

#     # ---- Multi-step forecast for the next N days ----
#     print(">> Building multi-step forecast…")

#     H = 2  # you can change this
#     work = dfm.copy()
#     last_date_all = work["date"].max()

#     latest_keys = (
#         work.sort_values(["store", "product", "date"])
#             .groupby(["store", "product"], as_index=False)
#             .tail(1)[["store", "product"]]
#             .reset_index(drop=True)
#     )

#     def _best_iter(m):
#         return (getattr(m, "best_iteration_", None)
#                 or getattr(m, "_best_iteration", None))

#     model_feature_cols = [c for c in dfm.columns if c not in ["y","sold_flag","date","store","product"]]
#     day_forecasts = []

#     for step in range(1, H + 1):
#         future_date = last_date_all + pd.Timedelta(days=step)
#         print(f"  -> Day +{step} ({future_date.date()})")

#         future = latest_keys.copy()
#         future["date"] = future_date
#         future = add_time_features(future)

#         tmp = pd.concat([work, future], ignore_index=True, sort=False)
#         tmp = add_lags_rolls(tmp, cfg["lags"], cfg["roll_windows"])
#         future_feats = tmp[tmp["date"] == future_date].copy()

#         if "stock" in work.columns:
#             last_stock = (
#                 work.sort_values(["store","product","date"])
#                     .groupby(["store","product"], as_index=False)
#                     .tail(1)[["store","product","stock"]]
#             )
#             future_feats = (future_feats.drop(columns=["stock"], errors="ignore")
#                             .merge(last_stock, on=["store","product"], how="left"))

#         Xf = future_feats[model_feature_cols].copy()
#         day_out = future_feats[["date","store","product"]].copy()

#         if 'model_clf' in locals() and model_clf is not None and 'model_reg_pos' in locals() and model_reg_pos is not None:
#             prob = model_clf.predict_proba(Xf)[:, 1]
#             best_t_local = locals().get('best_t', 0.5)
#             flag = (prob > best_t_local).astype(int)
#             best_iter_pos = _best_iter(model_reg_pos)
#             qty_cond = (model_reg_pos.predict(Xf, num_iteration=best_iter_pos)
#                         if best_iter_pos else model_reg_pos.predict(Xf))
#             yhat = flag * qty_cond
#             day_out["pred_prob"] = prob
#             day_out["pred_flag"] = flag
#             day_out["pred_qty_cond"] = qty_cond
#             day_out["pred_two_stage"] = yhat
#         elif 'model_reg' in locals() and model_reg is not None:
#             best_iter_reg_local = _best_iter(model_reg)
#             yhat = (model_reg.predict(Xf, num_iteration=best_iter_reg_local)
#                     if best_iter_reg_local else model_reg.predict(Xf))
#             day_out["pred_qty"] = yhat
#         else:
#             print("No trained models available; skipping multi-step.")
#             break

#         new_rows = future_feats[["date","store","product"]].copy()
#         new_rows["y"] = (day_out.get("pred_two_stage", day_out.get("pred_qty")))
#         if "sold_flag" in work.columns:
#             new_rows["sold_flag"] = (new_rows["y"] > 0).astype(int)
#         for col in work.columns:
#             if col not in new_rows.columns:
#                 new_rows[col] = np.nan
#         work = pd.concat([work, new_rows[work.columns]], ignore_index=True, sort=False)

#         day_out["horizon"] = step  # keep horizon per step
#         day_forecasts.append(day_out)

#     # part 7 - save all future days
#     future_df = None
#     if day_forecasts:
#         future_df = pd.concat(day_forecasts, ignore_index=True)
#         future_path = Path(cfg["outputs_dir"]) / f"predictions_next_{H}_days.csv"
#         future_df.to_csv(future_path, index=False)
#         print(f">> Wrote {future_path.name}")

#     # ---- NEW: write future forecasts to DB (upsert by date/store/product/horizon) ----
#     if future_df is not None and len(future_df):
#         # prepare DB
#         dbc = yaml.safe_load(open("db_config.yaml","r",encoding="utf-8"))
#         eng = get_engine(dbc["forecasts_db"])
#         init_forecasts_schema(eng)

#         run_ts = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
#         model_version = "lightgbm_two_stage_v1"

#         # normalize types
#         future_df = _to_native_dates(future_df, "date")
#         num_cols = [c for c in ["pred_prob","pred_qty_cond","pred_two_stage","pred_qty"] if c in future_df.columns]
#         int_cols = [c for c in ["pred_flag"] if c in future_df.columns]
#         future_df = _coerce_for_db(future_df, numeric_cols=num_cols, int_cols=int_cols)

#         # pick unified columns for DB (support both two-stage and single-regressor)
#         for missing in ["pred_prob","pred_flag","pred_qty_cond","pred_two_stage"]:
#             if missing not in future_df.columns:
#                 future_df[missing] = None

#         # map to IDs
#         with eng.begin() as con:
#             fact = ensure_dims_and_map(con, future_df[["date","store","product","horizon","pred_prob","pred_flag","pred_qty_cond","pred_two_stage"]])
#             fact["run_ts"] = run_ts
#             fact["model_version"] = model_version

#             # delete-then-insert "upsert" per (date, store_id, product_id, horizon)
#             # 1) delete existing keys
#             delete_sql = """
#             DELETE FROM forecasts_future
#             WHERE date = :date AND store_id = :store_id AND product_id = :product_id
#                   AND (:horizon IS NULL OR horizon = :horizon)
#             """
#             keys = fact[["date","store_id","product_id","horizon"]].drop_duplicates().to_dict(orient="records")
#             con.execute(text(delete_sql), keys)

#             # 2) insert fresh rows
#             backend = con.engine.url.get_backend_name()
#             if backend == "sqlite":
#                 insert_sql = """
#                 INSERT OR REPLACE INTO forecasts_future
#                   (run_ts, model_version, horizon, date, store_id, product_id,
#                    pred_prob, pred_flag, pred_qty_cond, pred_two_stage)
#                 VALUES (:run_ts, :model_version, :horizon, :date, :store_id, :product_id,
#                         :pred_prob, :pred_flag, :pred_qty_cond, :pred_two_stage)
#                 """
#             else:
#                 # if you later add a UNIQUE(date,store_id,product_id,horizon), you can switch to ON CONFLICT here
#                 insert_sql = """
#                 INSERT INTO forecasts_future
#                   (run_ts, model_version, horizon, date, store_id, product_id,
#                    pred_prob, pred_flag, pred_qty_cond, pred_two_stage)
#                 VALUES (:run_ts, :model_version, :horizon, :date, :store_id, :product_id,
#                         :pred_prob, :pred_flag, :pred_qty_cond, :pred_two_stage)
#                 """
#             con.execute(text(insert_sql), fact.to_dict(orient="records"))

#         print(">> Wrote future forecasts to DB (upserted by date/store/product/horizon).")

#     # dump metrics into json file
#     with open(out/"metrics.json","w",encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)
#     print(">> Done. See outputs/metrics.json")

# if __name__ == "__main__":
#     main()
