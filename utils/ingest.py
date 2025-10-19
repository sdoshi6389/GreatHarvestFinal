import pandas as pd, numpy as np, re, glob

#robust file reader
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

#finds first column that matches any candidate name
def _find_first_col(df, candidates):
    for c in candidates:
        # exact match
        for col in df.columns:
            if col.strip().lower() == c.strip().lower():
                return col
        # substring match
        for col in df.columns:
            if c.strip().lower() in col.strip().lower():
                return col
    return None

#normalize product strings for grouping
def normalize_product_name(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9\s\-\/]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def ingest_many(glob_pattern, cfg):

    #expands the wildcard to actual file paths
    files = sorted(glob.glob(glob_pattern))

    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    frames, meta = [], []
    for p in files:

        #reads csv into dataframe
        df = _try_read(p)

        #trims whitespace to avoid "Product " vs "Product"
        df.columns = [c.strip() for c in df.columns]

        #auto detect relevant columns using candidate lists in config.yaml file
        date_col  = _find_first_col(df, cfg["date_column_candidates"])
        prod_col  = _find_first_col(df, cfg["product_column_candidates"])
        store_col = _find_first_col(df, cfg["store_column_candidates"])
        tgt_col   = _find_first_col(df, cfg["target_column_candidates"])
        price_col = _find_first_col(df, cfg["price_column_candidates"])
        promo_col = _find_first_col(df, cfg["promo_column_candidates"])
        stock_col = _find_first_col(df, cfg["stock_column_candidates"])

        #if any required column is missing, record meta and skip this file
        if not (date_col and prod_col and tgt_col):
            meta.append({"file": p, "status":"skipped_missing_required",
                         "found": {"date":date_col,"product":prod_col,"target":tgt_col}})
            continue

        #parses date
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)

        #drops rows where date couldn't be parsed
        df = df.dropna(subset=[date_col]).copy()

        #create normalized columns for rest of pipeline
        df["date"]    = df[date_col].dt.normalize()
        df["product"] = df[prod_col].apply(normalize_product_name)
        df["store"]   = df[store_col].astype(str) if store_col else "store_all"
        df["y"]       = pd.to_numeric(df[tgt_col], errors="coerce")

        if price_col: df["price"] = pd.to_numeric(df[price_col], errors="coerce")
        if promo_col: df["promo"] = df[promo_col].astype(str).str.contains(r"1|true|yes|y", case=False, na=False).astype(int)
        if stock_col: df["stock"] = pd.to_numeric(df[stock_col], errors="coerce")

        #keep only the standardized columns we need to send down the pipeline
        keep = ["date","product","store","y"]
        for opt in ["price","promo","stock"]:
            if opt in df.columns: keep.append(opt)

        frames.append(df[keep])
        meta.append({"file": p, "status":"ingested", "rows": int(len(df))})

    #if every file was skipped, throw error
    if not frames:
        raise RuntimeError("No valid frames loaded. Check column mappings.")

    #combine all ingested dataframes into one across all files
    big = pd.concat(frames, ignore_index=True)

    #aggregate to daily per (store, product)
    agg = {"y":"sum"}
    if "price" in big.columns: agg["price"]="mean"
    if "stock" in big.columns: agg["stock"]="mean"
    if "promo" in big.columns: agg["promo"]="max"

    big = (big
           .groupby(["store","product","date"], as_index=False)
           .agg(agg)
           .sort_values(["store","product","date"])
           .reset_index(drop=True))

    return big, pd.DataFrame(meta)
