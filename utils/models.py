def fit_lightgbm_if_available(X_train, y_train, X_val, y_val, random_state=42):
    """
    Global LightGBM regressor (version-safe, sklearn API).
    Returns (model, {"best_iteration": ...}) or (None, {"error": "..."}).
    """
    try:
        import lightgbm as lgb
    except Exception:
        return None, {"error": "lightgbm not installed"}

    #uses sklearn-compatible LightGBM regressor API, not the raw Booster
    #parameters are tuned for balanced bias/variance on time series data
    model = lgb.LGBMRegressor(
        objective="regression", #standard regression objective
        learning_rate=0.05, #small learning rate for smoother convergence
        num_leaves=63, #controls tree complexity
        feature_fraction=0.8, #random subset of features per iteration
        subsample=0.8, #random subset of rows per iteration
        subsample_freq=1, #frequency for subsampling
        n_estimators=200, #maximum boosting rounds
        random_state=random_state,
        n_jobs=-1, #use all CPU cores
    )

    #set up callbacks for early stopping and logging if supported by installed LightGBM version
    callbacks = []

    #early stopping to prevent overfitting if validation loss has no improvement after 30 rounds
    if hasattr(lgb, "early_stopping"):
        callbacks.append(lgb.early_stopping(stopping_rounds=30))

    #log_evaluation controls how often validation results are printed
    if hasattr(lgb, "log_evaluation"):
        callbacks.append(lgb.log_evaluation(period=0))

    #trains the model
    #fit() uses sklearn API with eval_set for validation data and early stopping
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks or None)

    #extracts best iteration
    best_iter = getattr(model, "best_iteration_", None) or getattr(model, "_best_iteration", None)

    return model, {"best_iteration": best_iter}


def fit_lgbm_classifier_if_available(X_train, y_train, X_val, y_val, random_state=42):
    """
    Global LightGBM binary classifier (version-safe, sklearn API).
    Returns (model, {"best_iteration": ...}) or (None, {"error": "..."}).
    """
    try:
        import lightgbm as lgb
    except Exception:
        return None, {"error": "lightgbm not installed"}

    #same as regressor but has a classifier, note that num_leaves is smaller because binary classification is simpler
    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        num_leaves=31,
        n_estimators=400, #allow more rounds because early stopping will prune
        feature_fraction=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=random_state,
        n_jobs=-1,
    )

    callbacks = []

    #early stopping to prevent overfitting if validation loss has no improvement after 30 rounds
    if hasattr(lgb, "early_stopping"):
        callbacks.append(lgb.early_stopping(stopping_rounds=30))

    #log_evaluation controls how often validation results are printed
    if hasattr(lgb, "log_evaluation"):
        callbacks.append(lgb.log_evaluation(period=0))

    #train model on training set and validate on validation set
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks or None)

    #extract best iteration
    best_iter = getattr(model, "best_iteration_", None) or getattr(model, "_best_iteration", None)

    return model, {"best_iteration": best_iter}