"""Training pipeline and model benchmarking for Ames Housing Price Prediction."""

import os
import sys
from typing import Dict, Any, List
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_dataset, prepare_data

# Ensure standard output handles UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



def generate_eda_plots(df: pd.DataFrame, target_col: str = "SalePrice", reports_dir: str = "reports") -> None:
    """Generate and save foundational exploratory data analysis plots.

    Args:
        df: Cleaned Ames Housing DataFrame
        target_col: Target column name
        reports_dir: Directory where figures will be saved
    """
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.size"] = 11

    # 1. Target Variable Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df[target_col], kde=True, color="skyblue", bins=30)
    plt.title("Target Distribution: House Sale Price (SalePrice)", fontsize=13, weight="bold")
    plt.xlabel("Sale Price ($)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "target_distribution.png"), dpi=300)
    plt.close()

    # 2. Correlation Heatmap of Top 10 Numeric Features
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    top_corr = numeric_df.corr()[target_col].abs().sort_values(ascending=False).head(10).index
    sns.heatmap(numeric_df[top_corr].corr(), annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Top 10 Numeric Features with SalePrice", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # 3. Living Area vs. Sale Price
    if "Gr Liv Area" in df.columns:
        plt.figure(figsize=(9, 5))
        sns.scatterplot(data=df, x="Gr Liv Area", y=target_col, alpha=0.6, color="teal")
        plt.title("Above Grade Living Area (Gr Liv Area) vs. Sale Price", fontsize=13, weight="bold")
        plt.xlabel("Above Grade Living Area (sq ft)")
        plt.ylabel("Sale Price ($)")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "living_area_vs_price.png"), dpi=300)
        plt.close()

    print(f"[OK] EDA plots successfully saved to '{reports_dir}/'.")


def benchmark_models(
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    reports_dir: str = "reports",
) -> pd.DataFrame:
    """Benchmark Linear Regression, Random Forest, and XGBoost.

    Args:
        X_train_proc: Preprocessed training features
        X_test_proc: Preprocessed testing features
        y_train: Training target series
        y_test: Testing target series
        reports_dir: Path to save benchmark figure

    Returns:
        DataFrame containing evaluation metrics
    """
    models: Dict[str, Any] = {
        "Multiple Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    }

    results: List[Dict[str, Any]] = []

    print("\n--- Benchmarking Regression Architectures ---")
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results.append({
            "Model": name,
            "R2 Score": r2,
            "MAE ($)": mae,
            "RMSE ($)": rmse,
        })
        print(f"  {name:30s} | R²: {r2:.4f} | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f}")

    results_df = pd.DataFrame(results)

    # Plot Model Benchmarking
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="Model", y="R2 Score", palette="viridis")
    plt.title("Model Benchmarking: Comparison of R² Scores", fontsize=13, weight="bold")
    plt.ylim(0.80, 1.0)
    plt.ylabel("R² Score")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "model_benchmarking.png"), dpi=300)
    plt.close()

    return results_df


def tune_xgboost(
    X_train_proc: np.ndarray,
    X_test_proc: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    all_feature_names: List[str],
    reports_dir: str = "reports",
) -> XGBRegressor:
    """Optimize XGBoost via GridSearchCV and generate evaluation figures.

    Args:
        X_train_proc: Preprocessed training feature matrix
        X_test_proc: Preprocessed test feature matrix
        y_train: Training labels
        y_test: Testing labels
        all_feature_names: Full list of encoded column names
        reports_dir: Output directory for figures

    Returns:
        Best tuned XGBRegressor instance
    """
    print("\n--- Hyperparameter Optimization: XGBoost Regressor (3-Fold CV) ---")
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    }

    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_proc, y_train)

    best_xgb: XGBRegressor = grid_search.best_estimator_
    print(f"  Best Parameters: {grid_search.best_params_}")

    preds_tuned = best_xgb.predict(X_test_proc)
    r2_tuned = r2_score(y_test, preds_tuned)
    rmse_tuned = np.sqrt(mean_squared_error(y_test, preds_tuned))
    mae_tuned = mean_absolute_error(y_test, preds_tuned)

    print(f"  Tuned XGBoost R²: {r2_tuned:.4f}")
    print(f"  Tuned XGBoost MAE: ${mae_tuned:,.2f}")
    print(f"  Tuned XGBoost RMSE: ${rmse_tuned:,.2f}")

    # Cross-validation validation
    cv_scores = cross_val_score(best_xgb, X_train_proc, y_train, cv=3, scoring="r2")
    print(f"  Cross-Validation R² Folds: {cv_scores.round(4)}")
    print(f"  Mean CV R² Score: {cv_scores.mean():.4f}")

    # Figure: Actual vs. Predicted Prices
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds_tuned, alpha=0.6, color="crimson", edgecolors="w", s=40)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2, label="Perfect Fit (y = x)")
    plt.title("Actual vs. Predicted Sale Prices (Tuned XGBoost)", fontsize=13, weight="bold")
    plt.xlabel("Actual Sale Price ($)")
    plt.ylabel("Predicted Sale Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "actual_vs_predicted.png"), dpi=300)
    plt.close()

    # Figure: Top 10 Feature Importances
    plt.figure(figsize=(10, 6))
    importances = pd.Series(best_xgb.feature_importances_, index=all_feature_names)
    importances.nlargest(10).plot(kind="barh", color="darkorange")
    plt.title("Top 10 Influential Features for House Price Prediction (XGBoost)", fontsize=13, weight="bold")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "feature_importance.png"), dpi=300)
    plt.close()

    return best_xgb


def run_pipeline(
    data_path: str = "data/AmesHousing.csv",
    model_output_path: str = "models/house_price_pipeline.joblib",
    reports_dir: str = "reports",
) -> None:
    """Execute end-to-end training and export the serialized inference pipeline."""
    # Fallback to root dataset path if data/ doesn't exist
    if not os.path.exists(data_path) and os.path.exists("AmesHousing.csv"):
        data_path = "AmesHousing.csv"

    print(f"Loading Ames Housing dataset from '{data_path}'...")
    df = load_dataset(data_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} attributes.")

    # 1. EDA
    generate_eda_plots(df, target_col="SalePrice", reports_dir=reports_dir)

    # 2. Split and Preprocess
    (
        X_train,
        X_test,
        y_train,
        y_test,
        numerical_cols,
        categorical_cols,
        preprocessor,
    ) = prepare_data(df, target_column="SalePrice")

    print("\nFitting preprocessing ColumnTransformer...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Extract post-encoding feature names for interpretability
    encoded_cat = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
    all_feature_names = list(numerical_cols) + list(encoded_cat)
    print(f"Total processed feature dimensions: {X_train_proc.shape[1]}")

    # 3. Benchmark Models
    results_df = benchmark_models(X_train_proc, X_test_proc, y_train, y_test, reports_dir=reports_dir)
    results_df.to_csv(os.path.join(reports_dir, "benchmark_metrics.csv"), index=False)

    # 4. Hyperparameter Tuning
    best_xgb = tune_xgboost(X_train_proc, X_test_proc, y_train, y_test, all_feature_names, reports_dir=reports_dir)

    # 5. Export Full End-to-End Pipeline
    print(f"\nSerializing full production pipeline to '{model_output_path}'...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", best_xgb),
    ])
    joblib.dump(full_pipeline, model_output_path)
    print(f"[OK] Complete model pipeline successfully exported to '{model_output_path}'!")


if __name__ == "__main__":
    run_pipeline()
