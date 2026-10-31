"""Data preprocessing and transformation pipeline for Ames Housing dataset."""

from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the Ames Housing CSV dataset and clean identifier columns.

    Args:
        filepath: Path to AmesHousing.csv

    Returns:
        Cleaned pandas DataFrame
    """
    df = pd.read_csv(filepath)
    # Drop arbitrary identifier columns if present
    id_cols = [c for c in ["Order", "PID"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
    return df


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Segregate numeric and categorical column names.

    Args:
        X: Feature matrix DataFrame

    Returns:
        Tuple of (numerical_column_names, categorical_column_names)
    """
    numerical_cols = list(X.select_dtypes(include=["int64", "float64"]).columns)
    categorical_cols = list(
        X.select_dtypes(include=["object", "category", "string"]).columns
    )
    return numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for preprocessing.

    Numeric:
        - Median imputation for missing numeric values
        - StandardScaler for zero-mean, unit-variance scaling

    Categorical:
        - Constant 'Missing' imputation for missing nominal values
        - OneHotEncoder with handle_unknown='ignore' to prevent unseen test crashes

    Args:
        numerical_cols: List of numeric column names
        categorical_cols: List of categorical column names

    Returns:
        Configured ColumnTransformer
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame, target_column: str = "SalePrice", test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str], ColumnTransformer]:
    """Split data into train and test sets and instantiate the preprocessor.

    Args:
        df: Input DataFrame
        target_column: Target regression variable
        test_size: Fraction of dataset for testing
        random_state: Seed for reproducibility

    Returns:
        (X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, preprocessor)
    """
    target = target_column if target_column in df.columns else df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    numerical_cols, categorical_cols = get_feature_types(X)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols, preprocessor
