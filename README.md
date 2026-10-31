#  Ames Housing Price Prediction & Real Estate Valuation Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning pipeline and interactive web application designed to forecast residential property valuations in Ames, Iowa using **80+ structural, spatial, and environmental features**.

The system features **automated leak-free preprocessing pipelines**, comprehensive **regression model benchmarking**, **GridSearchCV hyperparameter optimization**, and an interactive **Streamlit dashboard** for real-time property valuation.

---

##  Project Highlights

- **Zero Data Leakage Architecture**: Engineered modular `scikit-learn` `ColumnTransformer` pipelines ensuring preprocessing transformations (median/constant imputation, standard scaling, and one-hot encoding) are fitted strictly on training folds.
- **Robust Model Benchmarking**: Evaluated parametric and non-parametric architectures (Multiple Linear Regression, Random Forest Regressor, and XGBoost Regressor).
- **Hyperparameter Optimization**: Tuned tree complexity, learning rate, and estimator counts using 3-fold cross-validated grid search, achieving an **$R^2$ score of 0.9074** and **MAE of $15,005** on unseen test data.
- **Production Artifact Serialization**: Exported full inference pipeline (`Pipeline(steps=[('preprocessor', ...), ('regressor', ...)])`) with `joblib` for instant batch and single-record predictions.
- **Interactive Valuation App**: Deployed a modern Streamlit application allowing users to adjust living area, finish quality, garage capacity, and neighborhood to obtain real-time valuation estimates and confidence intervals.

---

##  Architecture & Workflow

```mermaid
flowchart TD
    A["Raw Ames Housing Dataset<br/>(2,930 Records, 82 Features)"] --> B["Data Cleaning & ID Removal<br/>(Drop 'Order', 'PID')"]
    B --> C["Train-Test Partitioning<br/>(80% Train / 20% Test)"]

    subgraph Preprocessing ["scikit-learn ColumnTransformer"]
        D1["Numeric Attributes (36)"] --> E1["SimpleImputer (Median)"] --> F1["StandardScaler"]
        D2["Categorical Attributes (43)"] --> E2["SimpleImputer ('Missing')"] --> F2["OneHotEncoder (handle_unknown='ignore')"]
    end

    C --> Preprocessing
    Preprocessing --> G["Processed Feature Matrix<br/>(315 Post-Encoding Dimensions)"]

    subgraph Benchmarking ["Model Benchmarking & Tuning"]
        G --> M1["Multiple Linear Regression<br/>(R² = 0.8943)"]
        G --> M2["Random Forest Regressor<br/>(R² = 0.9107)"]
        G --> M3["XGBoost Regressor (Baseline)<br/>(R² = 0.9063)"]
        M3 --> H["GridSearchCV (3-Fold CV)<br/>Hyperparameter Tuning"]
        H --> M4["Optimized XGBoost Model<br/>(R² = 0.9074, MAE = $15,005)"]
    end

    M4 --> I["Export Pipeline Artifact<br/>('models/house_price_pipeline.joblib')"]
    I --> J["Streamlit Interactive Web UI<br/>('app.py')"]
```

---

##  Model Performance Benchmarking

All models were evaluated on the same held-out 20% test dataset ($N = 586$ properties):

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | 3-Fold Cross-Val $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** | 0.8943 | \$15,674.41 | \$29,110.19 | 0.8840 |
| **Random Forest Regressor** | 0.9107 | \$15,767.41 | \$26,759.52 | 0.8970 |
| **XGBoost Regressor (Baseline)** | 0.9063 | \$15,320.39 | \$27,404.86 | 0.9028 |
| **XGBoost Regressor (Tuned)**  | **0.9074** | **\$15,005.00** | **\$27,246.54** | **0.9059** |

### Optimal Hyperparameters (GridSearchCV)
- `n_estimators`: `200`
- `learning_rate`: `0.1`
- `max_depth`: `3`

---

##  Key Feature Drivers

Feature importance weights extracted from the optimized XGBoost model indicate that real estate valuations are primarily driven by:

1. **Overall Material & Finish Quality (`Overall Qual`)**: Strongest positive predictor; properties rated 8–10 command disproportionate valuation premiums.
2. **Above-Grade Living Area (`Gr Liv Area`)**: Directly scales price per square foot.
3. **Basement Finish Quality (`Bsmt Qual_Ex`)**: High-finish basements provide significant market uplift.
4. **Garage Capacity (`Garage Cars` & `Garage Area`)**: 2+ car garages reflect high-demand residential standards.
5. **Kitchen Quality (`Kitchen Qual_Ex`)**: Modern, high-end kitchens serve as major appraisal drivers.

---

##  Repository Structure

```text
├── app.py                     # Streamlit interactive valuation application
├── requirements.txt           # Environment dependencies
├── .gitignore                 # Standard git ignore rules
├── README.md                  # Comprehensive project documentation
├── data/
│   └── AmesHousing.csv        # Ames Housing dataset (2,930 records)
├── models/
│   └── house_price_pipeline.joblib  # Serialized end-to-end production pipeline
├── notebooks/
│   └── exploratory_data_analysis.ipynb  # Documented Jupyter Notebook walkthrough
├── reports/
│   ├── actual_vs_predicted.png    # Test set predictions scatter plot
│   ├── benchmark_metrics.csv      # Quantitative evaluation results table
│   ├── correlation_heatmap.png    # Top numeric feature correlation matrix
│   ├── feature_importance.png     # Top 10 XGBoost feature importances
│   ├── living_area_vs_price.png   # Scatter relationship plot
│   ├── model_benchmarking.png     # Algorithm R² comparison bar chart
│   └── target_distribution.png    # SalePrice distribution plot
└── src/
    ├── __init__.py
    ├── preprocess.py          # ColumnTransformer pipelines and data loaders
    ├── train.py               # Training, benchmarking, and artifact export script
    └── predict.py             # Inference functions for programmatic prediction
```

---

##  Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/priyal-vyas/ames-house-price-prediction.git
cd ames-house-price-prediction
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training & Evaluation
To regenerate the full pipeline, benchmark tables, and visualization plots:
```bash
python src/train.py
```

### 5. Launch the Streamlit Interactive App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test interactive house price predictions.

### 6. Programmatic Inference Example
```python
from src.predict import load_model_pipeline, predict_price

# Load production pipeline
pipeline = load_model_pipeline("models/house_price_pipeline.joblib")

# Single property feature dictionary
sample_property = {
    "Overall Qual": 8,
    "Gr Liv Area": 2100,
    "Total Bsmt SF": 1200,
    "Garage Cars": 2,
    "Year Built": 2005,
    "Neighborhood": "CollgCr",
    "Kitchen Qual": "Gd",
}

estimated_price = predict_price(sample_property, pipeline=pipeline)
print(f"Estimated Market Price: ${estimated_price:,.2f}")
```

---

##  Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: Scikit-Learn, XGBoost
- **Data Engineering & Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Application & Deployment**: Streamlit
- **Model Persistence**: Joblib

---

##  License & Citation

This project is released under the [MIT License](LICENSE).  
Dataset provided by Dean De Cock (Truman State University), *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*.
