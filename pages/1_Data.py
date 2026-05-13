import __main__
import pandas as pd
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

if not torch.cuda.is_available():
    _torch_load = torch.load

    def _torch_load_cpu(*args, **kwargs):
        kwargs.setdefault('map_location', 'cpu')
        return _torch_load(*args, **kwargs)
    torch.load = _torch_load_cpu

# =========================================
# IMPORT CUSTOM PREPROCESSOR CLASS
# =========================================
from models.preprocessor import EcommercePreprocessor
__main__.EcommercePreprocessor = EcommercePreprocessor

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Ecommerce Rating Prediction",
    layout="wide"
)

# =========================================
# LOAD DATASET
# =========================================
current_dir = os.getcwd()

dataset_path = os.path.join(
    current_dir,
    "dataset",
    "mart_product_info.xlsx"
)

df = pd.read_excel(dataset_path)

# =========================================
# PAGE TITLE
# =========================================
st.title("Ecommerce Product Rating Prediction")

# =========================================
# DATASET SOURCE
# =========================================
st.header("1. Dataset Source")

st.write("""
Source:
- Ecommerce Shopping Dataset from Kaggle

Dataset Link:
- https://www.kaggle.com/datasets/anvitkumar/shopping-dataset

Description:
- The dataset contains ecommerce product information collected
from online marketplace platforms, including:
    - product descriptions
    - pricing
    - discounts
    - seller information
    - customer ratings
    - offers
    - delivery information
""")

# =========================================
# DATASET DESCRIPTION
# =========================================
st.header("2. Dataset Description")

dataset_description = pd.DataFrame({
    "Column Name": [
        "product_id",
        "brand",
        "product_description",
        "product_specifications",
        "product_details",
        "category",
        "initial_price",
        "discount",
        "final_price",
        "ratings_count",
        "rating"
    ],
    "Description": [
        "Unique product identifier",
        "Product brand",
        "Detailed product description",
        "Product technical specifications",
        "Additional product details",
        "Product category",
        "Original product price",
        "Discount percentage",
        "Final discounted price",
        "Total number of ratings",
        "Target product rating"
    ]
})

dataset_description.index = range(
    1,
    len(dataset_description) + 1
)

st.dataframe(
    dataset_description,
    use_container_width=True
)

# =========================================
# DATASET OVERVIEW
# =========================================
st.header("3. Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Number of Records",
        len(df)
    )

with col2:
    st.metric(
        "Number of Columns",
        len(df.columns)
    )

st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# =========================================
# MISSING VALUES
# =========================================
st.subheader("Missing Values")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": [
        df[col].isnull().sum()
        for col in df.columns
    ]
})

st.dataframe(
    missing_df,
    use_container_width=True
)

# =========================================
# DUPLICATES
# =========================================
st.subheader("Duplicate Records")

duplicate_count = df.duplicated().sum()

st.write(
    f"Number of duplicate rows: {duplicate_count}"
)

# =========================================
# PREPROCESSING
# =========================================
st.header("4. Data Preprocessing")

st.write("""
The preprocessing pipeline includes:

- Text cleaning
- Feature engineering
- Sentence embeddings
- PCA dimensionality reduction
- Numerical feature extraction
- Offer feature extraction
- Feature scaling
""")

st.subheader("Example Text Cleaning")

st.code("""
def clean_text(x):

    if pd.isna(x):
        return ""

    return str(x).strip().lower()
""", language="python")

# =========================================
# TEXT EMBEDDING
# =========================================
st.subheader("Text Embedding")

st.write("""
Text information from multiple columns was combined
and converted into dense vector embeddings using:

- SentenceTransformer
- all-MiniLM-L6-v2
""")

st.code("""
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    'all-MiniLM-L6-v2'
)
""", language="python")

# =========================================
# FEATURE ENGINEERING
# =========================================
st.subheader("Feature Engineering")

st.write("""
Additional engineered features include:

- image count
- video count
- size count
- offer-related features
- discount flags
- delivery options
""")

st.code("""
offer_features = extract_offer_features(df)

df = pd.concat(
    [df, offer_features],
    axis=1
)
""", language="python")

# =========================================
# TRAIN TEST SPLIT
# =========================================
st.header("5. Train/Test Split")

st.code("""
X_df = df.drop(columns=['rating'])

y = df['rating']

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.2,
    random_state=42
)
""", language="python")

# =========================================
# LOAD MODEL
# =========================================
st.header("6. Load Trained Model")

model_path = os.path.join(
    current_dir,
    "models",
    "xgb_model.pkl"
)

preprocessor_path = os.path.join(
    current_dir,
    "models",
    "preprocessor.pkl"
)

with st.spinner("Loading model and preprocessor..."):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

st.success("Model and preprocessor loaded successfully.")

# =========================================
# DATA PREPARATION
# =========================================
st.header("7. Prepare Test Data")

progress_text = st.empty()
progress_bar = st.progress(0)

# =========================================
# SPLIT FEATURES/TARGET
# =========================================
progress_text.text("Splitting features and target...")

X_df = df.drop(columns=['rating'])
y = df['rating']

progress_bar.progress(25)

# =========================================
# TRAIN TEST SPLIT
# =========================================
progress_text.text("Creating train/test split...")

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df,
    y,
    test_size=0.2,
    random_state=42
)

progress_bar.progress(60)

# =========================================
# TRANSFORM FEATURES
# =========================================
progress_text.text("Transforming test features...")

X_test = preprocessor.transform(X_test_df)

progress_bar.progress(100)

progress_text.text("Data preparation completed.")

st.success("Test data prepared successfully.")

# =========================================
# PREDICTION
# =========================================
st.header("8. Model Prediction")

preds = model.predict(X_test)

prediction_df = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": preds
})

st.dataframe(
    prediction_df.head(20),
    use_container_width=True
)

# =========================================
# EVALUATION
# =========================================
st.header("9. Model Evaluation")

mse = mean_squared_error(
    y_test,
    preds
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    preds
)

# r2 = r2_score(
#     y_test,
#     preds
# )

metric_df = pd.DataFrame({
    "Metric": [
        "RMSE",
        "MAE",
        # "R² Score"
    ],
    "Value": [
        rmse,
        mae,
        # r2
    ]
})

st.dataframe(
    metric_df,
    use_container_width=True
)

# =========================================
# METRIC INTERPRETATION
# =========================================
st.subheader("Metric Interpretation")

st.write(f"""
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
""")

st.write("""
Interpretation:

- Lower RMSE indicates fewer large prediction errors.
- Lower MAE indicates better average prediction accuracy.
""")

# # =========================================
# # VISUALIZATION
# # =========================================
# st.header("9. Visualization")

# # =========================================
# # ACTUAL VS PREDICTED
# # =========================================
# st.subheader("Actual vs Predicted Ratings")

# fig, ax = plt.subplots(
#     figsize=(7, 5)
# )

# ax.scatter(
#     y_test,
#     preds,
#     alpha=0.6
# )

# ax.set_xlabel("Actual Ratings")
# ax.set_ylabel("Predicted Ratings")
# ax.set_title("Actual vs Predicted Ratings")

# st.pyplot(fig)

# # =========================================
# # RESIDUAL DISTRIBUTION
# # =========================================
# st.subheader("Residual Distribution")

# residuals = y_test - preds

# fig2, ax2 = plt.subplots(
#     figsize=(7, 5)
# )

# ax2.hist(
#     residuals,
#     bins=30
# )

# ax2.set_xlabel("Residual Error")
# ax2.set_ylabel("Frequency")
# ax2.set_title("Residual Distribution")

# st.pyplot(fig2)

# # =========================================
# # FEATURE IMPORTANCE
# # =========================================
st.header("10. Suggested Improvements")

st.write("""
Possible future improvements:

- Add sentiment analysis from reviews
- Add image embeddings
- Hyperparameter tuning
- Use CatBoost or LightGBM
- Add customer behavior features
- Add seller reputation features
""")
