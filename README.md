
# Product Rating Prediction App

An interactive **Streamlit application** that predicts product ratings based on product attributes, pricing, discounts, delivery options, promotional offers, and textual descriptions.

The application uses a trained **XGBoost Regressor** model to estimate the average customer rating of a product, helping users assess potential product performance before listing.

**Live Demo:** [Product Rating Prediction App](https://rating-prediction-app.streamlit.app)

**Source Code:** [GitHub Repository](https://github.com/PTToNhu/streamlit-rating-prediction)

## Table of Contents
1. [Application Overview](#application-overview)
2. [Machine Learning Model](#machine-learning-model)
3. [Model Evaluation](#model-evaluation)
4. [Running the Application](#running-the-application)
5. [Technologies Used](#technologies-used)

## Application Overview

The Streamlit application provides an interactive interface for predicting product ratings.

**Key Features**
- **Product Information Input:** Enter product attributes, including brand, category, pricing, discounts, product descriptions, delivery options, and promotional offers.
- **Automated Preprocessing:** Applies the same preprocessing and feature engineering pipeline used during model training.
- **Rating Prediction:** Generates an estimated product rating using the trained XGBoost Regressor.
- **Interactive Interface:** Allows users to submit product information and view prediction results directly through a web browser.

## Machine Learning Model

### Prediction Task

- **Problem Type:** Regression
- **Target Variable:** Average product rating
- **Algorithm:** XGBoost Regressor
- **Training Data:** `mart_product_info` Data Mart
- **Train/Test Split:** 80% training and 20% testing

### Data Preprocessing & Feature Engineering

Several preprocessing and feature engineering techniques were applied to transform raw product attributes into model-ready features.

| Feature | Technique |
|---|---|
| Brand | Target Encoding |
| Category | One-Hot Encoding + PCA (90% explained variance) |
| Product Images & Videos | Extracted image and video counts |
| Product Sizes | Extracted number of available sizes |
| Discounts | Binary encoding and discount value handling |
| Delivery Options | Extracted delivery-related features |
| Promotional Offers | Extracted pricing and offer indicators |
| Product Descriptions | SentenceTransformer (`all-MiniLM-L6-v2`) embeddings |

Textual information from product descriptions, specifications, and details was combined and converted into numerical embeddings using SentenceTransformer.

The resulting text embeddings were combined with numerical and binary features to construct the final input matrix.

### Model Configuration

The XGBoost Regressor was configured with the following parameters:

| Parameter | Value |
|---|---:|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 6 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `random_state` | 42 |

The trained model and preprocessing components were saved for reuse in the Streamlit application.

## Model Evaluation

The model was evaluated on the held-out test set using two regression metrics:

| Metric | Score |
|---|---:|
| RMSE | 0.5019 |
| MAE | 0.3981 |

- **MAE (Mean Absolute Error):** The model's predictions differ from actual ratings by approximately 0.3981 rating points on average.
- **RMSE (Root Mean Squared Error):** Measures prediction errors while penalizing larger deviations more heavily.

These results indicate that the model achieved relatively stable rating predictions on the test dataset.

## Running the Application

### 1. Clone the Repository

```bash
git clone https://github.com/PTToNhu/streamlit-rating-prediction.git
cd streamlit-rating-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available locally through the URL displayed in the terminal.

*Note: Ensure that the trained model and required preprocessing artifacts are available in the project directory before running the application.*

## Technologies Used

- **Python:** Data preprocessing, feature engineering, and model development
- **Streamlit:** Interactive web application
- **XGBoost:** Regression model for product rating prediction
- **Scikit-learn:** Data preprocessing and feature transformation
- **SentenceTransformers:** Text embedding generation
- **Pandas & NumPy:** Data manipulation and numerical computation
