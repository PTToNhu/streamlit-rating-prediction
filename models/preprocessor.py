# =========================
# IMPORT LIBRARIES
# =========================
import os

import os

import pandas as pd
import numpy as np
import ast
import json
import re
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import xgboost as xgb

# =========================
# HELPER FUNCTIONS
# =========================


def is_null(x):
    # None / NaN
    if x is None:
        return 1

    if isinstance(x, (float, np.floating)) and pd.isna(x):
        return 1

    # empty string
    if x == "":
        return 1

    # empty dict
    if x == "{}":
        return 1

    # [{}]
    if x == "[{}]":
        return 1

    return 0


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def n_components(df, col, encoder, variance_threshold=0.9):
    encoded = encoder.fit_transform(df[[col]])

    # Fit full PCA
    pca_full = PCA().fit(encoded)

    explained_var = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # tìm số components
    n_comps = np.argmax(cumulative_var >= variance_threshold) + 1

    print(f"Number of PCA components for {col}:", n_comps)

    # Fit PCA with optimal components
    pca_model = PCA(n_components=n_comps)
    pca_model.fit(encoded)

    return pca_model, encoded


def count_by_split_on_comma(x):
    if pd.isna(x) or x == "":
        return 0
    return len(x.split(','))


def count_object_in_array(x):
    if pd.isna(x) or x == "":
        return 0

    if isinstance(x, str):
        x = json.loads(x)

    return len(x)


def discount_percent(discount, apply_discount):
    if apply_discount == 1:
        return discount
    return 0


def extract_delivery_option(df):
    delivery_options = df['delivery_options']

    # convert string -> list
    parsed_options = delivery_options.apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    # đếm tổng các option
    counter = Counter()

    for options in parsed_options:
        counter.update(options)

    # convert thành dataframe
    option_count_df = (
        pd.DataFrame(
            counter.items(),
            columns=['delivery_option', 'count']
        )
        .sort_values(by='count', ascending=False)
        .reset_index(drop=True)
    )

    return option_count_df


def normalize_options(options):

    # null
    if options is None:
        return []

    # string list -> python list
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except:
            return []

    # đảm bảo list
    if not isinstance(options, list):
        return []

    return [str(x).lower().strip() for x in options]


def parse_delivery_options(options):
    options = normalize_options(options)

    features = {
        "original_product": 0,
        "pay_on_delivery": 0,
        "try_and_buy": 0,

        "returnable": 0,
        "exchangeable": 0,

        "return_days": 0,
        "exchange_days": 0,

        "restricted_return": 0,
        "size_exchange_only": 0,
    }

    text = " ".join(options).lower()

    # 100% original product
    if "original products" in text:
        features["original_product"] = 1

    # pay on delivery might be available
    if "pay on delivery might be available" in text:
        features["pay_on_delivery"] = 1
    # # pay on delivery is not available
    # if "pay on delivery is not available" in text:
    #     features["pay_on_delivery"] = 0

    # try & buy might be available
    if "try & buy might be available" in text:
        features["try_and_buy"] = 1

    # items like innerwear, socks, certain accessories and some high-value fragile items do not come under our return policy
    if "do not come under our return policy" in text:
        features["restricted_return"] = 1
        # features["returnable"] = 0
        # features["exchangeable"] = 0

    # this item is only returnable and not exchangeable
    if "only returnable and not exchangeable" in text:
        features["returnable"] = 1
        # features["exchangeable"] = 0

    # 8	this item is only exchangeable for the same or a different size, if available, and cannot be returned
    if "this item is only exchangeable for the same or a different size, if available, and cannot be returned" in text:
        features["size_exchange_only"] = 1
        # features["returnable"] = 0
        features["exchangeable"] = 1

    # extract days
    # easy 14 days returns and exchanges
    # easy 7 days returns and exchanges	161
    # easy 14 days returns
    # easy 14 days exchanges
    # easy 21 days returns and exchanges
    # easy 2 days returns and exchanges
    # easy 30 days returns and exchanges
    # easy 15 days returns and exchanges
    # easy 10 days returns and exchanges
    match = re.findall(r'(\d+)\s*days', text)

    if match:
        days = max(map(int, match))

        if "returns and exchanges" in text:
            features["returnable"] = 1
            features["return_days"] = days
            features["exchangeable"] = 1
            features["exchange_days"] = days

        elif "easy 14 days returns" in text:
            features["returnable"] = 1
            features["return_days"] = 14
        elif "easy 14 days exchanges" in text:
            features["exchangeable"] = 1
            features["exchange_days"] = 14

    return features


############################################################
# BEST OFFER
############################################################


def extract_best_offer(
    df,
    final_price_col,
    best_offer_col,
    target_col='best_price'
):

    def parse_best_price(value, final_price):

        # null
        if pd.isna(value):
            return np.nan, 0

        # string -> dict
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except:
                return np.nan, 0

        # phải là dict
        if not isinstance(value, dict):
            return np.nan, 0

        # empty dict
        if len(value) == 0:
            return float(final_price), 0

        # có best offer
        has_best_offer = 1

        # không có best_price
        if 'best_price' not in value:
            return float(final_price), has_best_offer

        best_price = value.get('best_price')

        # convert numeric
        try:
            best_price = str(best_price).replace(',', '').strip()
            return float(best_price), has_best_offer

        except:
            return np.nan, has_best_offer

    parsed = df.apply(
        lambda row: parse_best_price(
            row[best_offer_col],
            row[final_price_col]
        ),
        axis=1
    )

    df[target_col] = parsed.apply(lambda x: x[0])
    df['best_offer_discount_amount'] = parsed.apply(lambda x: x[1])

    return df


############################################################
# OFFER FEATURES
############################################################


def count_offer_from_more_offers(df, col='more_offers'):

    counter = Counter()

    offer_counts_per_row = []

    for value in df[col]:

        # null
        if pd.isna(value):
            offer_counts_per_row.append(0)
            continue

        # string -> python object
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except:
                offer_counts_per_row.append(0)
                continue

        # phải là list
        if not isinstance(value, list):
            offer_counts_per_row.append(0)
            continue

        offer_counts_per_row.append(len(value))

        for item in value:
            if isinstance(item, dict):
                title = item.get('title', '')
                counter.update([title])

    return counter, offer_counts_per_row


def extract_offer_features(df, col='more_offers'):

    features = []

    for value in df[col]:

        row = {
            'emi_available': 0,
            'citi_offer': 0,
            'indusind_offer': 0,
            'cashback_offer': 0,
            'yes_bank_offer': 0,
            'myntra_kotak_offer': 0,
        }

        # null
        if pd.isna(value):
            features.append(row)
            continue

        # string -> list
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except:
                features.append(row)
                continue

        # phải là list
        if not isinstance(value, list):
            features.append(row)
            continue

        texts = []

        for item in value:
            if isinstance(item, dict):
                texts.append(str(item.get('title', '')).lower())

        full_text = ' '.join(texts)

        if 'emi' in full_text:
            row['emi_available'] = 1

        if 'citi' in full_text:
            row['citi_offer'] = 1

        if 'indusind' in full_text:
            row['indusind_offer'] = 1

        if 'cashback' in full_text:
            row['cashback_offer'] = 1

        if 'yes bank' in full_text:
            row['yes_bank_offer'] = 1

        if 'kotak' in full_text:
            row['myntra_kotak_offer'] = 1

        features.append(row)

    features_df = pd.DataFrame(features)

    return pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1
    )


############################################################
# DROP COLUMNS
############################################################


def drop_cols(df, cols):
    return df.drop(columns=cols, errors='ignore')


def build_text(row):
    return f"""
    Description: {row['product_description']}
    Specifications: {row['product_specifications']}
    Details: {row['product_details']}
    """

############################################################
# MAIN PREPROCESSOR
############################################################


class EcommercePreprocessor:

    def __init__(self, device='cpu'):

        self.device = device
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )

        self.brand_encoder = TargetEncoder(
            smooth=True,
            target_type='continuous'
        )

        self.category_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )

        self.category_pca = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('embedder', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=getattr(self, 'device', 'cpu')
        )

    ########################################################
    # BUILD TEXT
    ########################################################

    def build_text(self, row):
        return f"""
      Description: {row['product_description']}
      Specifications: {row['product_specifications']}
      Details: {row['product_details']}
      """

    ########################################################
    # FIT
    ########################################################

    def fit(self, df, y):

        ####################################################
        # BRAND TARGET ENCODING
        ####################################################

        self.brand_encoder.fit(
            df[['brand']],
            y
        )

        ####################################################
        # CATEGORY PCA
        ####################################################

        category_encoded = (
            self.category_encoder
            .fit_transform(df[['category']])
        )

        pca_full = PCA().fit(category_encoded)

        cumulative_var = np.cumsum(
            pca_full.explained_variance_ratio_
        )

        n_components = (
            np.argmax(cumulative_var >= 0.9) + 1
        )

        print(
            f'Category PCA components: {n_components}'
        )

        self.category_pca = PCA(
            n_components=n_components
        )

        self.category_pca.fit(category_encoded)

        return self

    ########################################################
    # TRANSFORM
    ########################################################

    def transform(self, df):

        df = df.copy()

        ####################################################
        # BRAND ENCODING
        ####################################################

        df['brand_encoded'] = (
            self.brand_encoder
            .transform(df[['brand']])
        )

        ####################################################
        # IMAGE / VIDEO / SIZE FEATURES
        ####################################################

        df['images_count'] = (
            df['image_url']
            .apply(count_by_split_on_comma)
        )

        df['videos_count'] = (
            df['video_url']
            .apply(count_by_split_on_comma)
        )

        df['sizes_count'] = (
            df['sizes']
            .apply(count_object_in_array)
        )

        ####################################################
        # CATEGORY PCA
        ####################################################

        category_encoded = (
            self.category_encoder
            .transform(df[['category']])
        )

        category_pca = (
            self.category_pca
            .transform(category_encoded)
        )

        pca_cols = [
            f'category_pca_{i}'
            for i in range(category_pca.shape[1])
        ]

        pca_df = pd.DataFrame(
            category_pca,
            columns=pca_cols,
            index=df.index
        )

        df = pd.concat([df, pca_df], axis=1)

        ####################################################
        # APPLY DISCOUNT
        ####################################################

        df['apply_discount'] = (
            df['apply_discount']
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                'true': 1,
                'false': 0
            })
            .fillna(0)
            .astype(int)
        )

        df['discount'] = df['discount'].where(
            df['apply_discount'] == 1,
            0
        )

        ####################################################
        # DELIVERY FEATURES
        ####################################################

        delivery_features = (
            df['delivery_options']
            .apply(parse_delivery_options)
        )

        delivery_df = pd.DataFrame(
            delivery_features.tolist(),
            index=df.index
        )

        df = pd.concat([df, delivery_df], axis=1)

        ####################################################
        # BEST OFFER
        ####################################################

        df = extract_best_offer(
            df,
            final_price_col='final_price',
            best_offer_col='best_offer'
        )

        ####################################################
        # MORE OFFERS
        ####################################################

        df = extract_offer_features(df)

        ####################################################
        # TEXT EMBEDDINGS
        ####################################################
        print(df.columns.tolist())
        texts = df.apply(
            self.build_text,
            axis=1
        ).tolist()

        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True
        )

        ####################################################
        # DROP UNUSED COLS
        ####################################################

        drop_columns = [
            'product_id',
            'brand',
            'image_url',
            'video_url',
            'sizes',
            'breadcrumbs',
            'category',
            'seller_index',
            'currency_index',
            'delivery_options',
            'best_offer',
            'more_offers',
            'what_customers_said',
            'product_description',
            'product_specifications',
            'product_details',
        ]

        df = drop_cols(df, drop_columns)

        ####################################################
        # NUMERIC FEATURES
        ####################################################

        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        numeric_features = numeric_df.values

        ####################################################
        # FINAL FEATURES
        ####################################################

        X = np.hstack([
            embeddings,
            numeric_features
        ])

        return X
