import __main__
import streamlit as st
import pandas as pd
import joblib
import torch
import json

from models.preprocessor import EcommercePreprocessor

# =========================
# FIX TORCH CPU
# =========================

if not torch.cuda.is_available():
    _torch_load = torch.load

    def _torch_load_cpu(*args, **kwargs):
        kwargs.setdefault('map_location', 'cpu')
        return _torch_load(*args, **kwargs)

    torch.load = _torch_load_cpu

__main__.EcommercePreprocessor = EcommercePreprocessor

# =========================
# LOAD MODEL
# =========================

model = joblib.load(r"models\xgb_model.pkl")
preprocessor = joblib.load(r"models\preprocessor.pkl")
df = pd.read_excel(r"dataset\mart_product_info.xlsx")

# =========================
# TITLE
# =========================

st.title("Product Rating Prediction")

st.markdown("## Product Information")

# =========================
# INPUT FIELDS
# =========================

brand = st.selectbox(
    "Brand",
    df["brand"].dropna().unique().tolist(),
    accept_new_options=True
)

category = st.selectbox(
    "Category",
    df["category"].dropna().unique().tolist(),
    accept_new_options=True
)

image_url = st.text_input(
    "Image URL",
    value="http://assets.myntassets.com/assets/images/8376765/2019/2/14/c93b7d05-6311-497e-ad00-7395c266a9771550142767008-Lino-Perros-Women-Navy-Blue-Solid-Backpack-3271550142765402-1.jpg,http://assets.myntassets.com/assets/images/8376765/2019/2/14/3d5659ce-855a-4e08-9880-970866c824a31550142766980-Lino-Perros-Women-Navy-Blue-Solid-Backpack-3271550142765402-2.jpg,http://assets.myntassets.com/assets/images/8376765/2019/2/14/6a75cfc6-b0a0-4a7a-9561-cd425d3314f31550142766954-Lino-Perros-Women-Navy-Blue-Solid-Backpack-3271550142765402-3.jpg,http://assets.myntassets.com/assets/images/8376765/2019/2/14/6e5b0338-2242-4f64-8fcf-2fc2e7fff5851550142766927-Lino-Perros-Women-Navy-Blue-Solid-Backpack-3271550142765402-4.jpg,http://assets.myntassets.com/assets/images/8376765/2019/2/14/4ccfa2c4-f356-464b-94fd-b48633ab171e1550142766902-Lino-Perros-Women-Navy-Blue-Solid-Backpack-3271550142765402-5.jpg,http://assets.myntassets.com/assets/images/8376765/2019/2/14/0db3b6e8-676d-4459-8e4a-7c8b86ce401f1550142766879-Lino-Perros-Women-Navy-Blue-Solid-Backpack-3271550142765402-6.jpg"
)

video_url = st.text_input(
    "Video URL",
    value="rw-9136281_cae700"
)

product_description = st.text_area(
    "Product Description",
    value="Women Navy Blue Solid Backpack"
)

product_specifications = st.text_area(
    "Product Specifications",
    value="Foam sole, breathable fabric"
)

product_details = st.text_area(
    "Product Details",
    value="Navy Blue solid backpackNon-Padded haul loop1 main compartment with flap and button closureNon-Padded backPadded shoulder strap: Non-PaddedWater-resistance: No"
)

sizes = st.multiselect(
    "Sizes",
    ["XS", "S", "M", "L", "XL", "XXL"],
    default=["S", "M", "L"]
)

initial_price = st.number_input(
    "Initial Price",
    min_value=0.0,
    value=150.0
)

discount = st.slider(
    "Discount (%)",
    min_value=0,
    max_value=100,
    value=20
)

delivery_options = st.multiselect(
    "Delivery Options",
    [
        r"100% original products",
        r"pay on delivery might be available",
        r"try & buy might be available",
        r"easy 14 days returns and exchanges",
        r"easy 7 days returns and exchanges",
        r"items like innerwear, socks, certain accessories and some high-value fragile items do not come under our return policy",
        r"this item is only returnable and not exchangeable",
        r"easy 14 days returns",
        r"easy 14 days exchanges",
        r"easy 21 days returns and exchanges",
        r"easy 2 days returns and exchanges",
        r"easy 30 days returns and exchanges"
        r"pay on delivery is not available",
        r"easy 15 days returns and exchanges",
        r"easy 10 days returns and exchanges"
    ]
)

best_offer = st.text_input(
    "Best Offer",
    value=""
)

more_offers = st.text_input(
    "More Offers",
    value=""
)

# =========================
# PREDICT
# =========================

if st.button("Predict Rating"):

    input_df = pd.DataFrame({
        "brand": [brand],
        "category": [category],
        "product_description": [product_description],
        "product_specifications": [product_specifications],
        "product_details": [product_details],
        "image_url": [image_url],
        "video_url": [video_url],

        "sizes": [
            json.dumps(
                [{"size": s} for s in sizes]
            )
        ],
        "breadcrumb": [""],
        "seller_index": [0],
        "currency_index": [0],
        "initial_price": [initial_price],
        "final_price": [initial_price - (initial_price * discount / 100)],

        "discount": [discount],
        "apply_discount": ["TRUE" if discount > 0 else "FALSE"],

        "ratings_count": [0],

        "delivery_options": [
            json.dumps(delivery_options)
        ],
        "best_offer": [json.dumps(best_offer)],
        "more_offers": [json.dumps(more_offers)],
        "what_customers_said": [""]
    })

    # =========================
    # PREPROCESS
    # =========================

    X_processed = preprocessor.transform(input_df)

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(X_processed)[0]

    # =========================
    # OUTPUT
    # =========================

    st.subheader("Prediction Result")

    st.success(f"Predicted Rating: {prediction:.2f}")

    if prediction >= 4:
        st.success("High Rating Product")
    elif prediction >= 3:
        st.info("Average Rating Product")
    else:
        st.warning("Low Rating Product")

    st.markdown("## Input Data")
    st.dataframe(input_df)
