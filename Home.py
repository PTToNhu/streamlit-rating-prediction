import streamlit as st

# Set the favicon
st.set_page_config(page_title="Product Rating Prediction System", page_icon="🏠")

# Home page function
@st.cache_resource
def home_page():
    image_filename = "home.png"
    # home_image_path = os.path.join("images", image_filename)
    # st.image(home_image_path, width='stretch')

    st.markdown("## **This is a Streamlit app for predicting whether a product will receive a good rating.**")    
    st.write("**Features:**")
    st.write("- Predict product ratings based on product information, pricing, reviews,...")
    st.write("- Provide interactive visualizations and insights for product rating prediction results.")

    st.write("**Benefits:**")
    st.write("- Helps sellers and businesses estimate product ratings before launching products.")
    st.write("- Identifies factors that influence customer satisfaction and product performance.")
    st.write("- Streamlines rating prediction with an easy-to-use Streamlit interface.")

    st.write("**Dataset Features:**")
    st.write("- Product information: title, description, category, specifications, and details.")
    st.write("- Pricing features: initial price, final price, discount percentage, and offers.")
    st.write("- Customer feedback: ratings count, star distribution, and customer reviews.")
    st.write("- Seller and delivery information: seller details, delivery options, and product variations.")
    st.write("- Multimedia attributes: product images and video availability.")

    st.write("**Machine Learning Integrations:**")
    st.write("- Utilizes SentenceTransformer (`all-MiniLM-L6-v2`) for text embedding generation.")
    st.write("- Combines text embeddings with numerical product features for prediction.")
    st.write("- Uses XGBoost Regressor for product rating prediction.")
    st.write("- Supports scalable machine learning workflows for e-commerce analytics.")

# Main function
def main():

    st.title("Product Rating Prediction System")
    # st.image("images/login.png", width=250)

    # Display the home page content
    home_page()

if __name__ == "__main__":
    main()
