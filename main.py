import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle

# Load data from Parquet files
@st.cache_data
def load_data():
    # Adjust paths as necessary
    train_df = pd.read_parquet('train.parquet')
    test_df = pd.read_parquet('test.parquet')
    val_df = pd.read_parquet('val.parquet')

    return train_df, test_df, val_df

# Train the recommendation model
@st.cache_data
def train_model(train_df):
    # Create a user-item matrix with ratings or purchase counts
    user_item_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    user_item_matrix_np = user_item_matrix.values

    # Apply Truncated SVD to the user-item matrix
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(user_item_matrix_np)

    # Save model components for future use
    with open('svd_model.pkl', 'wb') as f:
        pickle.dump((svd, user_item_matrix.index, user_item_matrix.columns), f)

    return user_item_matrix.index, user_item_matrix.columns

# Get top-N recommendations for a user
def get_recommendations(user_id, n_recommendations):
    # Load the pre-trained SVD model
    with open('svd_model.pkl', 'rb') as f:
        svd, user_ids, item_ids = pickle.load(f)

    # Reconstruct the predicted ratings matrix
    predicted_ratings = np.dot(svd.transform(user_ids), svd.components_)
    predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=item_ids)

    # Get top-N recommendations for the specified user
    if user_id in predicted_ratings_df.index:
        user_ratings = predicted_ratings_df.loc[user_id].sort_values(ascending=False).head(n_recommendations)
        return user_ratings
    else:
        return None

# Streamlit UI
st.title("eCommerce Product Recommendation System")
st.write("This recommendation system uses collaborative filtering (SVD) to recommend products to users.")

# Load data and train the model
train_df, test_df, val_df = load_data()
user_ids, item_ids = train_model(train_df)

# Input: User ID
selected_user = st.selectbox("Select a User", user_ids)
n_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

# Display recommendations
if st.button("Recommend Items"):
    recommendations = get_recommendations(selected_user, n_recommendations)
    if recommendations is not None:
        st.write(f"Top-{n_recommendations} Recommendations for User {selected_user}:")
        for item, rating in recommendations.items():
            st.write(f"Item ID: {item}, Predicted Rating: {rating:.2f}")
    else:
        st.write("User not found in the dataset.")
