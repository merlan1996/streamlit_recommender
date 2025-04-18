import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ------------------------------
# 🔁 Load data and models
# ------------------------------
df = pd.read_csv("merged_tripadvisor_accommodation_businesses.csv")
X = joblib.load("business_features_matrix.pkl")
kmeans = joblib.load("kmeans_model.pkl")

# ✅ Apply the same cleaning steps as in Colab
text_cols = ['hotel_style', 'description', 'amenities.property', 'amenities.room_types', 'amenities.room']
df[text_cols] = df[text_cols].fillna("")
df['rating.overall'] = df['rating.overall'].fillna(0)
df['rating.count'] = df['rating.count'].fillna(0)
df = df[df['address.province'].notna()]
for col in text_cols:
    df[col] = df[col].str.lower()

# ✅ Assign clusters
df['cluster'] = kmeans.predict(X)

# ------------------------------
# 📌 Dropdown selection
# ------------------------------
business_options = dict(zip(df['Company_Name'], df['uid']))
selected_name = st.selectbox("Choose a business", options=business_options.keys())
selected_uid = business_options[selected_name]

# ------------------------------
# 🔍 Similarity function
# ------------------------------
def recommend_similar_businesses(selected_uid, X, df_original, top_n=5):
    idx = df_original.index[df_original['uid'] == selected_uid].tolist()
    if not idx:
        return pd.DataFrame()
    idx = idx[0]

    cluster_id = df_original.loc[idx, 'cluster']
    cluster_mask = df_original['cluster'] == cluster_id
    df_cluster = df_original[cluster_mask].reset_index(drop=True)

    # ✅ FIXED: convert mask to NumPy boolean index for sparse matrix
    X_cluster = X[cluster_mask.to_numpy()]
    selected_vector = X[idx]
    similarity_scores = cosine_similarity(selected_vector, X_cluster).flatten()

    df_cluster = df_cluster.copy()
    df_cluster['similarity_score'] = similarity_scores
    df_cluster = df_cluster[df_cluster['uid'] != selected_uid]

    return df_cluster.sort_values(by='similarity_score', ascending=False).head(top_n)[[
        'Company_Name', 'address.city', 'address.province', 'rating.overall', 'similarity_score'
    ]]

# ------------------------------
# 📊 Show results
# ------------------------------
recommendations = recommend_similar_businesses(selected_uid, X, df, top_n=5)

if not recommendations.empty:
    recommendations['similarity_score'] = (recommendations['similarity_score'] * 100).round().astype(int).astype(str) + '%'
    recommendations = recommendations.rename(columns={
        'Company_Name': 'Business Name',
        'address.city': 'City',
        'address.province': 'Province',
        'rating.overall': 'Rating',
        'similarity_score': 'Similarity'
    })

    st.subheader("Top 5 Similar Businesses")
    st.dataframe(recommendations.reset_index(drop=True))
