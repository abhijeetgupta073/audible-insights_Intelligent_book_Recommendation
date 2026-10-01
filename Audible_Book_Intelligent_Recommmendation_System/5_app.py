# ============================================================================
# STEP 5: STREAMLIT WEB APP WITH RECOMMENDATIONS
# Beautiful UI for book recommendations
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎧 Audible Insights",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD CACHED DATA (Only loads once, then reused)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all pickle files (cached for speed)"""
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('similarity_matrix.pkl', 'rb') as f:
        similarity_matrix = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('books_data_with_clusters.pkl', 'rb') as f:
        df = pickle.load(f)
    return vectorizer, similarity_matrix, kmeans_model, df

# Load all models
vectorizer, similarity_matrix, kmeans_model, df = load_models()

# ============================================================================
# RECOMMENDATION ALGORITHMS
# ============================================================================

def content_based_recommendation(book_name, n_recommendations=5):
    """
    Algorithm 1: Content-Based Filtering
    Finds similar books using similarity matrix
    """
    try:
        book_idx = df[df['Book Name'].str.contains(book_name, case=False, na=False)].index[0]
    except IndexError:
        return None, f"Book '{book_name}' not found"
    
    # Get similarity scores
    similarities = similarity_matrix[book_idx]
    
    # Get top N (excluding the book itself)
    top_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
    
    recommendations = df.iloc[top_indices][['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price']].copy()
    recommendations['Similarity Score'] = similarities[top_indices]
    recommendations['Algorithm'] = 'Content-Based'
    
    return recommendations, None


def clustering_based_recommendation(book_name, n_recommendations=5):
    """
    Algorithm 2: Clustering-Based Filtering
    Finds books in same cluster, sorted by rating
    """
    try:
        book_idx = df[df['Book Name'].str.contains(book_name, case=False, na=False)].index[0]
    except IndexError:
        return None, f"Book '{book_name}' not found"
    
    book_cluster = df.iloc[book_idx]['Cluster']
    
    # Get all books in same cluster
    cluster_books = df[df['Cluster'] == book_cluster].copy()
    
    # Sort by rating
    cluster_books = cluster_books.sort_values('Rating', ascending=False)
    
    # Remove the original book
    cluster_books = cluster_books[cluster_books.index != book_idx]
    
    recommendations = cluster_books.head(n_recommendations)[['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price']].copy()
    recommendations['Similarity Score'] = 0.0  # Dummy for display
    recommendations['Algorithm'] = 'Clustering-Based'
    
    return recommendations, None


def hybrid_recommendation(book_name, n_recommendations=5):
    """
    Algorithm 3: Hybrid Recommendation (BEST!)
    Combines: 40% similarity + 30% rating + 20% popularity + 10% cluster bonus
    """
    try:
        book_idx = df[df['Book Name'].str.contains(book_name, case=False, na=False)].index[0]
    except IndexError:
        return None, f"Book '{book_name}' not found"
    
    book_cluster = df.iloc[book_idx]['Cluster']
    
    # Component 1: Similarity (40% weight)
    similarities = similarity_matrix[book_idx]
    sim_scores = 0.4 * similarities
    
    # Component 2: Rating (30% weight)
    normalized_ratings = df['Rating'] / 5.0
    rating_scores = 0.3 * normalized_ratings
    
    # Component 3: Popularity (20% weight)
    max_reviews = df['Number of Reviews'].max()
    normalized_reviews = df['Number of Reviews'] / max_reviews
    popularity_scores = 0.2 * normalized_reviews
    
    # Component 4: Cluster Bonus (10% weight)
    cluster_bonus = np.where(df['Cluster'] == book_cluster, 0.1, 0.0)
    
    # Combine all components
    scores = sim_scores + rating_scores + popularity_scores + cluster_bonus
    scores[book_idx] = 0  # Don't recommend the book itself
    
    # Get top N
    top_indices = np.argsort(scores)[::-1][:n_recommendations]
    
    recommendations = df.iloc[top_indices][['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price']].copy()
    recommendations['Similarity Score'] = similarities[top_indices]
    recommendations['Algorithm'] = 'Hybrid'
    
    return recommendations, None

# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1>🎧 Audible Insights</h1>
    <h3>Intelligent Book Recommendations using AI & Machine Learning</h3>
</div>
""", unsafe_allow_html=True)

st.divider()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📚 Find Book Recommendations")
    
    # Search box
    search_term = st.text_input(
        "🔍 Enter book name:",
        placeholder="e.g., Atomic Habits, Dune, Sapiens...",
        help="Type the name of a book you like"
    )
    
    if search_term:
        # Algorithm selection
        st.subheader("🤖 Choose Recommendation Algorithm")
        
        col_algo1, col_algo2, col_algo3 = st.columns(3)
        
        with col_algo1:
            if st.button("📖 Content-Based", use_container_width=True):
                st.session_state.algorithm = "content"
        
        with col_algo2:
            if st.button("🏷️ Clustering-Based", use_container_width=True):
                st.session_state.algorithm = "clustering"
        
        with col_algo3:
            if st.button("⭐ Hybrid (BEST!)", use_container_width=True):
                st.session_state.algorithm = "hybrid"
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Get recommendations
        if 'algorithm' in st.session_state:
            algorithm = st.session_state.algorithm
            
            st.subheader(f"📝 Results - {algorithm.upper()} Algorithm")
            
            if algorithm == "content":
                recommendations, error = content_based_recommendation(search_term, n_recs)
            elif algorithm == "clustering":
                recommendations, error = clustering_based_recommendation(search_term, n_recs)
            else:  # hybrid
                recommendations, error = hybrid_recommendation(search_term, n_recs)
            
            if error:
                st.error(f"❌ {error}")
                st.info("💡 Try searching for: 'Atomic Habits', 'Dune', 'Sapiens', etc.")
            else:
                # Display recommendations
                for idx, (i, row) in enumerate(recommendations.iterrows(), 1):
                    with st.container():
                        col_num, col_book = st.columns([0.5, 3])
                        
                        with col_num:
                            st.markdown(f"### #{idx}")
                        
                        with col_book:
                            st.markdown(f"**{row['Book Name']}**")
                            st.write(f"👤 {row['Author']}")
                            
                            col_rating, col_reviews, col_price = st.columns(3)
                            with col_rating:
                                st.write(f"⭐ {row['Rating']:.1f}/5.0")
                            with col_reviews:
                                st.write(f"👥 {int(row['Number of Reviews']):,} reviews")
                            with col_price:
                                st.write(f"💰 ₹{row['Price']:.0f}")
                        
                        st.divider()

with col2:
    # Sidebar info
    st.subheader("📊 Dataset Info")
    
    st.metric("Total Books", len(df))
    st.metric("Features", "500 TF-IDF")
    st.metric("Clusters", 8)
    st.metric("Processing", "<1 second")
    
    st.subheader("🎯 How It Works")
    st.info("""
    **Content-Based:**
    Finds books with similar descriptions using TF-IDF and cosine similarity.
    
    **Clustering-Based:**
    Groups books by theme (Self-Help, Fiction, Science, etc.), recommends top-rated in same cluster.
    
    **Hybrid (Recommended):**
    Combines all signals:
    - 40% Description similarity
    - 30% Rating quality
    - 20% Popularity (reviews)
    - 10% Same theme bonus
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px; padding: 20px;'>
    <p>🎧 Audible Insights | Powered by Machine Learning</p>
    <p>Using: TF-IDF • Cosine Similarity • K-Means Clustering • Elbow Method</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DEBUG INFO (Hidden by default)
# ============================================================================

with st.expander("🔧 Debug Info"):
    st.write(f"**Loaded files:**")
    st.write(f"- Books: {len(df)}")
    st.write(f"- Similarity matrix: {similarity_matrix.shape}")
    st.write(f"- TF-IDF features: 500")
    st.write(f"- Clusters: 8")
    
    st.write(f"\n**Available books (sample):**")
    st.write(df['Book Name'].head(10).tolist())
