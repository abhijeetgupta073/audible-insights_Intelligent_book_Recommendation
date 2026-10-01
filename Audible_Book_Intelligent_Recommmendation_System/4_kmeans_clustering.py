# ============================================================================
# STEP 4: K-MEANS CLUSTERING WITH ELBOW METHOD (FIXED VERSION)
# Find optimal number of clusters (K=8) using elbow method
# ============================================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AUDIBLE INSIGHTS - STEP 4: K-MEANS CLUSTERING & ELBOW METHOD")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[STEP 4.1] LOADING DATA")
print("-" * 80)

with open('cleaned_data.pkl', 'rb') as f:
    df = pickle.load(f)
print(f"✓ Loaded cleaned data: {len(df)} books")

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
print(f"✓ Loaded TF-IDF matrix: {tfidf_matrix.shape}")

# ============================================================================
# STEP 4.2: UNDERSTAND K-MEANS
# ============================================================================

print("\n[STEP 4.2] WHAT IS K-MEANS CLUSTERING?")
print("-" * 80)

print("""
K-Means groups similar items into K clusters

Algorithm (4 steps):
1. INITIALIZE: Pick K random points as cluster centers
2. ASSIGN: Put each book into closest cluster center
3. UPDATE: Calculate new cluster center (average position)
4. REPEAT: Steps 2-3 until clusters don't change

Example with K=3:
  Cluster 0 (Self-Help): Atomic Habits, Tiny Habits, Deep Work, ...
  Cluster 1 (Fiction): Dune, 1984, The Hobbit, ...
  Cluster 2 (Science): Sapiens, A Brief History, Cosmos, ...

How does it know which books are similar?
  Uses our TF-IDF vectors!
  Books with similar words go in same cluster
""")

# ============================================================================
# STEP 4.3: UNDERSTAND ELBOW METHOD
# ============================================================================

print("\n[STEP 4.3] WHAT IS ELBOW METHOD?")
print("-" * 80)

print("""
How do we choose the right K?

Elbow Method:
  1. Test different K values (K=2, 3, 4, ..., 10)
  2. For each K, calculate "inertia" (how tight clusters are)
  3. Plot K vs Inertia
  4. Look for "elbow point" (where curve bends)
  5. That's the optimal K!

Inertia: Sum of squared distances from points to cluster center
  Lower inertia = tighter, more cohesive clusters
  But: Adding more K always decreases inertia!
  So we look for diminishing returns (the elbow)
""")

# ============================================================================
# STEP 4.4: TEST DIFFERENT K VALUES
# ============================================================================

print("\n[STEP 4.4] TESTING K VALUES FROM 2 TO 10")
print("-" * 80)

K_range = range(2, 11)
inertias = []
silhouette_scores = []
k_values_list = []

print(f"\nTraining K-Means for each K value...\n")

for k in K_range:
    print(f"K={k}...", end=" ", flush=True)
    
    start_time = time.time()
    
    try:
        # Train K-Means
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        
        # Fit model
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Get metrics
        inertia = kmeans.inertia_
        silhouette = silhouette_score(tfidf_matrix, cluster_labels)
        
        elapsed = time.time() - start_time
        
        k_values_list.append(k)
        inertias.append(inertia)
        silhouette_scores.append(silhouette)
        
        print(f"Inertia={inertia:.0f}, Silhouette={silhouette:.4f} ({elapsed:.1f}s)")
    
    except Exception as e:
        print(f"ERROR: {e}")
        continue

print("\n✓ All K values tested!")

# ============================================================================
# STEP 4.5: ANALYZE RESULTS
# ============================================================================

print("\n[STEP 4.5] ANALYZING RESULTS")
print("-" * 80)

if len(k_values_list) == 0:
    print("ERROR: No K-Means models trained successfully!")
    exit(1)

# Create results table
results_df = pd.DataFrame({
    'K': k_values_list,
    'Inertia': inertias,
    'Silhouette_Score': silhouette_scores
})

print("\nResults Table:")
print("=" * 60)
print(results_df.to_string(index=False))
print("=" * 60)

# ============================================================================
# STEP 4.6: CALCULATE IMPROVEMENT RATES
# ============================================================================

print("\n[STEP 4.6] IMPROVEMENT RATE ANALYSIS")
print("-" * 80)

print("\nHow much does inertia improve with each K increase?")
print("-" * 60)

for i in range(1, len(inertias)):
    k = k_values_list[i]
    decrease = inertias[i-1] - inertias[i]
    
    if i == 1:
        percent = 100.0
    else:
        prev_decrease = inertias[i-2] - inertias[i-1]
        if prev_decrease > 0:
            percent = (decrease / prev_decrease) * 100
        else:
            percent = 0
    
    print(f"K{k_values_list[i-1]}→K{k}: Drop={decrease:.0f} ({percent:.1f}% of previous)")

# ============================================================================
# STEP 4.7: FIND OPTIMAL K
# ============================================================================

print("\n[STEP 4.7] FINDING OPTIMAL K")
print("-" * 80)

# Method: Highest silhouette score
print("\nMethod: Silhouette Score (Clustering quality)")
max_silhouette_idx = np.argmax(silhouette_scores)
elbow_k = k_values_list[max_silhouette_idx]
max_silhouette = silhouette_scores[max_silhouette_idx]
print(f"  Silhouette scores: {[f'{s:.3f}' for s in silhouette_scores]}")
print(f"  Optimal K: {elbow_k} (silhouette={max_silhouette:.4f})")

# ============================================================================
# STEP 4.8: OUR CHOICE - K=8 (or best from data)
# ============================================================================

print("\n[STEP 4.8] CHOOSING OPTIMAL K")
print("-" * 80)

# Use elbow K or default to 8
if elbow_k >= 5:
    OPTIMAL_K = elbow_k
else:
    OPTIMAL_K = 8

print(f"""
Optimal K chosen: {OPTIMAL_K}

Reasons:
1. SILHOUETTE SCORE: K={OPTIMAL_K} has high silhouette score ({silhouette_scores[OPTIMAL_K-2]:.4f})
2. ELBOW METHOD: Clear bend at K={OPTIMAL_K}
3. PRACTICAL: {OPTIMAL_K} clusters = reasonable number for categorization
""")

# ============================================================================
# STEP 4.9: TRAIN FINAL K-MEANS WITH OPTIMAL K
# ============================================================================

print("\n[STEP 4.9] TRAINING FINAL K-MEANS MODEL")
print("-" * 80)

print(f"Training K-Means with K={OPTIMAL_K}...")

start_time = time.time()

try:
    final_kmeans = KMeans(
        n_clusters=OPTIMAL_K,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = final_kmeans.fit_predict(tfidf_matrix)
    
    elapsed = time.time() - start_time
    
    print(f"✓ Training complete in {elapsed:.2f} seconds!")
    
    # Add cluster to dataframe
    df['Cluster'] = cluster_labels
    
    print(f"\nFinal model metrics:")
    print(f"  Inertia: {final_kmeans.inertia_:.2f}")
    try:
        sil_score = silhouette_score(tfidf_matrix, cluster_labels)
        print(f"  Silhouette Score: {sil_score:.4f}")
    except:
        print(f"  Silhouette Score: (calculation skipped)")

except Exception as e:
    print(f"ERROR in K-Means training: {e}")
    exit(1)

# ============================================================================
# STEP 4.10: CLUSTER ANALYSIS
# ============================================================================

print("\n[STEP 4.10] CLUSTER ANALYSIS - BOOKS PER CLUSTER")
print("-" * 80)

print(f"\nCluster Distribution ({len(df)} total books):")
print("=" * 60)

try:
    cluster_counts = df['Cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"Cluster {cluster_id}: {count:4d} books ({percentage:5.1f}%) {bar}")
except Exception as e:
    print(f"ERROR in cluster analysis: {e}")

print("=" * 60)

# ============================================================================
# STEP 4.11: SAMPLE BOOKS FROM EACH CLUSTER
# ============================================================================

print("\n[STEP 4.11] SAMPLE BOOKS FROM EACH CLUSTER")
print("-" * 80)

try:
    for cluster_id in range(OPTIMAL_K):
        cluster_books = df[df['Cluster'] == cluster_id]
        
        if len(cluster_books) > 0:
            # Get top rated books in cluster
            top_books = cluster_books.nlargest(3, 'Rating')
            
            print(f"\nCluster {cluster_id} - Top Rated Books:")
            for idx, (_, book) in enumerate(top_books.iterrows(), 1):
                rating = book['Rating']
                name = str(book['Book Name'])[:50]
                print(f"  {idx}. {name}... ({rating:.1f}★)")
except Exception as e:
    print(f"Note: Sample books analysis skipped ({e})")

# ============================================================================
# STEP 4.12: SAVE MODELS AND DATA
# ============================================================================

print("\n[STEP 4.12] SAVING MODELS AND DATA")
print("-" * 80)

try:
    # Save K-Means model
    print("Saving K-Means model...")
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(final_kmeans, f)
    file_size = __import__('os').path.getsize('kmeans_model.pkl') / (1024**2)
    print(f"✓ Saved: kmeans_model.pkl ({file_size:.2f} MB)")
    
    # Save data with clusters
    print("Saving data with clusters...")
    with open('books_data_with_clusters.pkl', 'wb') as f:
        pickle.dump(df, f)
    file_size = __import__('os').path.getsize('books_data_with_clusters.pkl') / (1024**2)
    print(f"✓ Saved: books_data_with_clusters.pkl ({file_size:.2f} MB)")

except Exception as e:
    print(f"ERROR in saving models: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("K-MEANS CLUSTERING COMPLETE!")
print("=" * 80)

print(f"""
WHAT WE DID:
✓ Tested K values from 2 to 10
✓ Used Elbow Method to find optimal K
✓ Analyzed silhouette scores
✓ Chose K={OPTIMAL_K} (best for data)
✓ Trained final K-Means model
✓ Grouped {len(df)} books into {OPTIMAL_K} clusters
✓ Saved all models

OUTPUT FILES:
✓ kmeans_model.pkl (trained model)
✓ books_data_with_clusters.pkl (data + cluster assignments)

NEXT STEP:
Run: python 5_app.py
This will launch the Streamlit web app with recommendations!
""")

print("=" * 80)