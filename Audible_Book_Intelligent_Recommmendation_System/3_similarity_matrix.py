import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time

print("=" * 80)
print("STEP 3: SIMILARITY MATRIX")
print("=" * 80)

print("\n[STEP 3.1] LOADING DATA")
print("-" * 80)

with open('cleaned_data.pkl', 'rb') as f:
    df = pickle.load(f)
print(f"Loaded data: {len(df)} books")

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
print(f"Loaded TF-IDF matrix: {tfidf_matrix.shape}")

print("\n[STEP 3.2] CALCULATING SIMILARITY")
print("-" * 80)

num_books = tfidf_matrix.shape[0]
print(f"Calculating similarity for {num_books} x {num_books} = {num_books**2:,} comparisons...")

start_time = time.time()
similarity_matrix = cosine_similarity(tfidf_matrix)
elapsed = time.time() - start_time

print(f"Complete in {elapsed:.2f} seconds!")

print("\n[STEP 3.3] ANALYSIS")
print("-" * 80)

print(f"Matrix shape: {similarity_matrix.shape}")
print(f"Min: {similarity_matrix.min():.4f}")
print(f"Max: {similarity_matrix.max():.4f}")
print(f"Mean: {similarity_matrix.mean():.4f}")

diagonal = np.diag(similarity_matrix)
print(f"Diagonal (self-similarity): Min={diagonal.min():.4f}, Max={diagonal.max():.4f}")

print("\n[STEP 3.4] SAVING")
print("-" * 80)

with open('similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity_matrix, f)

file_size = __import__('os').path.getsize('similarity_matrix.pkl') / (1024**2)
print(f"Saved: similarity_matrix.pkl ({file_size:.2f} MB)")

print("\n" + "=" * 80)
print("SIMILARITY MATRIX COMPLETE!")
print("=" * 80)