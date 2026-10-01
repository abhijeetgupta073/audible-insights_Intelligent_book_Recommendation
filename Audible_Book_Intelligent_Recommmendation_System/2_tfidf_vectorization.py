# ============================================================================
# STEP 2: TF-IDF VECTORIZATION (FIXED VERSION)
# Convert book descriptions to numerical features (500 words)
# ============================================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AUDIBLE INSIGHTS - STEP 2: TF-IDF VECTORIZATION")
print("=" * 80)

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================

print("\n[STEP 2.1] LOADING CLEANED DATA")
print("-" * 80)

with open('cleaned_data.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"✓ Loaded cleaned data: {len(df)} books")
print(f"✓ Columns: {list(df.columns)}")

# ============================================================================
# STEP 2.2: UNDERSTAND TF-IDF
# ============================================================================

print("\n[STEP 2.2] WHAT IS TF-IDF?")
print("-" * 80)

print("""
TF-IDF = Term Frequency × Inverse Document Frequency

Simple explanation:
- TF (Term Frequency): How often a word appears in ONE book's description
- IDF (Inverse Document Frequency): How unique/rare is that word across ALL books

Example:
  Word "meditation":
    - Appears 5 times in book description (TF = high)
    - Appears in only 200/5403 books (IDF = high, rare word!)
    - Result: TF-IDF = HIGH (important word for recommendations!)
    
  Word "the":
    - Appears 50 times in book description (TF = high)
    - Appears in 5000/5403 books (IDF = low, common word!)
    - Result: TF-IDF = LOW (not useful for recommendations!)

So we get 500 most important words for each book!
""")

# ============================================================================
# STEP 2.3: CHECK DESCRIPTIONS
# ============================================================================

print("\n[STEP 2.3] EXAMINING BOOK DESCRIPTIONS")
print("-" * 80)

print(f"\nTotal books: {len(df)}")
print(f"Books with descriptions: {df['Description'].notna().sum()}")
print(f"Books without descriptions: {df['Description'].isna().sum()}")

print(f"\nSample descriptions:")
for i in range(3):
    print(f"\n{i+1}. {df.iloc[i]['Book Name'].upper()}")
    print(f"   {df.iloc[i]['Description'][:100]}...")

# ============================================================================
# STEP 2.4: CREATE TF-IDF VECTORIZER (FIXED - No max_df_error_rate)
# ============================================================================

print("\n[STEP 2.4] CREATING TF-IDF VECTORIZER")
print("-" * 80)

print("""
TfidfVectorizer parameters:
  max_features=500:
    └─ Use only TOP 500 most important words
    └─ Reduces noise, improves speed
    
  stop_words='english':
    └─ Ignore common words: 'the', 'a', 'is', 'and', etc.
    └─ These don't help with recommendations
    
  min_df=2:
    └─ Ignore words that appear in < 2 books
    └─ These are typos or very rare
    
  max_df=0.9:
    └─ Ignore words in > 90% of books
    └─ Too common, not discriminative
    
  ngram_range=(1, 2):
    └─ Use single words AND 2-word phrases
    └─ Example: 'meditation', 'daily meditation'
    └─ Captures better meaning!
""")

# FIXED: Removed max_df_error_rate which doesn't exist in scikit-learn 1.3.0
vectorizer = TfidfVectorizer(
    max_features=500,           # Top 500 words
    stop_words='english',       # Remove common words
    min_df=2,                   # Ignore rare words
    max_df=0.9,                 # Ignore very common words
    ngram_range=(1, 2),         # Single words + phrases
    lowercase=True,
    analyzer='word',
    sublinear_tf=True
)

print("✓ Vectorizer created!")

# ============================================================================
# STEP 2.5: FIT AND TRANSFORM DESCRIPTIONS
# ============================================================================

print("\n[STEP 2.5] FITTING VECTORIZER AND TRANSFORMING DESCRIPTIONS")
print("-" * 80)

import time
start_time = time.time()

print("Converting descriptions to vectors...")
tfidf_matrix = vectorizer.fit_transform(df['Description'])

vectorization_time = time.time() - start_time

print(f"✓ Vectorization complete in {vectorization_time:.2f} seconds!")

print(f"\nTF-IDF Matrix Details:")
print(f"  Shape: {tfidf_matrix.shape}")
print(f"    - Rows: {tfidf_matrix.shape[0]} (number of books)")
print(f"    - Columns: {tfidf_matrix.shape[1]} (number of features/words)")
print(f"  Data type: {tfidf_matrix.dtype}")
print(f"  Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.1f}%")
print(f"    └─ Most values are 0 (sparse matrix saves memory!)")

# ============================================================================
# STEP 2.6: SHOW TOP FEATURES
# ============================================================================

print("\n[STEP 2.6] TOP 20 MOST IMPORTANT WORDS/FEATURES")
print("-" * 80)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Get top features by average TF-IDF score
feature_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
top_indices = np.argsort(feature_scores)[-20:][::-1]

print("\nTop words used for recommendations:")
for i, idx in enumerate(top_indices, 1):
    word = feature_names[idx]
    score = feature_scores[idx]
    print(f"  {i:2d}. {word:20s} (importance score: {score:.4f})")

# ============================================================================
# STEP 2.7: ANALYZE WORD DISTRIBUTION
# ============================================================================

print("\n[STEP 2.7] WORD DISTRIBUTION ANALYSIS")
print("-" * 80)

# For each book, show number of non-zero features
words_per_book = np.diff(tfidf_matrix.indptr)
print(f"\nWords per book:")
print(f"  Min: {words_per_book.min()}")
print(f"  Max: {words_per_book.max()}")
print(f"  Average: {words_per_book.mean():.1f}")
print(f"  Median: {np.median(words_per_book):.1f}")

# ============================================================================
# STEP 2.8: EXAMPLE - VECTORIZE A NEW BOOK
# ============================================================================

print("\n[STEP 2.8] EXAMPLE: VECTORIZE A NEW BOOK")
print("-" * 80)

example_description = "A book about building daily habits and personal productivity"
print(f"New book description: {example_description}")

example_vector = vectorizer.transform([example_description])
print(f"\nConverted to vector:")
print(f"  Shape: {example_vector.shape}")
print(f"  Non-zero values: {example_vector.nnz}")

# Show top words from this description
example_scores = example_vector.toarray()[0]
top_example_indices = np.argsort(example_scores)[-5:][::-1]
print(f"\nTop 5 words in this description:")
for rank, idx in enumerate(top_example_indices, 1):
    word = feature_names[idx]
    score = example_scores[idx]
    print(f"  {rank}. {word:20s} (TF-IDF score: {score:.4f})")

# ============================================================================
# STEP 2.9: SAVE VECTORIZER AND MATRIX
# ============================================================================

print("\n[STEP 2.9] SAVING VECTORIZER AND MATRIX")
print("-" * 80)

# Save vectorizer
print("Saving TF-IDF vectorizer...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
file_size = __import__('os').path.getsize('tfidf_vectorizer.pkl') / (1024**2)
print(f"✓ Saved: tfidf_vectorizer.pkl ({file_size:.2f} MB)")

# Save matrix
print("Saving TF-IDF matrix...")
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
file_size = __import__('os').path.getsize('tfidf_matrix.pkl') / (1024**2)
print(f"✓ Saved: tfidf_matrix.pkl ({file_size:.2f} MB)")

# ============================================================================
# STEP 2.10: WHAT'S NEXT?
# ============================================================================

print("\n" + "=" * 80)
print("TF-IDF VECTORIZATION COMPLETE!")
print("=" * 80)

print(f"""
WHAT WE DID:
✓ Loaded cleaned data ({len(df)} books)
✓ Created TF-IDF vectorizer with 500 features
✓ Converted all descriptions to {tfidf_matrix.shape[0]}×{tfidf_matrix.shape[1]} matrix
✓ Saved vectorizer and matrix

OUTPUT FILES:
✓ tfidf_vectorizer.pkl (vectorizer object)
✓ tfidf_matrix.pkl (matrix)

NEXT STEP:
Run: python 3_similarity_matrix.py
This will calculate similarity between ALL pairs of books!
Time: ~30 seconds
Output: 5403×5403 similarity matrix for instant recommendations!
""")

print("=" * 80)