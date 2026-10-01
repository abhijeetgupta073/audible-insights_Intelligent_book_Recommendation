# ============================================================================
# STEP 1: DATA PREPROCESSING & CLEANING
# Complete data cleaning pipeline with all steps explained
# ============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AUDIBLE INSIGHTS - STEP 1: DATA PREPROCESSING & CLEANING")
print("=" * 80)

# ============================================================================
# LOAD RAW CSV FILES
# ============================================================================

print("\n[STEP 1.1] LOADING RAW CSV FILES")
print("-" * 80)

# Load both CSV files
df1 = pd.read_csv('Audible_Catlog.csv')
df2 = pd.read_csv('Audible_Catlog_Advanced_Features.csv')

print(f"\n✓ File 1 loaded: {len(df1)} rows × {len(df1.columns)} columns")
print(f"  Columns: {list(df1.columns)}")

print(f"\n✓ File 2 loaded: {len(df2)} rows × {len(df2.columns)} columns")
print(f"  Columns: {list(df2.columns)}")

# ============================================================================
# MERGE DATASETS
# ============================================================================

print("\n[STEP 1.2] MERGING DATASETS")
print("-" * 80)

# Start with df1 as base
df = df1.copy()

# Merge description from df2 if available
if 'Description' in df2.columns:
    description_map = df2.set_index('Book Name')['Description'].to_dict()
    df['Description'] = df['Book Name'].map(description_map)
    print("✓ Added Description column from File 2")
else:
    print("✗ Description column not found in File 2")
    df['Description'] = 'Unknown Book Description'

if 'Genre' in df2.columns:
    genre_map = df2.set_index('Book Name')['Genre'].to_dict()
    df['Genre'] = df['Book Name'].map(genre_map)
    print("✓ Added Genre column from File 2")

print(f"\n✓ Merged dataset: {len(df)} rows × {len(df.columns)} columns")

# ============================================================================
# STEP 1.3: HANDLE PRICE COLUMN (Remove ₹ symbol)
# ============================================================================

print("\n[STEP 1.3] CLEANING PRICE COLUMN (Remove ₹ symbol)")
print("-" * 80)

def safe_convert_price(value):
    """Convert price with ₹ symbol to numeric"""
    if pd.isna(value) or value == '':
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remove ₹ symbol and commas
    value_str = str(value).replace('₹', '').replace(',', '').strip()
    
    try:
        return float(value_str)
    except:
        return np.nan

if 'Price' in df.columns:
    print(f"\nBefore cleaning:")
    print(f"  Sample values: {df['Price'].head(3).tolist()}")
    print(f"  Data type: {df['Price'].dtype}")
    
    df['Price'] = df['Price'].apply(safe_convert_price)
    
    print(f"\nAfter cleaning:")
    print(f"  Sample values: {df['Price'].head(3).tolist()}")
    print(f"  Data type: {df['Price'].dtype}")
    print(f"  ✓ Price column cleaned!")

# ============================================================================
# STEP 1.4: HANDLE MISSING VALUES
# ============================================================================

print("\n[STEP 1.4] HANDLING MISSING VALUES")
print("-" * 80)

print("\nMissing values BEFORE filling:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0].to_string())

# Fill missing ratings with neutral value
if 'Rating' in df.columns:
    missing_rating = df['Rating'].isna().sum()
    df['Rating'] = df['Rating'].fillna(3.5)
    print(f"\n✓ Filled {missing_rating} missing ratings with 3.5 (neutral)")

# Fill missing review counts with 0 (new books)
if 'Number of Reviews' in df.columns:
    missing_reviews = df['Number of Reviews'].isna().sum()
    df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
    print(f"✓ Filled {missing_reviews} missing review counts with 0 (new books)")

# Fill missing prices with median
if 'Price' in df.columns:
    missing_price = df['Price'].isna().sum()
    median_price = df['Price'].median()
    df['Price'] = df['Price'].fillna(median_price)
    print(f"✓ Filled {missing_price} missing prices with median ({median_price:.2f})")

# Fill missing descriptions
if 'Description' in df.columns:
    missing_desc = df['Description'].isna().sum()
    df['Description'] = df['Description'].fillna('Unknown')
    print(f"✓ Filled {missing_desc} missing descriptions with 'Unknown'")

# Fill missing genres
if 'Genre' in df.columns:
    missing_genre = df['Genre'].isna().sum()
    df['Genre'] = df['Genre'].fillna('Unknown')
    print(f"✓ Filled {missing_genre} missing genres with 'Unknown'")

print("\nMissing values AFTER filling:")
missing_after = df.isnull().sum()
if missing_after.sum() == 0:
    print("✓ NO MISSING VALUES! All data filled!")
else:
    print(missing_after[missing_after > 0].to_string())

# ============================================================================
# STEP 1.5: FIX INVALID VALUES
# ============================================================================

print("\n[STEP 1.5] FIXING INVALID VALUES")
print("-" * 80)

# Cap ratings at 5.0
if 'Rating' in df.columns:
    invalid_ratings = (df['Rating'] > 5.0).sum()
    df['Rating'] = df['Rating'].apply(lambda x: min(x, 5.0) if pd.notna(x) else 3.5)
    if invalid_ratings > 0:
        print(f"✓ Capped {invalid_ratings} ratings above 5.0 to 5.0")

# Fix negative prices
if 'Price' in df.columns:
    negative_prices = (df['Price'] < 0).sum()
    df['Price'] = df['Price'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
    if negative_prices > 0:
        print(f"✓ Fixed {negative_prices} negative prices to 0")

# Fix negative review counts
if 'Number of Reviews' in df.columns:
    negative_reviews = (df['Number of Reviews'] < 0).sum()
    df['Number of Reviews'] = df['Number of Reviews'].apply(lambda x: max(x, 0) if pd.notna(x) else 0)
    if negative_reviews > 0:
        print(f"✓ Fixed {negative_reviews} negative review counts to 0")

print("✓ All invalid values fixed!")

# ============================================================================
# STEP 1.6: REMOVE DUPLICATES
# ============================================================================

print("\n[STEP 1.6] REMOVING DUPLICATES")
print("-" * 80)

before_dedup = len(df)

# Remove exact duplicates
df = df.drop_duplicates()

# Remove duplicates based on Book Name and Author (if both exist)
if 'Book Name' in df.columns and 'Author' in df.columns:
    df = df.drop_duplicates(subset=['Book Name', 'Author'], keep='first')

after_dedup = len(df)
removed = before_dedup - after_dedup

print(f"\nBefore deduplication: {before_dedup} rows")
print(f"After deduplication: {after_dedup} rows")
print(f"✓ Removed {removed} duplicate rows")

# ============================================================================
# STEP 1.7: STANDARDIZE TEXT COLUMNS
# ============================================================================

print("\n[STEP 1.7] STANDARDIZING TEXT COLUMNS")
print("-" * 80)

text_columns = ['Book Name', 'Author', 'Description', 'Genre']
for col in text_columns:
    if col in df.columns:
        # Strip whitespace
        df[col] = df[col].str.strip()
        # Convert to lowercase
        df[col] = df[col].str.lower()
        print(f"✓ Standardized {col} (stripped, lowercased)")

# ============================================================================
# STEP 1.8: DATA VALIDATION
# ============================================================================

print("\n[STEP 1.8] DATA VALIDATION")
print("-" * 80)

print(f"\nFinal dataset info:")
print(f"  Total rows: {len(df)}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

print(f"\nData ranges:")
if 'Rating' in df.columns:
    print(f"  Rating: {df['Rating'].min():.1f} - {df['Rating'].max():.1f}")
if 'Number of Reviews' in df.columns:
    print(f"  Reviews: {df['Number of Reviews'].min():.0f} - {df['Number of Reviews'].max():.0f}")
if 'Price' in df.columns:
    print(f"  Price: ₹{df['Price'].min():.2f} - ₹{df['Price'].max():.2f}")

# ============================================================================
# SAVE CLEANED DATA
# ============================================================================

print("\n[STEP 1.9] SAVING CLEANED DATA")
print("-" * 80)

# Save as pickle for fast loading
import pickle
with open('cleaned_data.pkl', 'wb') as f:
    pickle.dump(df, f)

file_size = __import__('os').path.getsize('cleaned_data.pkl') / (1024**2)
print(f"\n✓ Saved cleaned data as: cleaned_data.pkl ({file_size:.2f} MB)")

# Also save as CSV for inspection
df.to_csv('cleaned_data.csv', index=False)
print(f"✓ Saved cleaned data as: cleaned_data.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETE!")
print("=" * 80)

print(f"""
SUMMARY:
├─ Loaded: 2 CSV files
├─ Merged: Datasets combined
├─ Cleaned: ₹ symbols removed, missing values filled, invalid values fixed
├─ Deduped: {removed} duplicates removed
├─ Result: {len(df)} clean books ready for ML
└─ Output: cleaned_data.pkl + cleaned_data.csv

QUALITY METRICS:
├─ Missing values: {df.isnull().sum().sum()} (0 is perfect!)
├─ Duplicate rows: 0 (all removed!)
├─ Invalid ratings: 0 (all capped at 5.0!)
├─ Negative prices: 0 (all fixed!)
└─ Text standardized: Yes!

NEXT STEP:
Run: python 2_tfidf_vectorization.py
This will create TF-IDF vectors for recommendations!
""")

print("=" * 80)
