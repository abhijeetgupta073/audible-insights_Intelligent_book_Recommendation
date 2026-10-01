import os
import sys
import time

print("\n" + "=" * 80)
print("AUDIBLE INSIGHTS - COMPLETE PIPELINE")
print("=" * 80)

scripts = [
    ('1_data_preprocessing.py', 'DATA PREPROCESSING'),
    ('2_tfidf_vectorization.py', 'TF-IDF VECTORIZATION'),
    ('3_similarity_matrix.py', 'SIMILARITY MATRIX'),
    ('4_kmeans_clustering.py', 'K-MEANS CLUSTERING'),
]

for script_name, step_name in scripts:
    print(f"\n" + "=" * 80)
    print(f"STEP: {step_name}")
    print("=" * 80)
    print(f"Running: {script_name}\n")
    
    try:
        with open(script_name, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        exec(code)
        print(f"\n✓ {step_name} complete!")
    except FileNotFoundError:
        print(f"ERROR: {script_name} not found!")
        print(f"Make sure {script_name} is in the same folder!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

print("\n" + "=" * 80)
print("ALL STEPS COMPLETE!")
print("=" * 80)
print("""
NEXT: Launch the web app!
Command: streamlit run 5_app.py

This will open your browser with the recommendations app!
""")