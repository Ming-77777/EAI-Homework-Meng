"""
Inspect saved model.joblib and print human-readable summary.

Usage:
    python inspect_model.py

This script prints keys in the saved object, the feature list, saved threshold and best_C,
and the top 20 coefficients (by absolute value) mapped to feature names when available.
"""
from pathlib import Path
import joblib
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model.joblib'


def main():
    if not MODEL_PATH.exists():
        print('Model file not found:', MODEL_PATH)
        sys.exit(1)

    obj = joblib.load(MODEL_PATH)
    print('Loaded object type:', type(obj))
    if isinstance(obj, dict):
        print('Keys:', list(obj.keys()))
        model = obj.get('model')
        features = obj.get('features')
        threshold = obj.get('threshold')
        best_C = obj.get('best_C')
    else:
        model = obj
        features = None
        threshold = None
        best_C = None

    print('\nModel class:', type(model))
    print('Saved threshold:', threshold)
    print('Saved best_C:', best_C)
    if features is not None:
        print('Number of features:', len(features))

    # If linear model, print coefficients
    try:
        if hasattr(model, 'coef_') and features is not None:
            coefs = model.coef_[0]
            # Map features to coefficients
            feat_coef = list(zip(features, coefs))
            # sort by absolute value
            feat_coef_sorted = sorted(feat_coef, key=lambda x: abs(x[1]), reverse=True)
            print('\nTop 20 features by absolute coefficient:')
            for f, c in feat_coef_sorted[:20]:
                print(f'{f}: {c:.6f}')
        else:
            # fallback: try to print attributes
            print('\nModel attributes:')
            for attr in ['classes_', 'n_features_in_']:
                if hasattr(model, attr):
                    print(f'{attr}:', getattr(model, attr))
    except Exception as e:
        print('Could not extract coefficients:', e)

    # Example: show how to use the model for prediction (not executed)
    print('\nExample usage to load and predict (paste into Python):')
    print('import joblib')
    print("m = joblib.load('Graduate Student Responsibilities/aluminum_alloy_project/model.joblib')")
    print("model = m['model'] if isinstance(m, dict) else m")
    print("features = m.get('features', None)")
    print("# prepare a DataFrame X_sample with same columns as 'features' and run:")
    print("# probs = model.predict_proba(X_sample)[:,1]; preds = (probs >= m.get('threshold',0.5)).astype(int)")


if __name__ == '__main__':
    main()
