"""Inspect saved model.joblib and print summary."""
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'model.joblib'


def main():
    if not MODEL_PATH.exists():
        print('Model not found:', MODEL_PATH)
        return

    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        model = obj.get('model')
        features = obj.get('features')
        threshold = obj.get('threshold')
        best_C = obj.get('best_C')
        print('Keys:', list(obj.keys()))
    else:
        model = obj
        features = threshold = best_C = None

    print(f'Model: {type(model).__name__}')
    print(f'Threshold: {threshold}, best_C: {best_C}')
    if features:
        print(f'Features ({len(features)}): {features[:5]}...')

    if hasattr(model, 'coef_') and features:
        feat_coef = sorted(zip(features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
        print('\nTop 20 coefficients:')
        for f, c in feat_coef[:20]:
            print(f'  {f}: {c:.6f}')


if __name__ == '__main__':
    main()
