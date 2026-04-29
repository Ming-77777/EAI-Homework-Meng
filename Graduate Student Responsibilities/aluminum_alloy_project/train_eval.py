"""
End-to-end training and evaluation for the aluminum alloy screening proposal.

This script now:
- Detects and uses categorical fields `series` and `temper` (or common synonyms),
  plus numeric features like `YS`, `UTS`, and `elongation` when available.
- Performs a lightweight hyperparameter search on `C` using the validation set.
- Tunes a probability threshold on the validation set (attempts to meet recall target when possible,
  otherwise maximizes F1) and then evaluates on the test set.
"""
import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
CSV_URL = 'https://archive.materialscloud.org/records/jxxnh-d0p49/files/property.csv?download=1'
CSV_PATH = DATA_DIR / 'property.csv'


def download_csv():
    if CSV_PATH.exists():
        print('CSV already exists at', CSV_PATH)
        return
    print('Downloading CSV...')
    r = requests.get(CSV_URL, timeout=30)
    r.raise_for_status()
    CSV_PATH.write_bytes(r.content)
    print('Saved to', CSV_PATH)


def find_column(df, keywords):
    for c in df.columns:
        lc = c.lower()
        for k in keywords:
            if k in lc:
                return c
    return None


def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    print('Loaded CSV with shape', df.shape)

    # find yield strength or related numeric columns
    ys_col = find_column(df, ['ys', 'yield', 'y_s'])
    uts_col = find_column(df, ['uts', 'ultimate'])
    elong_col = find_column(df, ['elong', 'elongation'])
    if ys_col is None and uts_col is None:
        raise RuntimeError('Could not find a YS/UTS column in CSV. Columns: ' + ','.join(df.columns))
    print('Detected columns: YS=', ys_col, 'UTS=', uts_col, 'ELONG=', elong_col)

    # categorical fields: series and temper (temper synonyms include 'temper', 't', 'temp', 'aging', 'treatment')
    series_col = find_column(df, ['series'])
    temper_col = find_column(df, ['temper', 'temp', 't=', 'aging', 'treatment'])
    print('Detected categorical columns: series=', series_col, 'temper=', temper_col)

    # drop rows without any yield/uts info
    numeric_source = ys_col if ys_col is not None else uts_col
    df = df.dropna(subset=[numeric_source])

    # create binary label using YS when available else UTS
    df['label_pass'] = (df[numeric_source] >= 300).astype(int)

    # numeric features present
    numeric_features = []
    for col in [ys_col, uts_col, elong_col]:
        if col is not None and col in df.columns:
            numeric_features.append(col)

    # categorical features
    categorical_features = []
    if series_col is not None:
        categorical_features.append(series_col)
    if temper_col is not None and temper_col not in categorical_features:
        categorical_features.append(temper_col)

    print('Using numeric features:', numeric_features)
    print('Using categorical features:', categorical_features)

    # Prepare feature matrix
    features = []
    X_num = pd.DataFrame(index=df.index)
    if numeric_features:
        X_num = df[numeric_features].copy()
        # simple imputation: median
        for c in X_num.columns:
            if X_num[c].isnull().any():
                X_num[c] = X_num[c].fillna(X_num[c].median())
        features.append(X_num)

    X_cat = pd.DataFrame(index=df.index)
    if categorical_features:
        X_cat = pd.get_dummies(df[categorical_features].fillna('NA'), dummy_na=False)
        features.append(X_cat)

    if features:
        X = pd.concat(features, axis=1)
    else:
        # fallback: use all object dtype columns one-hot encoded
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        X = pd.get_dummies(df[obj_cols].fillna('NA')) if obj_cols else pd.DataFrame(index=df.index)

    y = df['label_pass']
    return X, y


def tune_threshold(y_true, y_scores, recall_target=0.75):
    best_thresh = 0.5
    best_prec = 0.0
    best_f1 = 0.0
    threshs = np.linspace(0.01, 0.99, 99)
    for t in threshs:
        y_pred = (y_scores >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # prefer thresholds meeting recall target with highest precision
        if rec >= recall_target and prec > best_prec:
            best_prec = prec
            best_thresh = t
            best_f1 = f1
    if best_prec > 0:
        return best_thresh, best_prec, best_f1
    # otherwise maximize f1
    best_f1 = -1
    for t in threshs:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    # compute precision at best_f1
    best_prec = precision_score(y_true, (y_scores >= best_thresh).astype(int), zero_division=0)
    return best_thresh, best_prec, best_f1


def train_and_eval(X, y):
    # split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp)

    print('Splits: train', X_train.shape, 'val', X_val.shape, 'test', X_test.shape)

    # lightweight hyperparameter search (C) using validation
    Cs = [0.01, 0.1, 1.0, 10.0]
    best_C = Cs[0]
    best_score = -1
    best_model = None
    for C in Cs:
        try:
            clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', C=C)
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            score = f1_score(y_val, y_val_pred, zero_division=0)
            print(f'C={C} val F1={score:.3f}')
            if score > best_score:
                best_score = score
                best_C = C
                best_model = clf
        except Exception as e:
            print('Error training with C=', C, e)

    if best_model is None:
        print('Falling back to default model training')
        best_C = 1.0
        best_model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', C=best_C)
        best_model.fit(X_train, y_train)

    # tune threshold on validation using predicted probabilities
    try:
        val_scores = best_model.predict_proba(X_val)[:, 1]
        thresh, prec_val, f1_val = tune_threshold(y_val.values, val_scores, recall_target=0.75)
        print(f'Selected threshold on val: {thresh:.3f} (prec={prec_val:.3f}, f1={f1_val:.3f})')
    except Exception as e:
        print('Could not tune threshold on validation:', e)
        thresh = 0.5

    # retrain final model on train+val
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', C=best_C)
    final_clf.fit(X_trainval, y_trainval)

    # evaluate on test
    y_test_proba = final_clf.predict_proba(X_test)[:, 1]
    y_test_pred_default = (y_test_proba >= 0.5).astype(int)
    y_test_pred = (y_test_proba >= thresh).astype(int)

    def report(y_true, y_pred, label):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{label} -> Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
        print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
        print('\nClassification Report:\n', classification_report(y_true, y_pred, zero_division=0))

    print('\nTest performance with default threshold 0.5:')
    report(y_test, y_test_pred_default, 'Default(0.5)')

    print('\nTest performance with tuned threshold {0:.3f}:'.format(thresh))
    report(y_test, y_test_pred, f'Tuned({thresh:.3f})')

    # save model and metadata
    model_path = ROOT / 'model.joblib'
    joblib.dump({'model': final_clf, 'features': list(X.columns), 'threshold': float(thresh), 'best_C': float(best_C)}, model_path)
    print('Saved model to', model_path)


def main():
    download_csv()
    X, y = load_and_prepare()
    if len(X) < 20:
        print('Warning: very small dataset after cleaning:', len(X))
    train_and_eval(X, y)


if __name__ == '__main__':
    main()
