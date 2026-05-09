"""Train/evaluate aluminum alloy Pass/Fail classifier and save report artifacts.

Key behaviors:
- YS is used only to construct label_pass (YS >= 300), never as input feature.
- Baseline input is fixed to categorical features: series + temper.
- UTS and elongation are optional extension features for comparison experiments.
- Saves metrics tables, confusion matrix figure, top-coefficients figure, and summary text.
"""
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
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

    ys_col = find_column(df, ['ys', 'yield', 'y_s'])
    uts_col = find_column(df, ['uts', 'ultimate'])
    elong_col = find_column(df, ['elong', 'elongation'])
    if ys_col is None:
        raise RuntimeError('Could not find a YS column in CSV. YS is required to build Pass/Fail label.')
    print('Detected columns: YS=', ys_col, 'UTS=', uts_col, 'ELONG=', elong_col)

    series_col = find_column(df, ['series'])
    temper_col = find_column(df, ['temper', 'temp', 't=', 'aging', 'treatment'])
    if series_col is None or temper_col is None:
        raise RuntimeError('Could not detect both baseline columns (series, temper).')
    print('Detected categorical columns: series=', series_col, 'temper=', temper_col)

    # keep only rows needed for baseline features and label construction
    df = df.dropna(subset=[ys_col, series_col, temper_col]).copy()

    # YS is only used for labeling, never as input.
    df['label_pass'] = (df[ys_col] >= 300).astype(int)

    class_counts = df['label_pass'].value_counts().sort_index()
    print('Cleaned sample count:', len(df))
    print('Pass/Fail distribution (0=Fail,1=Pass):', class_counts.to_dict())

    feature_columns = {
        'ys': ys_col,
        'uts': uts_col,
        'elongation': elong_col,
        'series': series_col,
        'temper': temper_col,
    }
    return df, feature_columns


def build_feature_matrix(df, feature_columns, include_uts=False, include_elongation=False):
    series_col, temper_col = feature_columns['series'], feature_columns['temper']
    uts_col, elong_col = feature_columns['uts'], feature_columns['elongation']

    parts = [pd.get_dummies(df[[series_col, temper_col]].fillna('NA'), dummy_na=False)]
    selected = ['series', 'temper']

    if include_uts and uts_col:
        x_uts = df[[uts_col]].copy()
        x_uts[uts_col] = x_uts[uts_col].fillna(x_uts[uts_col].median())
        parts.append(x_uts)
        selected.append('UTS')

    if include_elongation and elong_col:
        x_elong = df[[elong_col]].copy()
        x_elong[elong_col] = x_elong[elong_col].fillna(x_elong[elong_col].median())
        parts.append(x_elong)
        selected.append('elongation')

    return pd.concat(parts, axis=1), df['label_pass'], selected


def tune_threshold(y_true, y_scores, recall_target=0.75):
    results = []
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_scores >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append((t, prec, rec, f1))
    
    # Prefer threshold meeting recall target with highest precision
    meeting_target = [r for r in results if r[2] >= recall_target]
    if meeting_target:
        t, prec, _, f1 = max(meeting_target, key=lambda x: x[1])
    else:
        t, prec, _, f1 = max(results, key=lambda x: x[3])
    
    return t, prec, f1


def split_data(X, y):
    # split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.5, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_eval_single_experiment(name, X, y):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f'[{name}] Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')

    results = []
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', C=C)
        clf.fit(X_train, y_train)
        score = f1_score(y_val, clf.predict(X_val), zero_division=0)
        print(f'[{name}] C={C} val F1={score:.3f}')
        results.append((C, score, clf))
    
    best_C, _, best_model = max(results, key=lambda x: x[1])

    val_scores = best_model.predict_proba(X_val)[:, 1]
    thresh, prec_val, f1_val = tune_threshold(y_val.values, val_scores, recall_target=0.75)
    print(f'[{name}] Selected threshold on val: {thresh:.3f} (prec={prec_val:.3f}, f1={f1_val:.3f})')

    final_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', C=best_C)
    final_clf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    y_test_proba = final_clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= thresh).astype(int)

    def report(y_true, y_pred, label, exp_name):
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = (y_true == y_pred).mean()
        print(f"[{exp_name}] {label} -> Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, Accuracy: {acc:.3f}")
        print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
        print('\nClassification Report:\n', classification_report(y_true, y_pred, zero_division=0))
        return {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'accuracy': float(acc),
        }

    print(f'\n[{name}] Test performance with tuned threshold {thresh:.3f}:')
    tuned_metrics = report(y_test, y_test_pred, f'Tuned({thresh:.3f})', name)

    val_pred = (best_model.predict_proba(X_val)[:, 1] >= thresh).astype(int)
    val_metrics = {
        'precision': float(precision_score(y_val, val_pred, zero_division=0)),
        'recall': float(recall_score(y_val, val_pred, zero_division=0)),
        'f1': float(f1_score(y_val, val_pred, zero_division=0)),
        'accuracy': float((y_val.values == val_pred).mean()),
    }

    return {
        'name': name,
        'model': final_clf,
        'features': list(X.columns),
        'best_C': float(best_C),
        'threshold': float(thresh),
        'split_sizes': {
            'train': int(len(X_train)),
            'validation': int(len(X_val)),
            'test': int(len(X_test)),
        },
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'metrics_test_tuned': tuned_metrics,
        'metrics_validation_tuned': val_metrics,
        'classification_report': classification_report(y_test, y_test_pred, zero_division=0),
    }


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix (Test, Tuned Threshold)')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_xticks([0, 1], labels=['Fail(0)', 'Pass(1)'])
    ax.set_yticks([0, 1], labels=['Fail(0)', 'Pass(1)'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_top_coefficients(model, features, path, top_n=20):
    if not hasattr(model, 'coef_'):
        return
    feat_coef = sorted(zip(features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, values = zip(*feat_coef[::-1])

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#2f6690' if v >= 0 else '#d1495b' for v in values]
    ax.barh(names, values, color=colors)
    ax.set_title(f'Top {top_n} Coefficients')
    ax.set_xlabel('Value')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metrics_table(experiments, path):
    rows = []
    for exp in experiments:
        row = {
            'experiment': exp['name'],
            'best_C': exp['best_C'],
            'threshold': exp['threshold'],
            'test_precision': exp['metrics_test_tuned']['precision'],
            'test_recall': exp['metrics_test_tuned']['recall'],
            'test_f1': exp['metrics_test_tuned']['f1'],
            'test_accuracy': exp['metrics_test_tuned']['accuracy'],
            'val_precision': exp['metrics_validation_tuned']['precision'],
            'val_recall': exp['metrics_validation_tuned']['recall'],
            'val_f1': exp['metrics_validation_tuned']['f1'],
            'val_accuracy': exp['metrics_validation_tuned']['accuracy'],
            'train_size': exp['split_sizes']['train'],
            'validation_size': exp['split_sizes']['validation'],
            'test_size': exp['split_sizes']['test'],
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def save_summary(df, feature_columns, experiments, final_exp, path):
    class_counts = df['label_pass'].value_counts().sort_index().to_dict()
    lines = [
        'Aluminum Alloy Pass/Fail Screening Summary',
        '==========================================',
        '',
        f'Total cleaned samples: {len(df)}',
        f'Class distribution (0=Fail,1=Pass): {class_counts}',
        '',
        'Label rule:',
        f"- label_pass = 1 if {feature_columns['ys']} >= 300 MPa else 0",
        '- Note: YS is not used as an input feature.',
        '',
        'Experiment comparison:',
    ]

    for exp in experiments:
        m = exp['metrics_test_tuned']
        lines.extend([
            f"- {exp['name']}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, "
            f"f1={m['f1']:.3f}, accuracy={m['accuracy']:.3f}, best_C={exp['best_C']}, threshold={exp['threshold']:.3f}",
            f"  split sizes: train={exp['split_sizes']['train']}, val={exp['split_sizes']['validation']}, test={exp['split_sizes']['test']}",
        ])

    fm = final_exp['metrics_test_tuned']
    lines.extend([
        '',
        'Final selected setup:',
        f"- Experiment: {final_exp['name']}",
        f"- Features: {', '.join(final_exp['selected_feature_names'])}",
        f"- best_C: {final_exp['best_C']}",
        f"- threshold: {final_exp['threshold']:.3f}",
        f"- Test precision/recall/f1/accuracy: {fm['precision']:.3f}/{fm['recall']:.3f}/{fm['f1']:.3f}/{fm['accuracy']:.3f}",
        '',
        'Saved artifacts:',
        '- model.joblib',
        '- outputs/metrics_comparison.csv',
        '- outputs/confusion_matrix_final.png',
        '- outputs/top20_coefficients_final.png',
        '- outputs/final_classification_report.txt',
        '- outputs/summary.txt',
    ])

    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    download_csv()
    df, feature_columns = load_and_prepare()

    experiments = []
    configs = [
        ('series+temper', False),
        ('series+temper+UTS', True),
    ]
    
    for exp_name, include_uts in configs:
        X, y, selected = build_feature_matrix(df, feature_columns, include_uts=include_uts)
        if len(X) < 20:
            print(f'Warning: small dataset: {len(X)}')
        exp = train_and_eval_single_experiment(exp_name, X, y)
        exp['selected_feature_names'] = selected
        experiments.append(exp)

    final_exp = max(experiments, key=lambda e: e['metrics_validation_tuned']['f1'])

    model_path = ROOT / 'model.joblib'
    joblib.dump(
        {
            'model': final_exp['model'],
            'features': final_exp['features'],
            'threshold': final_exp['threshold'],
            'best_C': final_exp['best_C'],
            'final_experiment': final_exp['name'],
            'selected_feature_names': final_exp['selected_feature_names'],
        },
        model_path,
    )
    print('Saved model to', model_path)
    save_metrics_table(experiments, OUTPUT_DIR / 'metrics_comparison.csv')
    save_confusion_matrix(final_exp['confusion_matrix'], OUTPUT_DIR / 'confusion_matrix_final.png')
    save_top_coefficients(final_exp['model'], final_exp['features'], OUTPUT_DIR / 'top20_coefficients_final.png')
    (OUTPUT_DIR / 'final_classification_report.txt').write_text(final_exp['classification_report'], encoding='utf-8')
    save_summary(df, feature_columns, experiments, final_exp, OUTPUT_DIR / 'summary.txt')
    print('Results saved to', OUTPUT_DIR)


if __name__ == '__main__':
    main()
