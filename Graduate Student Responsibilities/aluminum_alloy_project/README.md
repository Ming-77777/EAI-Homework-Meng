# Aluminum Alloy Preliminary Screening

Logistic regression classifier for aluminum alloy Pass/Fail prediction (YS >= 300 MPa).

## Data Source

The dataset comes from Materials Cloud:
https://archive.materialscloud.org/records/jxxnh-d0p49/files/property.csv?download=1

The raw CSV is saved locally at `data/property.csv` after running the training script.

## Setup

Prerequisites: Python 3.8+, pip

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "Graduate Student Responsibilities/aluminum_alloy_project/requirements.txt"
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Graduate Student Responsibilities/aluminum_alloy_project/requirements.txt"
```

## Usage

```bash
python "Graduate Student Responsibilities/aluminum_alloy_project/train_eval.py"
```

This downloads data, trains a model, and saves:
- `model.joblib`: trained classifier with metadata
- `outputs/metrics_comparison.csv`: experiment metrics
- `outputs/confusion_matrix_final.png`: confusion matrix
- `outputs/top20_coefficients_final.png`: coefficient plot
- `outputs/final_classification_report.txt`: final test classification report
- `outputs/summary.txt`: full results summary

## Model Details

- **Label**: YS >= 300 MPa (Pass/Fail)
- **Features**: series + temper (baseline), optionally +UTS
- **Important**: YS is not used as an input feature; it is only used to construct the Pass/Fail label. This avoids label leakage.
- **Method**: Logistic regression with C hyperparameter tuning
- **Validation**: 70/15/15 train/val/test split with stratification
- **Threshold**: tuned on validation set for recall >= 0.75

Final model uses series+temper+UTS (F1: 0.862 on test)

## Inspect Model

```bash
python "Graduate Student Responsibilities/aluminum_alloy_project/inspect_model.py"
```

Or load programmatically:
```python
import joblib
obj = joblib.load('model.joblib')
model = obj['model']
features = obj['features']
threshold = obj['threshold']
```

## Dependencies

- pandas, scikit-learn, requests, joblib, matplotlib