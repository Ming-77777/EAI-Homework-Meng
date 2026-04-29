# Aluminum Alloy Preliminary Screening

End-to-end code for the proposal: supervised learning (logistic regression) to classify aluminum alloy records as Pass/Fail (YS >= 300 MPa).

Files:
- `train_eval.py`: download data, preprocess, train logistic regression, evaluate, save model.
- `requirements.txt`: Python dependencies.

Prerequisites
- Python 3.8+ (project tested with Python 3.12 in the provided virtual environment).
- pip available.

Setup (create and activate a virtual environment)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "Graduate Student Responsibilities/aluminum_alloy_project/requirements.txt"
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "Graduate Student Responsibilities/aluminum_alloy_project/requirements.txt"
```

Note: This repository already includes a `.venv` used during development; if you prefer using that environment, activate it instead.

Run the training & evaluation

```bash
python "Graduate Student Responsibilities/aluminum_alloy_project/train_eval.py"
```

What the script does
- Downloads the CSV to `Graduate Student Responsibilities/aluminum_alloy_project/data/property.csv` if not present.
- Preprocesses the data, creates the binary label (`YS >= 300 MPa` → Pass), one-hot-encodes categorical features, and trains a logistic regression baseline.
- Prints evaluation metrics (precision, recall, F1) for the Pass class and a classification report.
- Saves the trained model to `Graduate Student Responsibilities/aluminum_alloy_project/model.joblib`.

Reproducibility
- Set a fixed random seed in `train_eval.py` if you require deterministic splits across runs. The script currently uses `random_state=42` for splitting.

Troubleshooting
- If download fails, open the URL in a browser to confirm access: https://archive.materialscloud.org/records/jxxnh-d0p49/files/property.csv?download=1
- If installation fails on Windows due to execution policy, run PowerShell as administrator and allow script execution or use the CMD activation: `.\\.venv\\Scripts\\activate.bat`.

Contact
- For help reproducing results or modifying the pipeline, ask me and I can add visualization, hyperparameter search, or extended features.

Dependencies (requirements)

The project requires the following Python packages (also listed in `requirements.txt`):

- pandas
- scikit-learn
- requests
- joblib

You can install them via `requirements.txt` or directly with pip:

```bash
pip install -r "Graduate Student Responsibilities/aluminum_alloy_project/requirements.txt"
# or install directly
pip install pandas scikit-learn requests joblib
```

Note: The `requirements.txt` file is kept for convenience; its contents are duplicated here for clarity.

Inspecting the saved model

The training script saves a binary file `model.joblib` containing the trained model and metadata. To inspect
the saved file in a human-readable way, run the included helper script:

```bash
python "Graduate Student Responsibilities/aluminum_alloy_project/inspect_model.py"
```

What `inspect_model.py` shows:
- The top-level keys in the saved object (e.g., `model`, `features`, `threshold`, `best_C`).
- The list of feature names and their count.
- The saved probability threshold and chosen `C` value.
- If the model is linear (has `coef_`), the script prints the top 20 features ranked by absolute coefficient value.

Quick programmatic example (Python) to load and use the model:

```python
import joblib
obj = joblib.load('Graduate Student Responsibilities/aluminum_alloy_project/model.joblib')
model = obj['model'] if isinstance(obj, dict) else obj
features = obj.get('features', None)
threshold = obj.get('threshold', 0.5)

# Prepare a DataFrame `X_sample` with columns matching `features` (same order)
# probs = model.predict_proba(X_sample)[:,1]
# preds = (probs >= threshold).astype(int)
```