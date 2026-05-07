# Aluminum Alloy Preliminary Screening

End-to-end code for the proposal: supervised learning (logistic regression) to classify aluminum alloy records as Pass/Fail (YS >= 300 MPa).

Important modeling setup in this version:
- `YS` is used only for label construction (`label_pass = 1 if YS >= 300`), not as an input feature.
- Baseline input is fixed to `series + temper`.
- `UTS` and `elongation` are optional extension features.
- The script includes a built-in comparison experiment: `series + temper` vs `series + temper + UTS`.

Files:
- `train_eval.py`: download data, preprocess, run experiments, train logistic regression, evaluate, save model and artifacts.
- `inspect_model.py`: inspect `model.joblib` in a human-readable format.
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
- Cleans data and prints cleaned sample count and Pass/Fail class distribution.
- Builds two experiment feature sets:
	- Baseline: `series + temper`
	- Comparison: `series + temper + UTS`
- For each experiment:
	- Performs train/validation/test split and prints split sizes.
	- Runs a lightweight hyperparameter search for `C`.
	- Tunes probability threshold on validation.
	- Evaluates on test and computes precision, recall, F1, and accuracy.
- Saves the final selected model (`model.joblib`) and report artifacts under `outputs/`.

Reproducibility
- The script uses `random_state=42` for splitting.
- The final experiment is selected by validation F1 (to avoid test-set-based selection).

Discussion
- Adding `UTS` improves test F1 from 0.792 to 0.862, which suggests that additional mechanical-property information helps classification, while the baseline remains `series + temper` to preserve the early-screening setting.

Output files

After running `train_eval.py`, these files are generated:

- `model.joblib`
	- Saved model package including:
		- `model`
		- `features` (encoded feature columns)
		- `selected_feature_names` (human-readable final feature set)
		- `best_C`
		- `threshold`
		- `final_experiment`

- `outputs/metrics_comparison.csv`
	- Table of precision, recall, F1, accuracy (validation/test), split sizes, best `C`, and threshold for each experiment.

- `outputs/confusion_matrix_final.png`
	- Formal confusion matrix figure for the selected final experiment.

- `outputs/top20_coefficients_final.png`
	- Top-20 logistic-regression coefficient plot (by absolute magnitude), intended for model-level interpretation only.
	- Coefficients indicate association direction/strength in this fitted model, not causal effects.

- `outputs/final_classification_report.txt`
	- Full scikit-learn classification report of final test predictions.

- `outputs/summary.txt`
	- Ready-to-copy summary for reports, including cleaned sample info, class distribution, experiment comparison, final selected feature set, best `C`, threshold, and key metrics.

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
- matplotlib

You can install them via `requirements.txt` or directly with pip:

```bash
pip install -r "Graduate Student Responsibilities/aluminum_alloy_project/requirements.txt"
# or install directly
pip install pandas scikit-learn requests joblib matplotlib
```

Note: The `requirements.txt` file is kept for convenience; its contents are duplicated here for clarity.

Inspecting the saved model

The training script saves a binary file `model.joblib` containing the trained model and metadata. To inspect
the saved file in a human-readable way, run the included helper script:

```bash
python "Graduate Student Responsibilities/aluminum_alloy_project/inspect_model.py"
```

What `inspect_model.py` shows:
- The top-level keys in the saved object (e.g., `model`, `features`, `selected_feature_names`, `threshold`, `best_C`, `final_experiment`).
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