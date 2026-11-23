# Kaudulla Elephant Movement Lab

An interactive Streamlit dashboard and Jupyter Notebook comparing ensemble-learning approaches on the supplied Sri Lankan Kaudulla elephant GPS telemetry dataset.

## Included

- `app.py` — single-page Streamlit dashboard with in-page navigation (Overview, Model arena, Movement explorer, Live prediction, Methodology).
- `Ensemble_Learning_Kaudulla.ipynb` — project notebook with data preparation, modelling workflow, visual evaluation and written interpretation.
- `train_models.py` — reproducible model-training pipeline.
- `data/kaudulla_elephants_clean.csv` — supplied dataset.
- `models/` — fitted pipelines saved with joblib.
- `results/` — metrics, feature importance, confusion matrix, run summary and sampled map data.

## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). The included fitted models and result files are already generated. Re-run `python train_models.py` if you want to reproduce training from the CSV.

## Notebook

Open `Ensemble_Learning_Kaudulla.ipynb` in Jupyter or VS Code. Keep the notebook in the project root so it can import `train_models.py` and read `data/`.

## Target construction and caution

The original file has 86,747 rows and many missing latitude/longitude values. The supervised target is engineered from consecutive valid GPS fixes for the same collar. Only pairs separated by 0.25–2.5 hours and with plausible Sri Lankan coordinates are retained. Haversine speed is calculated and values above 20 km/h are excluded as likely GPS jumps. Classes are defined using the 33rd and 67th percentiles of observed speed (approximately 0.090 km/h and 0.238 km/h in this dataset): Low Movement, Moderate Movement, and High Movement. Quantile-based bins avoid an almost-empty high-speed class in the supplied GPS sample. These are operational thresholds, not validated biological behaviour labels.

Random Forest hyperparameters are tuned with a compact 3-fold GridSearchCV (max_depth and min_samples_leaf) on a reproducible training subset. The random stratified split can place observations from the same collar in both train and test partitions, so scores may overstate generalisation to unseen elephants. The notebook discusses this limitation and recommends collar-wise validation for a stronger follow-up experiment.

## AI Usage Acknowledgment

Generative AI tools, including ChatGPT, were used as supporting resources during the development of this project, primarily for guidance on dashboard design, code structure, documentation, and Jupyter Notebook organisation.

## Submission scope

This project package intentionally contains the notebook and local dashboard project only. It does not include a GitHub repository link or a presentation video.
