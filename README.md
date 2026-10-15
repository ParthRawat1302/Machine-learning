# Machine Learning Projects

This workspace contains two end-to-end machine learning projects. Each project
includes notebooks for exploration, preprocessing, and modeling, plus a Python
pipeline and a report with results.

## Projects

| Folder | Task | Target and metric |
|---|---|---|
| [`Vehicle-Classification`](Vehicle-Classification/) | Classify a vehicle from its specifications | `vehicle_class` (Sedan, SUV, Sports Car); accuracy, including evaluation under simulated measurement noise |
| [`Domestic-Flight-Regression`](Domestic-Flight-Regression/) | Predict domestic flight ticket prices | `price`; RMSE |

Both folders contain `01_EDA.ipynb`, `02_Preprocessing.ipynb`, and
`03_Models.ipynb`. Open and run these notebooks in order to follow the analysis
and modeling workflow. The corresponding Python pipeline consolidates the
project code for command-line runs.

## Setup

Use Python 3 with Jupyter, NumPy, pandas, and scikit-learn installed. The
vehicle pipeline also uses SciPy and joblib; some optional model experiments
may require additional packages or GPU support.

```bash
python -m pip install numpy pandas scikit-learn scipy joblib jupyter
```

## Data

The datasets are not included in these project folders. Create a `data/`
directory inside each folder and place the corresponding CSV files there.

**Vehicle classification** expects:

- `Vehicle-Classification/data/vehicle_train_10000.csv`
- `Vehicle-Classification/data/vehicle_test.csv`
- `Vehicle-Classification/data/vehicle_sample_submission.csv` (for submission formatting)

**Domestic flight regression** expects:

- `Domestic-Flight-Regression/data/train.csv`
- `Domestic-Flight-Regression/data/test.csv`

## Run Pipelines

From the workspace root:

```bash
python Vehicle-Classification/vehicle_pipeline.py --quick
python Domestic-Flight-Regression/flight_pipeline.py
```

The vehicle pipeline supports `--data PATH` and `--out PATH` to override its
default data and results folders. Its full run can take substantially longer
than `--quick`; `--no-gpu` skips GPU-dependent model experiments.

The flight pipeline writes its result tables and predictions under
`Domestic-Flight-Regression/results/`.

## Reports

- [Vehicle classification report](Vehicle-Classification/Report%20-%20ML%20Vehicle%20Classification.pdf)
- [Vehicle classification metrics](Vehicle-Classification/Model_Metrics.pdf)
- [Domestic flight regression report](Domestic-Flight-Regression/Report%20-%20ML%20Regression.pdf)
- [Domestic flight regression metrics](Domestic-Flight-Regression/Model_Metrics.pdf)