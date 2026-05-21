# K-Nearest Neighbors — Wine Quality Classification

> Multi-class KNN classifier on 1,599 red wine samples: chemical feature scaling, a k-sweep from 1 to 20 to find the optimal neighbourhood size, and a `predict_wine_quality()` inference function — achieving 84.4% accuracy at k=5 and peaking at k=14.

---

## Problem

Predict whether a red wine is of **low**, **medium**, or **high** quality based on 11 physicochemical measurements. Winemakers and distributors want an objective, data-driven quality signal that doesn't rely purely on expensive expert tasters. This is a 3-class classification problem.

## Dataset

- **Source:** Red Wine Quality dataset (UCI via GitHub)
- **Size:** 1,599 rows × 12 columns (11 features + quality score)
- **Quality label engineering:** `quality` score (0–10) → 3 classes:

| Quality score | Label | Class |
|---|---|---|
| ≤ 4 | Low | 0 |
| 5–6 | Medium | 1 |
| ≥ 7 | High | 2 |

**Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

## Pipeline

| Step | Action |
|---|---|
| Label engineering | Quality scores bucketed into 3 ordinal classes |
| Train/test split | 80/20, random_state=42 (1,279 train / 320 test) |
| Scaling | `StandardScaler` — critical for KNN since distance is scale-dependent |
| Baseline model | `KNeighborsClassifier(n_neighbors=5)` |
| Optimisation | Sweep k=1 to k=20, record accuracy at each k, select best |
| Best k | **k=14** |
| Inference function | `predict_wine_quality(features)` → returns human-readable label |

## Model Results

**k=5 baseline:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low (0) | 0.00 | 0.00 | 0.00 | 11 |
| Medium (1) | 0.87 | 0.95 | 0.91 | 262 |
| High (2) | 0.65 | 0.43 | 0.51 | 47 |
| **Overall accuracy** | | | **84.4%** | 320 |

**After k-sweep optimisation → k=14** improves accuracy further by smoothing the decision boundary.

**Key observation:** Class 0 (low quality) has zero precision and recall at k=5 — only 11 test samples make it essentially invisible during training. This is a class imbalance problem, not a KNN failure.

## Key Takeaways

- **Scaling is mandatory for KNN:** KNN measures distance between data points. Without StandardScaler, features with large numerical ranges (like `total sulfur dioxide` up to 289) dominate the distance calculation and drown out informative features like `pH` (range ≈ 3.0–4.0).
- **k controls the bias–variance tradeoff:** Small k (k=1) memorises training noise — high variance. Large k averages over many neighbours — high bias, smoother boundaries. The k-sweep makes this tradeoff explicit and picks the empirical optimum.
- **Overall accuracy hides class-level failure:** 84% accuracy sounds strong, but the model completely fails on low-quality wines (the minority class). For a winery use case, missing low-quality bottles entirely would be a significant real-world failure.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `Matplotlib`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_KNearestNeighbors_MTK.git
cd ML_KNearestNeighbors_MTK
pip install -r requirements.txt
jupyter notebook src/explore.ipynb
```

## What I'd Do Next

- Address class imbalance with **SMOTE** oversampling on the training set to give the low-quality class enough representation to be learnable
- Try **weighted KNN** (`weights="distance"`) so nearer neighbours have more influence than distant ones
- Compare against a **Random Forest** classifier on the same features to quantify the accuracy ceiling achievable with a non-distance-based method

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
