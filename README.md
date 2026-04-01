# Hands-On Machine Learning Workshop

### Tree-Based Models · Gradient Boosting · XGBoost

A practical, code-first workshop on tree-based and boosting algorithms in Python.
Each session pairs a theory guide (markdown) with hands-on `.py` files that run as interactive notebooks in VS Code using the Jupyter extension.

---

## Repository Structure

```
Hands-on-material/
│
├── DT and RF/
│   ├── hands_on_1.3_decision_trees.py       # Decision Trees (13 sections, 680 lines)
│   ├── hands_on_1.5_random_forest.py        # Random Forest  (10 sections, 455 lines)
│   └── hands_on_1.6_model_comparison.py     # Model Comparison (11 sections, 425 lines)
│
├── GBM and XGBoost/
│   └── hands_on_2_4_2_6_2_7_GBM.py         # GBM + XGBoost + Final Comparison (25 sections, 1177 lines)
│
└── Example_heart_disease/
    ├── session1_heart_disease.py            # Session 1 on Heart Disease dataset (26 sections, 1260 lines)
    └── session2_heart_disease.py            # Session 2 on Heart Disease dataset (23 sections, 1314 lines)
```

---

## Session 1 — Decision Trees & Random Forest

**Folder:** `DT and RF/`

**Dataset:** Breast Cancer Wisconsin — built into `sklearn`, 569 samples, 30 numeric features, binary classification (malignant / benign)

### `hands_on_1.3_decision_trees.py`

Covers Decision Trees from the ground up.

- Load and explore the Breast Cancer dataset
- Train a default Decision Tree (observe overfitting)
- Visualize the tree — how to read nodes, colors, gini, splits
- Understand the bias-variance tradeoff through depth tuning
- Cost-complexity pruning (`ccp_alpha`) — how alpha controls tree size
- Final model evaluation: classification report, confusion matrix, specificity
- Full explanation of precision, recall, F1, and when each matters

### `hands_on_1.5_random_forest.py`

Covers Random Forest step by step.

- Train a Random Forest with OOB validation
- Effect of number of trees — OOB and test score convergence
- Feature importance using MDI (Mean Decrease in Impurity)
- Permutation feature importance — why it is more reliable than MDI
- Visualize a single tree inside the forest
- OOB error convergence plotted tree-by-tree using `warm_start`

### `hands_on_1.6_model_comparison.py`

Compares Decision Tree (default), Decision Tree (tuned), and Random Forest.

- Cross-validation accuracy and standard deviation
- Test accuracy and AUC-ROC score
- ROC curve comparison plot
- Cross-validation score distribution boxplot
- Interpretability summary: tree depth, top split feature, OOB score

---

## Session 2 — Gradient Boosting & XGBoost

**Folder:** `GBM and XGBoost/`

**Dataset:** Breast Cancer Wisconsin (same as Session 1 — all four models compared at the end)

### `hands_on_2_4_2_6_2_7_GBM.py`

**Section 4 — Gradient Boosting (GBM)**

- Manual step-by-step GBM demo on a 1D regression problem — visualizes residual correction across 5 rounds
- Train `GradientBoostingClassifier` with all parameters explained
- Learning rate vs number of trees tradeoff (table of 5 configurations)
- Training loss (deviance) curve with early stopping
- GBM feature importance

**Section 6 — XGBoost**

- Train `XGBClassifier` — new parameters explained vs GBM (gamma, reg_lambda, reg_alpha, colsample_bytree)
- XGBoost with early stopping using `eval_set` and `early_stopping_rounds`
- Training vs validation loss curves
- Regularization effect analysis (gamma / L2 / L1 / combined)
- Three types of XGBoost feature importance: Weight, Gain, Cover
- Hyperparameter tuning with `GridSearchCV`
- SHAP values for model explainability (bonus — requires `pip install shap`)

**Section 7 — Final Comparison**

- All four models: Decision Tree, Random Forest, GBM, XGBoost
- Comparison table: Train Acc, Test Acc, CV Mean, CV Std, AUC-ROC
- ROC curve comparison, performance heatmap, CV boxplot
- Feature importance side-by-side (2×2 grid)
- Confusion matrices side-by-side (2×2 grid)

---

## Example — Heart Disease Practice Dataset

**Folder:** `Example_heart_disease/`

**Dataset:** Cleveland Heart Disease (UCI via OpenML) — 303 patients, 13 clinical features, binary classification (no disease / heart disease present)

Features include age, chest pain type, resting blood pressure, max heart rate, thalassemia type, number of major vessels, and more. Results are medically interpretable, making it a good dataset for checking whether feature importance makes clinical sense.

These two files mirror the exact structure and section flow of Sessions 1 and 2 applied to a new dataset. Use them to independently practice and verify your understanding after the main sessions.

| File                        | Mirrors                 | Sections |
| --------------------------- | ----------------------- | -------- |
| `session1_heart_disease.py` | Full Session 1 workflow | 26       |
| `session2_heart_disease.py` | Full Session 2 workflow | 23       |

---

## Setup

**Requirements**

- Python 3.8 or higher
- VS Code with the **Python** and **Jupyter** extensions installed

**Install dependencies**

```bash
pip install scikit-learn pandas numpy matplotlib seaborn xgboost
```

**Optional — for SHAP explainability (Section 6 of GBM file)**

```bash
pip install shap
```

**Running the files**

Open any `.py` file in VS Code.
Each `# %%` line is a code cell — run with `Shift + Enter`.
Each `# %% [markdown]` line is a markdown description cell.
Use **Run All** from the toolbar to run the entire file at once.

---

## What You Will Learn

**Session 1**

- How Decision Trees split data using Gini impurity and Information Gain
- Why deep trees overfit and how depth limits and pruning fix it
- How Random Forest uses bootstrap sampling and feature randomization to reduce variance
- What OOB score is and why it is a free built-in validation estimate
- How MDI and permutation feature importance differ and when each is reliable

**Session 2**

- How Gradient Boosting works — correcting residuals sequentially
- The learning rate vs number of trees tradeoff
- What XGBoost adds over standard GBM — second-order gradients, regularization, speed
- How to use early stopping and loss curves to find the right number of trees
- How L1, L2, and gamma regularization control overfitting in XGBoost
- How to explain model predictions using SHAP values
- How to compare all four model families in a single evaluation framework

---

## Tools Used

| Tool           | Purpose                                                   |
| -------------- | --------------------------------------------------------- |
| `scikit-learn` | Decision Tree, Random Forest, GBM, metrics, preprocessing |
| `xgboost`      | XGBoost classifier                                        |
| `pandas`       | Data handling                                             |
| `numpy`        | Numerical operations                                      |
| `matplotlib`   | Plotting                                                  |
| `seaborn`      | Statistical visualizations                                |
| `shap`         | Model explainability (optional)                           |

---

## Author

**Sandip Garai**
Hands-On Machine Learning Workshop
