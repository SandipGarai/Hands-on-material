# %% [markdown]
# ============================================================
# Model Interpretation & Comparison - Hands-On Python
# ============================================================
#
# In this notebook we compare different models:
#
# 1. Decision Tree (default)
# 2. Decision Tree (tuned)
# 3. Random Forest
#
# We will study:
#
# Cross-validation accuracy
# Test accuracy
# ROC-AUC score
# ROC curve
# CV score distribution
# Model interpretability
#
# Goal:
# Understand how to compare ML models properly.
#
# ROC Curve
# -> shows how well model separates classes
#
# AUC-ROC
# -> area under ROC curve
# -> higher is better
#
# Cross-validation scores
# -> shows stability of model
# -> less variation = better model
#
# We use all to compare models reliably.
#
# ============================================================
# %% [markdown]
# ============================================================
# 1. Setup
# ============================================================

# %%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# 2. Load Dataset
# ============================================================

# %%
cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

y = pd.Series(
    cancer.target,
    name="target"
)


# %% [markdown]
# ============================================================
# 3. Train-Test Split
# ============================================================

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


# %% [markdown]
# ============================================================
# 4. Train Base Models
# ============================================================

# %%
# default tree
dt_default = DecisionTreeClassifier(
    random_state=SEED
)

dt_default.fit(X_train, y_train)


# %%
# tuned tree (example depth)
best_depth = 4

dt_final = DecisionTreeClassifier(
    max_depth=best_depth,
    random_state=SEED
)

dt_final.fit(X_train, y_train)


# %%
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# %% [markdown]
# ============================================================
# 5. Feature Importance for RF (needed later)
# ============================================================

# %%
feat_imp_df = pd.DataFrame({
    "Feature": cancer.feature_names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)


# %% [markdown]
# ============================================================
# 6. Model Comparison
# ============================================================
#
# Now we compare multiple models.
#
# Models:
#
# Decision Tree (default)
# Decision Tree (tuned)
# Random Forest
#
# We compare using:
#
# Cross-validation accuracy
# Test accuracy
# ROC-AUC score
#
# Why multiple metrics?
#
# Accuracy alone is not enough.
# We also check stability and class separation.
#
# ============================================================


# %% [markdown]
# ============================================================
# 6.1 Comprehensive Model Comparison
# ============================================================
#
# We compute:
#
# CV Mean accuracy
# CV Std deviation
# Test accuracy
# AUC-ROC
#
# Cross-validation:
# splits training data into folds
#
# AUC-ROC:
# measures classification quality
#
# ============================================================

# %%

models = {
    'Decision Tree\n(Default)':  DecisionTreeClassifier(random_state=SEED),
    'Decision Tree\n(Tuned)':    DecisionTreeClassifier(max_depth=best_depth, random_state=SEED),
    'Random Forest\n(100 trees)': RandomForestClassifier(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='accuracy'
    )

    test_score = model.score(X_test, y_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    results[name] = {
        'CV Mean': cv_scores.mean(),
        'CV Std':  cv_scores.std(),
        'Test Acc': test_score,
        'AUC-ROC': auc
    }


print("=" * 65)
print(f"{'Model':<30} {'CV Acc':>8} {'±':>4} {'Test Acc':>9} {'AUC-ROC':>8}")
print("=" * 65)

for name, res in results.items():

    clean_name = name.replace('\n', ' ')

    print(
        f"{clean_name:<30} "
        f"{res['CV Mean']:>8.4f} "
        f"{res['CV Std']:>4.4f} "
        f"{res['Test Acc']:>9.4f} "
        f"{res['AUC-ROC']:>8.4f}"
    )

print("=" * 65)


# %% [markdown]
# ============================================================
# 6.2 ROC Curve Comparison
# ============================================================
#
# ROC curve shows class separation ability.
#
# X-axis -> False Positive Rate
# Y-axis -> True Positive Rate
#
# Better model -> curve closer to top-left
#
# AUC = area under curve
#
# Higher AUC -> better model
#
# ============================================================

# %%
plt.figure(figsize=(8, 6))

for name, model in models.items():

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_proba
    )

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    clean_name = name.replace('\n', ' ')

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{clean_name} (AUC={auc:.3f})"
    )


plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve Comparison')

plt.legend(loc='lower right')

plt.grid(True, alpha=0.3)

plt.show()


# %% [markdown]
# ============================================================
# 6.3 Cross-Validation Score Distribution
# ============================================================
#
# Cross-validation checks model stability.
#
# We run 10-fold CV.
#
# We plot distribution of scores.
#
# Narrow box -> stable model
# Wide box -> unstable model
#
# ============================================================

# %%
fig, ax = plt.subplots(figsize=(10, 5))

cv_data = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=10,
        scoring='accuracy'
    )

    cv_data[name.replace('\n', ' ')] = scores


cv_df = pd.DataFrame(cv_data)

cv_df.boxplot(
    ax=ax,
    grid=False
)

ax.set_ylabel('Accuracy')

ax.set_title('10-Fold Cross-Validation Score Distribution')

plt.xticks(rotation=10)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


# %% [markdown]
# ============================================================
# 6.4 Key Interpretability Summary
# ============================================================
#
# We print important properties of models.
#
# Decision Tree:
# depth
# number of leaves
# top feature
#
# Random Forest:
# number of trees
# OOB score
# most important feature
#
# ============================================================

# %%
print("\n=== MODEL INTERPRETABILITY SUMMARY ===\n")

print("Decision Tree (Tuned)")

print("Depth:", dt_final.get_depth())
print("Leaves:", dt_final.get_n_leaves())

top_feature_index = dt_final.tree_.feature[0]

print(
    "Top Feature:",
    cancer.feature_names[top_feature_index]
)


print("\nRandom Forest")

print("Trees: 100")
print("OOB:", rf.oob_score_)

top_feat = feat_imp_df.iloc[0]

print("Top Feature:", top_feat["Feature"])
print("Importance:", top_feat["Importance"])

# %%
