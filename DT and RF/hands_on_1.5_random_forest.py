# %% [markdown]
# ============================================================
# Random Forest - Hands-On Python
# ============================================================
#
# In this notebook we study Random Forest step by step.
#
# Random Forest is an ensemble method based on:
#
# 1. Bagging (Bootstrap Aggregation)
# 2. Decision Trees
# 3. Feature Randomization
#
# Goals of this notebook:
#
# - Train a Random Forest classifier
# - Understand Out-of-Bag validation
# - Study effect of number of trees
# - Understand feature importance
# - Compare MDI vs Permutation importance
# - Visualize trees inside the forest
# - Observe OOB error convergence
#
# Dataset used:
# Breast Cancer dataset from sklearn
#
# ============================================================

# %% [markdown]
# ============================================================
# 1. Setup
# ============================================================
#
# We import required libraries.
#
# numpy       -> numerical operations
# pandas      -> tabular data
# matplotlib  -> plotting
# seaborn     -> better plots
# sklearn     -> machine learning models
#
# We also set a random seed so results are reproducible.
#
# ============================================================
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# 2. Load Dataset
# ============================================================
#
# We use the Breast Cancer dataset from sklearn.
#
# Task:
# Binary classification
#
# Classes:
# 0 -> malignant
# 1 -> benign
#
# Features:
# 30 numeric measurements of cell nuclei
#
# We convert the dataset to pandas DataFrame
# for easier handling and visualization.
#
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

print("Shape:", X.shape)
print("Classes:", cancer.target_names)


# %% [markdown]
# ============================================================
# 3. Train-Test Split
# ============================================================
#
# We split the dataset into:
#
# Training set -> used to train model
# Test set -> used to evaluate model
#
# test_size = 0.2
# -> 20% for testing
#
# stratify=y
# -> keeps class ratio same in both sets
#
# random_state
# -> reproducible results
#
# ============================================================
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# %% [markdown]
# ============================================================
# 5.1 Train a Random Forest
# ============================================================
#
# Random Forest builds many decision trees.
#
# For each tree:
#
# 1. Draw bootstrap sample
# 2. Train decision tree
# 3. Use random subset of features at each split
#
# Final prediction:
#
# Classification -> majority vote
# Regression -> average
#
# Important parameters:
#
# n_estimators
# -> number of trees
#
# max_features
# -> features considered per split
#
# bootstrap
# -> enables bagging
#
# oob_score
# -> use Out-of-Bag samples for validation
#
# n_jobs = -1
# -> use all CPU cores
#
# ============================================================
# %%
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=SEED,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print(f"OOB Score (built-in validation): {rf.oob_score_:.4f}")
print(f"Train Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"Test Accuracy:  {rf.score(X_test, y_test):.4f}")


# %% [markdown]
# ============================================================
# 5.2 Effect of Number of Trees
# ============================================================
#
# Increasing number of trees reduces variance.
#
# Few trees:
# -> unstable model
#
# Many trees:
# -> stable model
#
# But after some point,
# accuracy stops improving.
#
# We plot:
#
# OOB score
# Test accuracy
#
# to see convergence.
# Why?
#
# Random Forest reduces variance by averaging many trees.
#
# When number of trees increases:
#
# - model becomes more stable
# - predictions become less noisy
# - accuracy improves
#
# But after some point,
# adding more trees does not help much.
#
# This is called convergence.
#
# ------------------------------------------------------------
# How to see convergence in the graph
# ------------------------------------------------------------
#
# X-axis  -> number of trees
# Y-axis  -> accuracy
#
# We plot two curves:
#
# OOB score  -> validation accuracy using out-of-bag samples
# Test score -> accuracy on test data
#
# At first:
# accuracy changes a lot
#
# As trees increase:
# curves become flat
#
# When curve becomes flat,
# model has converged.
#
# That means:
# adding more trees will not improve performance.
#
# ------------------------------------------------------------
# What we learn from this plot
# ------------------------------------------------------------
#
# Choose number of trees where:
#
# accuracy is high
# curve is stable
#
# Usually between 100–300 trees is enough.
#
# ============================================================

# ============================================================
# %%
n_trees = [1, 5, 10, 20, 50, 100, 200, 300, 500]
oob_scores = []
test_scores = []

for n in n_trees:
    rf_temp = RandomForestClassifier(
        n_estimators=n, oob_score=True, random_state=SEED, n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    oob_scores.append(rf_temp.oob_score_)
    test_scores.append(rf_temp.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(n_trees, oob_scores,  'o-', label='OOB Score')
plt.plot(n_trees, test_scores, 's-', label='Test Score')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Accuracy vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ============================================================
# 5.3 Feature Importance - MDI
# ============================================================
#
# MDI = Mean Decrease in Impurity
#
# Each split reduces impurity.
#
# Features that reduce impurity more
# are considered more important.
#
# Importance is averaged over all trees.
#
# Advantage:
# fast (computed during training)
#
# Limitation:
# biased toward high-cardinality features
#
# ============================================================
# %%
importances = rf.feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=feat_imp_df.head(15),
    x='Importance',
    y='Feature'
)

plt.title('MDI Importance')
plt.show()

print(feat_imp_df.head(10))


# %% [markdown]
# ============================================================
# 5.4 Permutation Feature Importance
# ============================================================
#
# Instead of impurity,
# we measure importance using accuracy.
#
# Steps:
#
# 1. Compute original accuracy
# 2. Shuffle one feature
# 3. Predict again
# 4. Measure accuracy drop
#
# Large drop -> important feature
#
# More reliable than MDI
# but slower.
#
# ============================================================
# %%
perm_imp = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=SEED,
    n_jobs=-1
)

perm_imp_df = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance_Mean': perm_imp.importances_mean,
    'Importance_Std':  perm_imp.importances_std
}).sort_values('Importance_Mean', ascending=False)

plt.figure(figsize=(10, 8))

plt.barh(
    perm_imp_df['Feature'].head(15)[::-1],
    perm_imp_df['Importance_Mean'].head(15)[::-1]
)

plt.title("Permutation Importance")
plt.show()


# %% [markdown]
# ============================================================
# 5.5 Visualize Individual Trees in the Forest
# ============================================================
#
# Random Forest contains many trees.
#
# We can inspect one tree
# to understand how splits work.
#
# We limit depth for readability.
#
# This tree is only one member
# of the ensemble.
#
# Final prediction uses all trees.
#
# ============================================================
# %%
single_tree = rf.estimators_[0]

plt.figure(figsize=(20, 8))

plot_tree(
    single_tree,
    max_depth=3,
    feature_names=cancer.feature_names,
    class_names=cancer.target_names,
    filled=True
)

plt.show()


# %% [markdown]
# ============================================================
# 5.6 OOB Error Visualization per Tree
# ============================================================
#
# OOB = Out-of-Bag validation
#
# Each tree uses bootstrap sample.
#
# About 37% of samples are left out.
#
# These samples are used as validation.
#
# As number of trees increases:
#
# OOB error decreases
# then stabilizes.
#
# This helps choose number of trees.
#
# ============================================================
# %%
oob_errors = []

rf_incremental = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    warm_start=True,
    random_state=SEED
)

for n in range(1, 201):
    rf_incremental.set_params(n_estimators=n)
    rf_incremental.fit(X_train, y_train)
    oob_errors.append(1 - rf_incremental.oob_score_)

plt.figure(figsize=(10, 5))

plt.plot(range(1, 201), oob_errors)

plt.xlabel("Trees")
plt.ylabel("OOB Error")

plt.show()

# %%
