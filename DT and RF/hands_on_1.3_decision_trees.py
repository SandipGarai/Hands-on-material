# %% [markdown]
# ============================================================
# Decision Trees - Hands-On Python
# ============================================================
#
# This file is designed for VS Code + ipykernel.
#
# Each `# %%` creates a code cell.
# Each `# %% [markdown]` creates a markdown cell.
#
# In this notebook we will learn:
#
# 1. How to load a dataset
# 2. How to explore data
# 3. How to train a decision tree
# 4. How to visualize the tree
# 5. How overfitting happens
# 6. Bias–variance tradeoff
# 7. Pruning to reduce overfitting
# 8. Final model evaluation
#
# Goal: Understand Decision Trees both theoretically and practically
# ============================================================


# %% [markdown]
# ============================================================
# 1. Setup
# ============================================================
#
# Install once in terminal:
#
# pip install scikit-learn pandas matplotlib seaborn
#
# Libraries used:
#
# numpy       → numerical operations
# pandas      → data tables
# matplotlib  → plotting
# seaborn     → visualization
# sklearn     → machine learning
#
# ============================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# 2. Load Dataset
# ============================================================
#
# We use Breast Cancer dataset from sklearn.
#
# This is a binary classification dataset.
#
# Target:
# 0 → malignant
# 1 → benign
#
# Features:
# 30 numeric measurements of cell nuclei
#
# We convert data to pandas for easier handling.
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

print("Dataset Shape:", X.shape)
print("\nFeature Names:\n", cancer.feature_names)
print("\nTarget Classes:", cancer.target_names)
print("\nClass Distribution:\n", y.value_counts())


# %% [markdown]
# ============================================================
# 3. Exploratory Data Analysis
# ============================================================
#
# We plot distributions of some important features.
#
# Goal:
# See how features differ between classes.
#
# If distributions are different,
# decision tree can split easily.
# ============================================================

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area"
]

for ax, feat in zip(axes.flatten(), features):

    for label, color in zip([0, 1], ["red", "blue"]):

        ax.hist(
            X[feat][y == label],
            bins=30,
            alpha=0.6,
            color=color,
            label=cancer.target_names[label]
        )

    ax.set_title(feat)
    ax.legend()

plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# 4. Train Test Split
# ============================================================
#
# We split dataset into:
#
# Training set → used to train model
# Test set → used to evaluate model
#
# test_size = 0.2 → 20% test
#
# stratify=y keeps class ratio same
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
# 5. Train Decision Tree (Default)
# ============================================================
#
# Default tree grows until pure.
#
# This usually causes overfitting.
#
# Overfitting:
# Train accuracy high
# Test accuracy lower
#
# We will check:
#
# depth
# leaves
# accuracy
#
# ============================================================

# %%
dt = DecisionTreeClassifier(
    random_state=SEED
)

dt.fit(X_train, y_train)

train_acc = dt.score(X_train, y_train)
test_acc = dt.score(X_test, y_test)

print("Depth:", dt.get_depth())
print("Leaves:", dt.get_n_leaves())
print("Train acc:", train_acc)
print("Test acc :", test_acc)


# %% [markdown]
# ============================================================
# 6. Visualize Tree
# ============================================================
#
# Large trees are hard to read.
#
# We limit depth to 3
# so tree is interpretable.
#
# ============================================================

# %%
dt_small = DecisionTreeClassifier(
    max_depth=3,
    random_state=SEED
)

dt_small.fit(X_train, y_train)

plt.figure(figsize=(18, 6))

plot_tree(
    dt_small,
    feature_names=cancer.feature_names,
    class_names=cancer.target_names,
    filled=True
)

plt.show()

# %% [markdown]
# ============================================================
# Understanding the Decision Tree Visualization (Real Example)
# ============================================================
#
# Each box in the tree is a node.
# The tree is read from top (root) to bottom (leaf).
#
# Each node shows:
#
# feature <= threshold
# gini
# samples
# value = [class0, class1]
# class = predicted class
#
# ------------------------------------------------------------
# How are colors decided?
# ------------------------------------------------------------
#
# Colors are decided from the predicted class at that node.
#
# In this dataset:
#
# cancer.target_names =
# ['malignant', 'benign']
#
# So:
#
# value = [malignant_count, benign_count]
#
# The class with larger count becomes the predicted class.
#
# sklearn colors the node based on that class.
#
# Blue  -> benign (class 1)
# Orange -> malignant (class 0)
#
# Dark color  -> node mostly one class (pure)
# Light color -> node mixed classes
#
# ------------------------------------------------------------
# Example from the ROOT node in our tree
# ------------------------------------------------------------
#
# worst radius <= 16.795
# gini = 0.468
# samples = 455
# value = [170, 285]
# class = benign
#
# Explanation:
#
# samples = 455
# -> total points in this node
#
# value = [170, 285]
# -> 170 malignant
# -> 285 benign
#
# Since benign > malignant,
#
# predicted class = benign
#
# So node color = blue
#
# Because both classes present,
# color is light blue (not pure).
#
# ------------------------------------------------------------
# Example of orange node
# ------------------------------------------------------------
#
# texture error <= 0.473
# gini = 0.1
# samples = 151
# value = [143, 8]
# class = malignant
#
# Here:
#
# 143 malignant
# 8 benign
#
# malignant is majority
#
# so class = malignant
#
# node color = orange
#
# Since almost pure,
# color is dark orange.
#
# ------------------------------------------------------------
# Rule for colors
# ------------------------------------------------------------
#
# value = [class0, class1]
#
# if class1 > class0 → blue
# if class0 > class1 → orange
#
# darker color = more pure node
#
# lighter color = more mixed node
#
# ============================================================
# %% [markdown]
# ============================================================
# 7. Bias-Variance Tradeoff (Depth vs Accuracy)
# ============================================================
#
# In decision trees, max_depth controls model complexity.
#
# Small depth:
# -> tree is simple
# -> high bias (underfitting)
#
# Large depth:
# -> tree very complex
# -> high variance (overfitting)
#
# We train trees with different depths
# and compare train vs test accuracy.
#
# Best depth = highest test accuracy.
#
# ============================================================

# %%
depths = range(1, 20)

train_scores = []
test_scores = []
depth_models = []

for d in depths:

    clf = DecisionTreeClassifier(
        max_depth=d,
        random_state=SEED
    )

    clf.fit(X_train, y_train)

    depth_models.append(clf)

    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))


plt.plot(depths, train_scores, label="Train")
plt.plot(depths, test_scores, label="Test")

plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


best_depth = depths[np.argmax(test_scores)]

print("Best depth:", best_depth)


# %% [markdown]
# ============================================================
# 8. Pruning (Cost Complexity)
# ============================================================
#
# Fully grown tree may overfit.
#
# Pruning removes weak branches.
#
# Parameter:
# ccp_alpha
#
# small alpha -> big tree
# large alpha -> small tree
#
# We choose alpha with best test accuracy.
#
# ============================================================

# %%
path = dt.cost_complexity_pruning_path(
    X_train,
    y_train
)

alphas = path.ccp_alphas

train_acc = []
test_acc = []
alpha_models = []

for a in alphas:

    clf = DecisionTreeClassifier(
        random_state=SEED,
        ccp_alpha=a
    )

    clf.fit(X_train, y_train)

    alpha_models.append(clf)

    train_acc.append(clf.score(X_train, y_train))
    test_acc.append(clf.score(X_test, y_test))


plt.plot(alphas, train_acc, label="Train")
plt.plot(alphas, test_acc, label="Test")

plt.xlabel("alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


best_alpha = alphas[np.argmax(test_acc)]

print("Best alpha:", best_alpha)


# %% [markdown]
# ------------------------------------------------------------
# How ccp_alpha controls tree size
# ------------------------------------------------------------
#
# ccp_alpha does not directly set depth.
#
# Larger alpha -> more pruning
# -> fewer nodes
# -> smaller depth
#
# Small alpha -> almost no pruning
# -> deep tree
#
# max_depth -> limits growth
# ccp_alpha -> grow then prune
#
# ------------------------------------------------------------

# %%
depth_list = [m.get_depth() for m in alpha_models]
leaf_list = [m.get_n_leaves() for m in alpha_models]

for a, d, l in zip(alphas, depth_list, leaf_list):

    print(
        f"alpha={a:.5f} depth={d} leaves={l}"
    )


plt.plot(alphas, depth_list, marker="o")

plt.xlabel("alpha")
plt.ylabel("depth")
plt.title("Depth vs ccp_alpha")

plt.show()


# %% [markdown]
# ============================================================
# 9. Final Model
# ============================================================
#
# We use:
#
# best_depth from section 7
# best_alpha from section 8
#
# Final tree = good bias-variance + pruning
#
# ============================================================

# %%
final_tree = DecisionTreeClassifier(
    random_state=SEED,
    max_depth=best_depth,
    ccp_alpha=best_alpha
)

final_tree.fit(X_train, y_train)

pred = final_tree.predict(X_test)

print(classification_report(
    y_test,
    pred,
    target_names=cancer.target_names
))


cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=cancer.target_names
)

disp.plot()
plt.show()

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

print("Specificity:", specificity)
# %% [markdown]
# ============================================================
# Understanding Model Performance Metrics
# ============================================================
#
# After training the model, we evaluate its performance.
#
# sklearn prints a classification report with:
#
# precision
# recall
# f1-score
# support
# accuracy
# macro avg
# weighted avg
#
# These metrics are computed using the confusion matrix.
#
# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
#
#                Predicted
#              0        1
#
# Actual 0    TN       FP
# Actual 1    FN       TP
#
# TN = true negative
# TP = true positive
# FP = false positive
# FN = false negative
#
# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------
#
# Accuracy = correct predictions / total samples
#
# Accuracy = (TP + TN) / total
#
# Shows overall correctness.
#
# ------------------------------------------------------------
# Precision
# ------------------------------------------------------------
#
# Precision = TP / (TP + FP)
#
# Among predicted positives,
# how many are actually correct.
#
# High precision = few false positives
#
# ------------------------------------------------------------
# Recall (Sensitivity)
# ------------------------------------------------------------
#
# Recall = TP / (TP + FN)
#
# Among actual positives,
# how many we found.
#
# High recall = few false negatives
#
# ------------------------------------------------------------
# Specificity
# ------------------------------------------------------------
#
# Specificity = TN / (TN + FP)
#
# How well we detect negatives.
#
# sklearn does not print it,
# but it comes from confusion matrix.
#
# ------------------------------------------------------------
# F1-score
# ------------------------------------------------------------
#
# F1 = harmonic mean of precision and recall
#
# F1 = 2 * (precision * recall) / (precision + recall)
#
# Used when classes are imbalanced.
#
# ------------------------------------------------------------
# Support
# ------------------------------------------------------------
#
# Number of samples of each class
#
# ------------------------------------------------------------
# Macro average
# ------------------------------------------------------------
#
# Average of metrics across classes
# without considering class size.
#
# All classes weighted equally.
#
# ------------------------------------------------------------
# Weighted average
# ------------------------------------------------------------
#
# Average weighted by class size.
#
# Larger classes affect more.
#
# ------------------------------------------------------------
# Why we use multiple metrics
# ------------------------------------------------------------
#
# Accuracy alone is not enough.
#
# Example:
# If 90% data is benign,
# model can get 90% accuracy
# by always predicting benign.
#
# So we also check precision, recall, f1.
#
# ============================================================
