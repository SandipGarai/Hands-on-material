# %% [markdown]
# ============================================================
# Session 1: Tree-Based Models
# Decision Trees & Random Forest
# ============================================================
#
# Dataset: Heart Disease (Cleveland, UCI)
#
# Task:
# Predict whether a patient has heart disease or not.
#
# This is a binary classification problem.
# 0 = No heart disease
# 1 = Heart disease present
#
# What we cover in this session:
#
# Section 1  -> Setup
# Section 2  -> Load & explore the dataset
# Section 3  -> Train-Test Split
# Section 4  -> Decision Tree (default)
# Section 5  -> Visualize the tree
# Section 6  -> Bias-Variance tradeoff (depth tuning)
# Section 7  -> Cost-Complexity Pruning
# Section 8  -> Final Decision Tree + Evaluation
# Section 9  -> Random Forest
# Section 10 -> Effect of number of trees
# Section 11 -> Feature Importance (MDI)
# Section 12 -> Permutation Feature Importance
# Section 13 -> Visualize a single tree inside the forest
# Section 14 -> OOB Error Convergence
# Section 15 -> Model Comparison (DT vs RF)
#
# ============================================================


# %% [markdown]
# ============================================================
# Section 1: Setup
# ============================================================
#
# Install once in terminal if not already installed:
#
# pip install scikit-learn pandas matplotlib seaborn
#
# Libraries:
#
# numpy      -> numerical operations (arrays, math)
# pandas     -> tabular data (DataFrames)
# matplotlib -> basic plotting
# seaborn    -> nicer statistical plots
# sklearn    -> all machine learning tools
#
# We also set a random seed.
# This makes results reproducible.
# Every time you run the code, you get the same output.
#
# ============================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# Section 2: Load and Explore the Dataset
# ============================================================
#
# We use the Cleveland Heart Disease dataset from OpenML.
#
# This is a real medical dataset with 303 patients.
# Each row is one patient.
# Each column is a clinical measurement.
#
# Features (13 total):
#
# age         -> patient age in years
# sex         -> 1 = male, 0 = female
# cp          -> chest pain type (0 to 3)
#                0 = typical angina
#                1 = atypical angina
#                2 = non-anginal pain
#                3 = asymptomatic
# trestbps    -> resting blood pressure (mm Hg)
# chol        -> serum cholesterol (mg/dl)
# fbs         -> fasting blood sugar > 120 mg/dl (1=yes, 0=no)
# restecg     -> resting ECG results (0, 1, 2)
# thalach     -> maximum heart rate achieved
# exang       -> exercise induced angina (1=yes, 0=no)
# oldpeak     -> ST depression induced by exercise
# slope       -> slope of peak exercise ST segment (0, 1, 2)
# ca          -> number of major vessels colored by fluoroscopy (0-3)
# thal        -> thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
#
# Target:
# 0 = no heart disease
# 1 = heart disease present
#
# Why this dataset?
# Medical data is realistic and the features have meaning.
# You can ask: "Does this make sense medically?"
# That is good practice for real ML work.
#
# ============================================================

# %%
heart = fetch_openml(name='heart-c', version=1, as_frame=True)

X_raw = heart.data.copy()
y_raw = heart.target.copy()

# The target in this OpenML version uses string labels like 'P_0', 'P_1'...'P_4'
# (some versions use '0','1','2','3','4' directly - both are handled below).
# 'P_0' / '0' = no heart disease.  'P_1'...'P_4' / '1'...'4' = disease present.
#
# We use LabelEncoder to convert whatever strings are present to integers
# in sorted order, then collapse everything > 0 to 1 (binary target).
# This works regardless of whether the labels are 'P_0' style or '0' style.
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_raw.astype(str))
y = pd.Series((y_encoded > 0).astype(int), name='target')
print("Original target labels found:", le_target.classes_)

# Drop rows with missing values (a few patients have missing ca or thal).
mask = X_raw.notna().all(axis=1)
X_raw = X_raw[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

# Encode non-numeric columns to numbers.
# sklearn needs numeric input.
#
# Why not just use select_dtypes(include='object')?
# OpenML sometimes returns columns as pandas Categorical dtype
# instead of plain object dtype (e.g., 'sex' comes as category).
# select_dtypes('object') would miss those columns entirely.
#
# The safe approach:
# Step 1 - convert ALL columns to string first (works on any dtype)
# Step 2 - then encode with LabelEncoder
# Step 3 - finally cast the whole DataFrame to float
#
X = X_raw.copy()

# Convert Categorical columns to plain string first,
# then use LabelEncoder to map strings to integers.
for col in X.columns:
    if X[col].dtype.name == 'category' or X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Now all columns are integer-encoded. Cast to float for sklearn.
X = X.apply(pd.to_numeric, errors='coerce').astype(float)

# Store feature names for later use.
FEATURE_NAMES = list(X.columns)
CLASS_NAMES = ['No Disease', 'Heart Disease']

print("Dataset shape:", X.shape)
print("\nFeatures:", FEATURE_NAMES)
print("\nClass distribution:")
print(y.value_counts().rename({0: 'No Disease', 1: 'Heart Disease'}))

# %% [markdown]
# ============================================================
# What does the class distribution tell us?
# ============================================================
#
# We check how many patients are in each class.
#
# If one class has far more patients than the other,
# the dataset is imbalanced.
#
# Example of a problem with imbalanced data:
# If 90% of patients have no disease,
# a model that always predicts "no disease"
# gets 90% accuracy without learning anything.
#
# In the Heart Disease dataset, classes are fairly balanced.
# So accuracy is a reasonable metric here.
#
# ============================================================

# %%
plt.figure(figsize=(5, 4))
y.value_counts().plot(
    kind='bar',
    color=['steelblue', 'salmon'],
    edgecolor='white'
)
plt.xticks([0, 1], CLASS_NAMES, rotation=0)
plt.ylabel('Number of Patients')
plt.title('Class Distribution - Heart Disease Dataset')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Exploring Feature Distributions
# ============================================================
#
# Before training any model, we look at the data.
#
# We plot histograms of key features split by class.
#
# What to look for:
# If the two classes have different distributions for a feature,
# the decision tree will find it easy to split on that feature.
#
# Features where both classes look the same
# are not very useful for prediction.
#
# Try to identify which features seem most separating
# just by looking at the plots.
#
# ============================================================

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

features_to_plot = ['age', 'thalach', 'oldpeak', 'chol', 'trestbps', 'ca']

for ax, feat in zip(axes.flatten(), features_to_plot):
    for label, color in zip([0, 1], ['steelblue', 'salmon']):
        ax.hist(
            X[feat][y == label],
            bins=25,
            alpha=0.6,
            color=color,
            label=CLASS_NAMES[label]
        )
    ax.set_title(feat)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Feature Distributions by Class', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Basic Statistics
# ============================================================
#
# We print summary statistics for each feature.
#
# mean  -> average value
# std   -> spread (standard deviation)
# min   -> smallest value
# max   -> largest value
# 25/50/75% -> percentiles
#
# This helps you understand:
# - Scale of each feature (age is 0-80, chol is 100-600)
# - Whether there are extreme values
# - Whether data looks realistic
#
# Decision Trees do NOT need feature scaling.
# They split by thresholds, not distances.
# So we don't need to normalize or standardize here.
#
# ============================================================

# %%
print("Basic Statistics:\n")
print(X.describe().round(2))


# %% [markdown]
# ============================================================
# Section 3: Train-Test Split
# ============================================================
#
# We split data into two parts:
#
# Training set (80%) -> model learns from this
# Test set (20%)     -> we evaluate model on this
#
# The model never sees test data during training.
# This simulates how the model will perform on new patients.
#
# stratify=y
# -> Ensures both sets have the same class ratio.
# -> Without this, one set might end up with more
#    heart disease patients than the other by chance.
#
# random_state=SEED
# -> Makes the split the same every time you run the code.
#
# ============================================================

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print(
    f"Training set: {X_train.shape[0]} patients, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} patients, {X_test.shape[1]} features")
print(
    f"\nClass ratio in training: {y_train.mean():.2f} (proportion with disease)")
print(f"Class ratio in test:     {y_test.mean():.2f}")


# %% [markdown]
# ============================================================
# Section 4: Train Decision Tree (Default - No Depth Limit)
# ============================================================
#
# A Decision Tree works like a flowchart.
#
# At each node, it asks a yes/no question about a feature:
# "Is thalach <= 150?"
# "Is ca <= 0.5?"
#
# Based on the answer, you go left or right.
# You keep going until you reach a leaf node.
# The leaf tells you the predicted class.
#
# By default, sklearn grows the tree until every leaf is pure.
# Pure means all samples in that leaf belong to one class.
#
# Problem: A fully grown tree memorizes the training data.
# It learns every patient's exact profile.
# On new patients (test set), it does much worse.
# This is called OVERFITTING.
#
# Expected result below:
# Training accuracy = 1.0 (perfect, tree memorized data)
# Test accuracy = noticeably lower
# That gap = overfitting
#
# ============================================================

# %%
dt_default = DecisionTreeClassifier(random_state=SEED)
dt_default.fit(X_train, y_train)

train_acc = dt_default.score(X_train, y_train)
test_acc = dt_default.score(X_test, y_test)

print("=== Decision Tree (No Depth Limit) ===")
print(f"Tree Depth:        {dt_default.get_depth()}")
print(f"Number of Leaves:  {dt_default.get_n_leaves()}")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")
print(f"\nGap (overfit signal): {train_acc - test_acc:.4f}")


# %% [markdown]
# ============================================================
# Section 5: Visualize the Tree
# ============================================================
#
# The full tree has many levels and is unreadable.
# We train a smaller tree (max_depth=3) just for visualization.
#
# How to read the tree visualization:
#
# Each box (node) shows:
#
# feature <= threshold
#   -> the question asked at this node
#   -> if TRUE, go LEFT
#   -> if FALSE, go RIGHT
#
# gini
#   -> impurity of this node
#   -> 0.0 = perfectly pure (one class only)
#   -> 0.5 = maximally mixed (equal classes)
#
# samples
#   -> number of training patients reaching this node
#
# value = [a, b]
#   -> a = count of class 0 (No Disease)
#   -> b = count of class 1 (Heart Disease)
#
# class
#   -> predicted class at this node
#   -> whichever of a or b is larger
#
# Colors:
#   Blue   -> majority class is Heart Disease (class 1)
#   Orange -> majority class is No Disease (class 0)
#   Darker = more pure node
#   Lighter = more mixed node
#
# Read from top (root) to bottom (leaves).
# Leaves are the final predictions.
#
# ============================================================

# %%
dt_vis = DecisionTreeClassifier(max_depth=3, random_state=SEED)
dt_vis.fit(X_train, y_train)

plt.figure(figsize=(22, 8))
plot_tree(
    dt_vis,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    fontsize=9
)
plt.title('Decision Tree (max_depth=3) - Heart Disease Dataset', fontsize=13)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Reading the Tree as Text Rules
# ============================================================
#
# Sometimes it is easier to read the tree as if-else rules.
# export_text does exactly that.
#
# Example rule might look like:
# |--- thal <= 2.5
# |   |--- ca <= 0.5
# |   |   |--- class: No Disease
# |   |--- ca > 0.5
# |   |   |--- class: Heart Disease
#
# This means:
# If thal <= 2.5 AND ca <= 0.5 -> predict No Disease
# If thal <= 2.5 AND ca > 0.5  -> predict Heart Disease
#
# These are the actual decision rules the model learned.
#
# ============================================================

# %%
print("Decision Tree Rules (max_depth=3):\n")
print(export_text(dt_vis, feature_names=FEATURE_NAMES))


# %% [markdown]
# ============================================================
# Section 6: Bias-Variance Tradeoff (Effect of Tree Depth)
# ============================================================
#
# This is one of the most important concepts in machine learning.
#
# Bias  = how much the model is wrong on average
# Variance = how much model predictions change with different data
#
# In Decision Trees, depth controls this tradeoff:
#
# Very shallow tree (depth=1 or 2):
# -> Makes very simple decisions
# -> High bias (misses complex patterns)
# -> Low variance (stable)
# -> Underfitting
#
# Very deep tree (depth=15+):
# -> Memorizes training data
# -> Low bias on training
# -> High variance (results change a lot with new data)
# -> Overfitting
#
# We plot train and test accuracy for each depth.
# The best depth is where test accuracy peaks.
#
# What to look for in the plot:
# -> Train accuracy keeps rising as depth increases
# -> Test accuracy rises then falls or flatlines
# -> The peak of test accuracy is the sweet spot
# -> After that peak, adding depth only hurts generalization
#
# ============================================================

# %%
depths = range(1, 20)
train_scores = []
test_scores = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=SEED)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

best_depth = depths[np.argmax(test_scores)]
print(f"Best depth by test accuracy: {best_depth}")
print(f"Test accuracy at best depth: {max(test_scores):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(depths, train_scores, 'o-', label='Train Accuracy', color='steelblue')
plt.plot(depths, test_scores,  's-', label='Test Accuracy',  color='salmon')
plt.axvline(x=best_depth, linestyle='--', color='green',
            label=f'Best Depth = {best_depth}')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Bias-Variance Tradeoff: Accuracy vs Tree Depth\nHeart Disease Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Section 7: Cost-Complexity Pruning (Post-Pruning)
# ============================================================
#
# Pruning is another way to control overfitting.
#
# Instead of limiting depth before training (pre-pruning),
# we grow the full tree first and then cut branches (post-pruning).
#
# The parameter is ccp_alpha (cost complexity parameter).
#
# How it works:
# alpha = 0      -> no pruning, full tree kept
# alpha = small  -> minor pruning, tree slightly smaller
# alpha = large  -> heavy pruning, very small tree
#
# sklearn gives us all possible alpha values to try
# via cost_complexity_pruning_path().
#
# We train one tree per alpha and find which gives
# the best test accuracy.
#
# Relationship between alpha and depth:
# Higher alpha -> more branches removed -> shallower tree
# Lower alpha  -> fewer branches removed -> deeper tree
#
# ============================================================

# %%
# Compute all alpha values from the full default tree.
path = dt_default.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

alpha_train_acc = []
alpha_test_acc = []
alpha_models = []

for a in alphas:
    clf = DecisionTreeClassifier(random_state=SEED, ccp_alpha=a)
    clf.fit(X_train, y_train)
    alpha_models.append(clf)
    alpha_train_acc.append(clf.score(X_train, y_train))
    alpha_test_acc.append(clf.score(X_test, y_test))

best_alpha = alphas[np.argmax(alpha_test_acc)]
print(f"Best alpha: {best_alpha:.5f}")
print(f"Test accuracy at best alpha: {max(alpha_test_acc):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(alphas, alpha_train_acc, 'o-',
         label='Train Accuracy', color='steelblue')
plt.plot(alphas, alpha_test_acc,  's-', label='Test Accuracy',  color='salmon')
plt.axvline(x=best_alpha, linestyle='--', color='green',
            label=f'Best alpha = {best_alpha:.5f}')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Pruning: Accuracy vs ccp_alpha\nHeart Disease Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# How alpha affects tree size
# ============================================================
#
# This prints a table showing how the tree structure changes
# as alpha increases.
#
# With small alpha -> tree stays large (deep, many leaves)
# With large alpha -> tree gets aggressively pruned (small and simple)
#
# You can see the depth drop as alpha increases.
# This confirms that ccp_alpha controls complexity.
#
# ============================================================

# %%
depth_list = [m.get_depth() for m in alpha_models]
leaf_list = [m.get_n_leaves() for m in alpha_models]

print(f"{'alpha':>10}  {'depth':>6}  {'leaves':>7}")
print("-" * 28)
for a, d, l in zip(alphas, depth_list, leaf_list):
    print(f"{a:>10.5f}  {d:>6}  {l:>7}")

plt.figure(figsize=(10, 4))
plt.plot(alphas, depth_list, 'o-', color='steelblue')
plt.xlabel('ccp_alpha')
plt.ylabel('Tree Depth')
plt.title('Tree Depth vs ccp_alpha')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Section 8: Final Decision Tree + Full Evaluation
# ============================================================
#
# We now train the final Decision Tree using:
# best_depth  -> from the bias-variance analysis
# best_alpha  -> from the pruning analysis
#
# Using both together gives us:
# -> A tree that doesn't grow too deep (depth limit)
# -> A tree with weak branches removed (pruning)
#
# We then evaluate with a full classification report
# and a confusion matrix.
#
# ============================================================

# %%
dt_final = DecisionTreeClassifier(
    max_depth=best_depth,
    ccp_alpha=best_alpha,
    random_state=SEED
)
dt_final.fit(X_train, y_train)

y_pred = dt_final.predict(X_test)

print("=== Final Decision Tree ===")
print(f"Depth:  {dt_final.get_depth()}")
print(f"Leaves: {dt_final.get_n_leaves()}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=CLASS_NAMES).plot(cmap='Blues')
plt.title('Confusion Matrix - Final Decision Tree')
plt.show()


# %% [markdown]
# ============================================================
# Understanding the Confusion Matrix and Metrics
# ============================================================
#
# The confusion matrix has 4 cells:
#
#                    Predicted:         Predicted:
#                    No Disease         Heart Disease
# Actual: No Disease     TN                  FP
# Actual: Heart Disease  FN                  TP
#
# TN = True Negative   -> correctly predicted No Disease
# TP = True Positive   -> correctly predicted Heart Disease
# FP = False Positive  -> predicted Disease, actually No Disease
# FN = False Negative  -> predicted No Disease, actually Disease
#
# Why FN matters most for this problem:
# A False Negative means we MISSED a real heart disease patient.
# That patient won't get treatment when they need it.
# This is more dangerous than a False Positive.
#
# Metrics:
#
# Accuracy = (TP + TN) / total
#   -> Overall correctness. Simple but can mislead on imbalanced data.
#
# Precision = TP / (TP + FP)
#   -> Of all patients we said "has disease", how many actually do?
#   -> High precision = fewer false alarms
#
# Recall = TP / (TP + FN)
#   -> Of all actual disease patients, how many did we catch?
#   -> High recall = fewer missed cases
#   -> For medical diagnosis, this is usually the priority
#
# F1-score = 2 * (precision * recall) / (precision + recall)
#   -> Balances precision and recall into one number
#   -> Useful when both matter
#
# Specificity = TN / (TN + FP)
#   -> Of all healthy patients, how many did we correctly identify?
#
# ============================================================

# %%
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
recall = tp / (tp + fn)

print(f"True Negatives:  {tn}  (correctly predicted No Disease)")
print(f"False Positives: {fp}  (predicted Disease, actually Healthy)")
print(f"False Negatives: {fn}  (MISSED - predicted Healthy, actually Disease)")
print(f"True Positives:  {tp}  (correctly predicted Heart Disease)")
print(f"\nSpecificity (correct healthy detections): {specificity:.4f}")
print(f"Recall     (correct disease detections):  {recall:.4f}")


# %% [markdown]
# ============================================================
# Section 9: Random Forest
# ============================================================
#
# A single Decision Tree has high variance.
# Small changes in the data lead to very different trees.
#
# Random Forest solves this by training many trees
# and averaging their predictions.
#
# How Random Forest works:
#
# Step 1: Draw a bootstrap sample
#   -> Sample N patients WITH replacement from training data
#   -> Some patients appear multiple times
#   -> Some patients are left out (these are OOB samples)
#
# Step 2: Train a Decision Tree on that sample
#   -> At each split, consider only sqrt(features) features
#   -> This randomness makes trees different from each other
#
# Step 3: Repeat for n_estimators trees
#
# Step 4: For a new patient, each tree votes
#   -> Majority vote wins (classification)
#
# Why does this help?
# -> Each tree sees different data and uses different features
# -> Trees are diverse and make different mistakes
# -> When you average many diverse trees, errors cancel out
# -> Result: much more stable and accurate model
#
# Key parameters:
#
# n_estimators = 100
#   -> number of trees in the forest
#   -> more trees = better (up to a point), but slower
#
# max_features = 'sqrt'
#   -> number of features considered at each split
#   -> sqrt(13) ≈ 3 for this dataset
#   -> forces diversity between trees
#
# oob_score = True
#   -> use Out-of-Bag samples as a free validation set
#   -> each tree validates on patients it did NOT train on
#
# bootstrap = True
#   -> enables bootstrap sampling (required for OOB)
#
# n_jobs = -1
#   -> use all CPU cores (trains trees in parallel)
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

print("=== Random Forest ===")
print(f"OOB Score (free validation): {rf.oob_score_:.4f}")
print(f"Training Accuracy:           {rf.score(X_train, y_train):.4f}")
print(f"Test Accuracy:               {rf.score(X_test, y_test):.4f}")


# %% [markdown]
# ============================================================
# What is OOB Score?
# ============================================================
#
# When we draw a bootstrap sample, roughly 37% of patients
# are NOT selected for that tree (they are "out of bag").
#
# We can use these left-out patients to validate the tree
# without touching the test set.
#
# This is a free validation built into Random Forest.
#
# OOB score should be close to test accuracy.
# If they are very different, something might be wrong.
#
# It is also useful when you don't want to do
# a separate cross-validation step.
#
# ============================================================


# %% [markdown]
# ============================================================
# Section 10: Effect of Number of Trees
# ============================================================
#
# More trees = more stable predictions.
# But at some point, adding trees stops helping.
# This is called convergence.
#
# We plot:
# OOB score  -> how accuracy changes as we add trees
# Test score -> same on the held-out test set
#
# What to look for in the plot:
# -> At first (few trees), accuracy fluctuates a lot
# -> As trees increase, both curves stabilize
# -> When the curves become flat, we have enough trees
#
# After convergence, adding more trees wastes compute time
# without improving accuracy.
# Usually 100-200 trees is sufficient for most datasets.
#
# ============================================================

# %%
n_trees_list = [1, 5, 10, 20, 50, 100, 200, 300]
oob_scores_n = []
test_scores_n = []

for n in n_trees_list:
    rf_temp = RandomForestClassifier(
        n_estimators=n, oob_score=True, random_state=SEED, n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    oob_scores_n.append(rf_temp.oob_score_)
    test_scores_n.append(rf_temp.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(n_trees_list, oob_scores_n,  'o-',
         label='OOB Score',  color='steelblue')
plt.plot(n_trees_list, test_scores_n, 's-', label='Test Score', color='salmon')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest: Accuracy vs Number of Trees\nHeart Disease Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Section 11: Feature Importance - MDI (Mean Decrease in Impurity)
# ============================================================
#
# Random Forest tells us which features it found most useful.
#
# MDI (Mean Decrease in Impurity):
# Each time a feature is used to split a node,
# we measure how much that split reduced impurity (Gini).
# We average this reduction across all trees.
# Features used more often and with bigger reductions = more important.
#
# Advantage of MDI:
# -> Fast (computed during training automatically)
# -> Already available after rf.fit()
#
# Limitation of MDI:
# -> Can be biased toward features with many unique values
#    (e.g., a continuous feature like age gets more chances to split)
# -> This is why we also compute permutation importance in Section 12
#
# For the Heart Disease dataset, look for:
# -> thal (thalassemia type) is often the top feature
# -> ca (number of major vessels) is usually very important
# -> thalach (max heart rate) and cp (chest pain type) also rank high
# -> These make medical sense: blood vessel blockage and heart rate
#    are strong indicators of heart disease
#
# ============================================================

# %%
feat_imp_df = pd.DataFrame({
    'Feature':    FEATURE_NAMES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='Blues_r')
plt.title('Random Forest: Feature Importance (MDI)\nHeart Disease Dataset')
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.show()

print("Top 5 most important features (MDI):")
print(feat_imp_df.head(5).to_string(index=False))


# %% [markdown]
# ============================================================
# Section 12: Permutation Feature Importance
# ============================================================
#
# A more reliable way to measure feature importance.
#
# How it works:
#
# Step 1: Compute baseline accuracy on test set
# Step 2: Shuffle the values of one feature (randomly scramble it)
# Step 3: Compute accuracy again with that feature shuffled
# Step 4: Accuracy drop = how important that feature was
#
# Large drop = feature was critical
# Small drop = feature didn't matter much
#
# We repeat this 30 times per feature (n_repeats=30)
# so we get a distribution of importance scores.
# This lets us see both the mean importance and the variability.
#
# Why is this better than MDI?
# -> Measured on test data (real generalization)
# -> Not biased by feature cardinality
# -> Tells us what happens if we didn't have that feature
#
# Limitation: Slower (requires multiple predictions per feature).
#
# Compare the rankings here vs MDI from Section 11.
# If they agree, you can be confident about those features.
# If they disagree, investigate the discrepancy.
#
# ============================================================

# %%
perm_imp = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=SEED,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    'Feature':        FEATURE_NAMES,
    'Importance_Mean': perm_imp.importances_mean,
    'Importance_Std':  perm_imp.importances_std
}).sort_values('Importance_Mean', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(
    perm_df['Feature'][::-1],
    perm_df['Importance_Mean'][::-1],
    xerr=perm_df['Importance_Std'][::-1],
    color='steelblue', alpha=0.8
)
plt.xlabel('Mean Accuracy Drop when Feature is Shuffled')
plt.title('Permutation Feature Importance\nHeart Disease Dataset')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("Top 5 most important features (Permutation):")
print(perm_df.head(5).to_string(index=False))


# %% [markdown]
# ============================================================
# Section 13: Visualize a Single Tree Inside the Forest
# ============================================================
#
# The forest contains 100 trees.
# We can look at any one of them.
#
# This tree is only one member of the ensemble.
# The final prediction uses ALL 100 trees together.
#
# Points to notice:
# -> Individual trees in Random Forest are usually NOT pruned
# -> They grow fully (or with a depth limit)
# -> Their individual accuracy might be lower than the forest
# -> But combined, they are stronger
#
# We use max_depth=3 in plot_tree just for readability.
# The actual tree goes deeper.
#
# ============================================================

# %%
single_tree = rf.estimators_[0]

plt.figure(figsize=(22, 8))
plot_tree(
    single_tree,
    max_depth=3,
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    filled=True,
    fontsize=8
)
plt.title('One Tree from the Random Forest (showing first 3 levels)')
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Section 14: OOB Error Convergence (Tree by Tree)
# ============================================================
#
# This plot shows how OOB error changes as we add one tree at a time.
#
# OOB Error = 1 - OOB Accuracy
# Lower is better.
#
# We use warm_start=True which lets us add trees incrementally
# without retraining from scratch each time.
#
# What to expect:
# -> Error starts high (1-5 trees = noisy)
# -> Error drops quickly as more trees are added
# -> Error flattens out (convergence)
# -> After ~100 trees, adding more trees changes very little
#
# The point where the curve flattens is where we have
# "enough" trees. Training more beyond that is wasteful.
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
plt.plot(range(1, 201), oob_errors, color='steelblue', lw=1.5)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Convergence as Trees are Added\nHeart Disease Dataset')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Section 15: Model Comparison
# ============================================================
#
# We compare all three models:
#
# Decision Tree (Default) -> overfits, no restrictions
# Decision Tree (Tuned)   -> best depth + pruning applied
# Random Forest           -> ensemble of 100 trees
#
# Metrics used:
#
# CV Mean  -> average accuracy across 5-fold cross-validation
#   -> more reliable than a single train/test split
#
# CV Std   -> standard deviation of CV scores
#   -> how stable the model is across different data subsets
#   -> lower std = more reliable model
#
# Test Acc -> accuracy on the held-out test set
#
# AUC-ROC  -> Area Under the ROC Curve
#   -> measures ability to separate classes
#   -> ranges from 0.5 (random) to 1.0 (perfect)
#   -> more informative than accuracy when classes are imbalanced
#
# ============================================================

# %%
models = {
    'DT (Default)': DecisionTreeClassifier(random_state=SEED),
    'DT (Tuned)':   DecisionTreeClassifier(max_depth=best_depth, ccp_alpha=best_alpha, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    cv_sc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'CV Mean':  cv_sc.mean(),
        'CV Std':   cv_sc.std(),
        'Test Acc': model.score(X_test, y_test),
        'AUC-ROC':  roc_auc_score(y_test, y_proba)
    }

print("=" * 68)
print(f"{'Model':<18} {'CV Mean':>8} {'CV Std':>8} {'Test Acc':>9} {'AUC-ROC':>8}")
print("=" * 68)
for name, res in results.items():
    print(f"{name:<18} {res['CV Mean']:>8.4f} {res['CV Std']:>8.4f} "
          f"{res['Test Acc']:>9.4f} {res['AUC-ROC']:>8.4f}")
print("=" * 68)


# %% [markdown]
# ============================================================
# ROC Curve Comparison
# ============================================================
#
# The ROC curve plots:
# X-axis -> False Positive Rate (FPR)
#   -> how often we incorrectly say "Heart Disease" for healthy patients
#
# Y-axis -> True Positive Rate (TPR = Recall)
#   -> how often we correctly catch actual heart disease patients
#
# A perfect model hugs the top-left corner.
# The dashed line is a random classifier (AUC=0.5).
#
# AUC = area under this curve.
# Higher AUC = model separates classes better.
#
# For heart disease:
# We want high True Positive Rate even at some cost of False Positives.
# A missed heart disease case (FN) is more costly than
# an unnecessary follow-up visit (FP).
#
# ============================================================

# %%
plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve Comparison - Heart Disease Dataset')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Cross-Validation Score Distribution
# ============================================================
#
# We run 10-fold cross-validation on the full dataset.
# Each fold gives one accuracy score.
# We plot the distribution of those scores as a boxplot.
#
# What to look for:
# -> The median (red line) = typical accuracy
# -> The box height = variability (narrow is better)
# -> Outlier dots = unusual folds where model struggled
#
# Decision Tree (Default) should have the widest box
# because it overfits some folds and not others.
#
# Random Forest should have a narrower box
# because ensemble averaging makes it more consistent.
#
# ============================================================

# %%
fig, ax = plt.subplots(figsize=(10, 5))

cv_data = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_data[name] = scores

cv_df = pd.DataFrame(cv_data)
cv_df.boxplot(
    ax=ax, grid=False,
    boxprops=dict(color='steelblue', linewidth=2),
    medianprops=dict(color='red', linewidth=2.5),
    whiskerprops=dict(color='steelblue'),
    capprops=dict(color='steelblue'),
    flierprops=dict(marker='o', color='orange')
)
ax.set_ylabel('Accuracy (10-Fold CV)')
ax.set_title('Cross-Validation Score Distribution\nHeart Disease Dataset')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ============================================================
# Interpretability Summary
# ============================================================
#
# We print key facts about the tuned Decision Tree
# and the Random Forest side by side.
#
# Decision Tree is interpretable:
# -> We can print exact rules
# -> We know exactly how each prediction is made
# -> Good for explaining to doctors or patients
#
# Random Forest is more accurate but less interpretable:
# -> We cannot easily read 100 trees
# -> We use feature importance as a proxy for understanding
# -> Good when accuracy is the priority
#
# The top feature from Random Forest is a strong candidate
# for further medical investigation.
#
# ============================================================

# %%
print("\n=== INTERPRETABILITY SUMMARY ===\n")

print("Decision Tree (Tuned):")
print(f"  Depth:       {dt_final.get_depth()}")
print(f"  Leaves:      {dt_final.get_n_leaves()}")
top_feat_idx = dt_final.tree_.feature[0]
print(f"  Root Split:  {FEATURE_NAMES[top_feat_idx]}")

print("\nRandom Forest:")
print(f"  Trees:       {rf.n_estimators}")
print(f"  OOB Score:   {rf.oob_score_:.4f}")
top_rf_feat = feat_imp_df.iloc[0]
print(f"  Top Feature: {top_rf_feat['Feature']}")
print(f"  Importance:  {top_rf_feat['Importance']:.4f}")
