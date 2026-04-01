# %% [markdown]
# ============================================================
# Session 2: Boosting Algorithms
# Gradient Boosting & XGBoost
# ============================================================
#
# Dataset: Heart Disease (Cleveland, UCI)
# Same dataset as Session 1.
#
# This lets you directly compare boosting models
# against Decision Trees and Random Forest from Session 1.
#
# What we cover in this session:
#
# Section 1  -> Setup
# Section 2  -> Load Dataset (same as Session 1)
# Section 3  -> Manual GBM Demo (conceptual)
# Section 4  -> GradientBoostingClassifier (sklearn)
# Section 5  -> Learning Rate vs Number of Trees
# Section 6  -> Training Loss Curve + Early Stopping
# Section 7  -> GBM Feature Importance
# Section 8  -> XGBoost
# Section 9  -> XGBoost with Early Stopping
# Section 10 -> XGBoost Loss Curves
# Section 11 -> Regularization in XGBoost
# Section 12 -> XGBoost Feature Importance (3 types)
# Section 13 -> Hyperparameter Tuning (GridSearchCV)
# Section 14 -> SHAP Values (Bonus)
# Section 15 -> Final Comparison of All Models
#
# ============================================================


# %% [markdown]
# ============================================================
# Section 1: Setup
# ============================================================
#
# Install in terminal if not already done:
#
# pip install scikit-learn xgboost pandas matplotlib seaborn
# pip install shap  (optional, for Section 14)
#
# New libraries compared to Session 1:
#
# GradientBoostingClassifier -> sklearn's boosting model
# xgboost                    -> XGBoost library
# GridSearchCV               -> systematic hyperparameter search
# DecisionTreeRegressor      -> used in manual GBM demo
#
# ============================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# Section 2: Load Dataset
# ============================================================
#
# Same Heart Disease dataset as Session 1.
#
# We repeat the loading steps here so this file is
# self-contained and can be run independently.
#
# If you already ran Session 1 in the same environment,
# the dataset might already be cached locally by fetch_openml.
# It will load instantly the second time.
#
# Dataset reminder:
# 303 patients, 13 clinical features, binary target
# 0 = No heart disease, 1 = Heart disease present
#
# ============================================================

# %%
heart = fetch_openml(name='heart-c', version=1, as_frame=True)

X_raw = heart.data.copy()
y_raw = heart.target.copy()

# The target uses string labels like 'P_0','P_1'...'P_4' (or '0'..'4').
# LabelEncoder sorts them, then we collapse: 0 = no disease, 1+ = disease.
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_raw.astype(str))
y = pd.Series((y_encoded > 0).astype(int), name='target')
print("Original target labels found:", le_target.classes_)

mask = X_raw.notna().all(axis=1)
X_raw = X_raw[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

# Encode non-numeric columns to numbers.
# OpenML sometimes returns columns as pandas Categorical dtype
# instead of plain object dtype (e.g., 'sex' stored as category).
# select_dtypes('object') would miss those entirely.
#
# Safe fix:
# Check each column by .dtype.name and encode anything
# that is 'category' OR object dtype.
# Then cast to float using pd.to_numeric (handles edge cases).
#
X = X_raw.copy()
for col in X.columns:
    if X[col].dtype.name == 'category' or X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.apply(pd.to_numeric, errors='coerce').astype(float)

FEATURE_NAMES = list(X.columns)
CLASS_NAMES = ['No Disease', 'Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Train: {X_train.shape[0]} patients | Test: {X_test.shape[0]} patients")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {CLASS_NAMES}")


# %% [markdown]
# ============================================================
# Section 3: Manual GBM Demo (Conceptual)
# ============================================================
#
# Before using sklearn's GBM, let's build the core idea manually.
# We use a simple 1D regression problem so we can visualize it.
#
# The core idea of Gradient Boosting:
#
# Round 0 (initial prediction):
#   Predict the mean of y for every sample.
#   This is our starting point. Not good, but a start.
#
# Each subsequent round:
#   1. Compute residuals = actual y - current prediction
#      These residuals are what the model got WRONG.
#
#   2. Train a small Decision Tree to predict the residuals.
#      The tree learns to correct the current mistakes.
#
#   3. Update prediction:
#      new_prediction = old_prediction + learning_rate * tree_prediction
#
# This is called "correcting residuals" or
# "gradient descent in function space."
#
# The learning_rate controls how much we trust each correction.
# Small learning rate = cautious, small steps, needs more rounds.
# Large learning rate = aggressive, fast but can overshoot.
#
# Watch the red curve get closer to the blue dots
# with each round. Each round = one correction step.
#
# ============================================================

# %%
np.random.seed(SEED)
X_demo = np.linspace(0, 10, 100).reshape(-1, 1)
y_demo = np.sin(X_demo).ravel() + np.random.normal(0, 0.15, 100)

# Round 0: predict the mean of all y values
F0 = np.mean(y_demo)
predictions = np.full_like(y_demo, F0)

learning_rate_demo = 0.5
n_rounds = 5

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Plot Round 0
axes[0].scatter(X_demo, y_demo, alpha=0.5, color='steelblue', label='True y')
axes[0].axhline(F0, color='red', lw=2, label=f'F0 = mean = {F0:.2f}')
axes[0].set_title('Round 0: Start with the mean')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

for m in range(1, n_rounds + 1):
    # Step 1: compute residuals (what we got wrong)
    residuals = y_demo - predictions

    # Step 2: fit a tree to predict the residuals
    tree = DecisionTreeRegressor(max_depth=2, random_state=SEED)
    tree.fit(X_demo, residuals)

    # Step 3: update prediction by adding a fraction of the tree's output
    predictions += learning_rate_demo * tree.predict(X_demo)

    mse = np.mean((y_demo - predictions) ** 2)

    ax = axes[m]
    ax.scatter(X_demo, y_demo, alpha=0.3, color='steelblue', label='True y')
    ax.plot(X_demo, predictions, color='red', lw=2, label=f'Prediction F{m}')
    ax.set_title(f'Round {m}: MSE = {mse:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'GBM: Each round trains a tree on the residuals from the previous round\n'
    'Red curve gets closer to blue dots with every correction',
    fontsize=12, y=1.03
)
plt.tight_layout()
plt.savefig('gbm_residual_demo.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# What did we just see?
# ============================================================
#
# Round 0: Flat horizontal line (just the mean).
#          MSE is large. Model knows nothing about the pattern.
#
# Round 1: Tree fits the biggest errors from Round 0.
#          Prediction starts following the curve roughly.
#          MSE drops noticeably.
#
# Round 2-3: Each round corrects smaller residuals.
#            Curve matches the data better each time.
#
# Round 5: MSE is much smaller. Curve follows the data well.
#
# Key insight:
# GBM doesn't change the previous trees.
# It only ADDS a new tree each round to fix leftover errors.
# This is sequential and additive learning.
#
# ============================================================


# %% [markdown]
# ============================================================
# Section 4: Train GradientBoostingClassifier (sklearn)
# ============================================================
#
# Now we apply GBM to our real classification task.
#
# Parameters explained:
#
# n_estimators = 100
#   -> number of boosting rounds (trees)
#   -> each round adds one correction tree
#
# learning_rate = 0.1
#   -> how much each tree contributes
#   -> standard starting value
#   -> must balance with n_estimators (see Section 5)
#
# max_depth = 3
#   -> depth of each individual tree
#   -> GBM uses SHALLOW trees on purpose
#   -> shallow trees = "weak learners"
#   -> deep trees in boosting = overfitting risk
#   -> depth 3-5 is standard for GBM
#
# subsample = 0.8
#   -> use 80% of training samples per round
#   -> same idea as bagging but without replacement
#   -> adds randomness, reduces overfitting
#   -> this variant is called Stochastic GBM
#
# max_features = 'sqrt'
#   -> consider sqrt(features) at each split
#   -> adds more randomness, similar to Random Forest
#
# What to look for in the output:
# Training accuracy should be high (0.95+)
# Test accuracy should be close (not 0.1+ lower)
# If there is a large gap -> overfitting
#
# ============================================================

# %%
gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    max_features='sqrt',
    random_state=SEED
)
gbm.fit(X_train, y_train)

print("=== Gradient Boosting Classifier ===")
print(f"Training Accuracy: {gbm.score(X_train, y_train):.4f}")
print(f"Test Accuracy:     {gbm.score(X_test, y_test):.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, gbm.predict(X_test), target_names=CLASS_NAMES))


# %% [markdown]
# ============================================================
# Section 5: Learning Rate vs Number of Trees
# ============================================================
#
# This is the most important tradeoff in Gradient Boosting.
#
# The rule:
# Small learning_rate -> needs more trees to converge
# Large learning_rate -> needs fewer trees
#
# BUT:
# Large learning_rate + too few trees = underfitting
# Large learning_rate + many trees    = overfitting (overshoots)
#
# Small learning_rate + many trees    = best generalization
# Small learning_rate + few trees     = underfitting (never converged)
#
# Practical starting point:
# lr=0.1, n=100  -> good default for most datasets
# lr=0.05, n=200 -> slightly better generalization, slower to train
# lr=0.01, n=500 -> often best accuracy but much slower
#
# In the table below:
# Compare Train Acc vs Test Acc for each configuration.
# Configurations where Train >> Test = overfitting.
# Configurations where both are low = underfitting.
#
# ============================================================

# %%
lr_configs = [
    {'lr': 1.0,  'n': 10,  'label': 'lr=1.0,  n=10'},
    {'lr': 0.5,  'n': 50,  'label': 'lr=0.5,  n=50'},
    {'lr': 0.1,  'n': 100, 'label': 'lr=0.1,  n=100'},
    {'lr': 0.05, 'n': 200, 'label': 'lr=0.05, n=200'},
    {'lr': 0.01, 'n': 500, 'label': 'lr=0.01, n=500'},
]

print("=== Learning Rate vs n_estimators ===\n")
print(f"{'Config':<22} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 44)

for config in lr_configs:
    gbm_temp = GradientBoostingClassifier(
        n_estimators=config['n'],
        learning_rate=config['lr'],
        max_depth=3,
        random_state=SEED
    )
    gbm_temp.fit(X_train, y_train)
    print(f"{config['label']:<22} "
          f"{gbm_temp.score(X_train, y_train):>10.4f} "
          f"{gbm_temp.score(X_test,  y_test):>10.4f}")


# %% [markdown]
# ============================================================
# Section 6: Training Loss Curve + Early Stopping
# ============================================================
#
# This plot shows how the model's loss (error) changes
# as we add more boosting rounds.
#
# For classification, the loss is log-loss (deviance).
# Lower = model is more confident and correct.
#
# Parameters used here:
#
# n_estimators = 300
#   -> allow up to 300 rounds
#
# validation_fraction = 0.2
#   -> hold out 20% of training data internally for monitoring
#
# n_iter_no_change = 20
#   -> if loss doesn't improve for 20 rounds, stop early
#
# What to look for in the plot:
# -> Loss should generally go DOWN as rounds increase
# -> The green dashed line = where early stopping triggered
# -> If loss went down then back up before stopping:
#    the model was starting to overfit before stopping saved it
# -> If loss is still going down when it stops:
#    more rounds might help (increase n_estimators)
#
# Note: sklearn GBM only tracks training loss internally.
# XGBoost (Section 10) gives BOTH train and validation loss.
# That is more informative.
#
# ============================================================

# %%
gbm_curve = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.2,
    n_iter_no_change=20,
    random_state=SEED
)
gbm_curve.fit(X_train, y_train)

train_deviance = gbm_curve.train_score_

plt.figure(figsize=(10, 5))
plt.plot(train_deviance, label='Training Deviance (Log Loss)',
         color='steelblue', lw=2)
plt.axvline(x=gbm_curve.n_estimators_ - 1, linestyle='--', color='green',
            label=f'Early stopped at round {gbm_curve.n_estimators_}')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss (Deviance)')
plt.title('GBM: Training Loss over Boosting Rounds\nHeart Disease Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gbm_loss_curve.png', dpi=150)
plt.show()

print(f"Early stopping triggered at round: {gbm_curve.n_estimators_}")
print(
    f"Test Accuracy with early stopping: {gbm_curve.score(X_test, y_test):.4f}")


# %% [markdown]
# ============================================================
# Section 7: GBM Feature Importance
# ============================================================
#
# GBM computes feature importance the same way as Random Forest:
# MDI (Mean Decrease in Impurity).
#
# Each feature gets credit based on:
# -> How often it was used to split across all 100 trees
# -> How much each split reduced the loss on average
#
# Compare these results with Session 1 Random Forest importance.
#
# Questions to consider:
# Do GBM and Random Forest agree on the top features?
# If yes -> those features are very likely truly important.
# If no  -> could be correlated features (either works equally well)
#           or each model captured different aspects of the data.
#
# For Heart Disease, strong candidates are:
# thal (thalassemia blood disorder type)
# ca (number of major vessels showing fluoroscopy)
# thalach (maximum heart rate achieved)
# cp (chest pain type)
# oldpeak (ST depression on ECG)
#
# These all have clinical reasoning behind them.
# That gives us more confidence in the model.
#
# ============================================================

# %%
gbm_imp_df = pd.DataFrame({
    'Feature':    FEATURE_NAMES,
    'Importance': gbm.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=gbm_imp_df, x='Importance', y='Feature', palette='magma')
plt.title('GBM: Top Feature Importances\nHeart Disease Dataset')
plt.xlabel('Feature Importance (Mean Decrease in Impurity)')
plt.tight_layout()
plt.savefig('gbm_feature_importance.png', dpi=150)
plt.show()

print("Top 5 features (GBM):")
print(gbm_imp_df.head(5).to_string(index=False))


# %% [markdown]
# ============================================================
# Section 8: XGBoost
# ============================================================
#
# XGBoost = eXtreme Gradient Boosting
#
# Same algorithm as GBM but with several improvements:
#
# 1. Second-order gradients (Taylor expansion)
#    -> More precise update direction at each step
#    -> sklearn GBM uses first-order only
#
# 2. Built-in regularization (L1 and L2)
#    -> Directly penalizes model complexity
#    -> Reduces overfitting without relying only on depth limits
#
# 3. Parallel tree building
#    -> Builds each tree faster using multiple CPU cores
#    -> sklearn GBM builds trees sequentially (slower)
#
# 4. Column subsampling per split level (colsample_bytree)
#    -> Further randomization beyond row subsampling
#
# 5. Handles missing values natively
#    -> GBM requires imputation first
#
# 6. Early stopping with train+validation tracking
#    -> Monitors validation loss to stop at the right point
#    -> Much more informative than sklearn GBM's early stopping
#
# New parameters vs GBM:
#
# colsample_bytree = 0.8
#   -> Use 80% of features per tree (in addition to per-split)
#   -> Adds more diversity between trees
#
# gamma = 0
#   -> Minimum loss reduction needed to make a split
#   -> gamma=0 = split if ANY improvement (default)
#   -> gamma=1 = only split if improvement > 1 (conservative)
#   -> Acts like minimum impurity decrease in sklearn
#
# reg_lambda = 1
#   -> L2 regularization on leaf weights
#   -> Penalizes large weight values
#   -> Default=1, increase to prevent overfitting
#
# reg_alpha = 0
#   -> L1 regularization on leaf weights
#   -> Pushes small weights to exactly zero
#   -> Useful when many features are irrelevant
#
# eval_metric = 'logloss'
#   -> What XGBoost uses internally to measure error
#   -> logloss is standard for binary classification
#
# ============================================================

# %%
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_lambda=1,
    reg_alpha=0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

print("=== XGBoost Classifier ===")
print(f"Training Accuracy: {xgb_model.score(X_train, y_train):.4f}")
print(f"Test Accuracy:     {xgb_model.score(X_test, y_test):.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, xgb_model.predict(
    X_test), target_names=CLASS_NAMES))


# %% [markdown]
# ============================================================
# Section 9: XGBoost with Early Stopping
# ============================================================
#
# We set n_estimators very high (1000)
# and let XGBoost decide when to stop.
#
# early_stopping_rounds = 30
#   -> Stop if validation loss doesn't improve for 30 rounds
#   -> XGBoost monitors the last item in eval_set
#
# eval_set = [(X_train, y_train), (X_test, y_test)]
#   -> validation_0 = training loss (first item)
#   -> validation_1 = validation/test loss (last item)
#   -> XGBoost monitors validation_1 for stopping
#
# After fitting, check:
#
# best_iteration
#   -> the round where validation loss was lowest
#   -> use this value as n_estimators in production
#
# best_score
#   -> the best validation loss achieved
#
# This is more reliable than manually guessing n_estimators.
# You just need to set it high enough and let early stopping find
# the right number automatically.
#
# ============================================================

# %%
xgb_early = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_lambda=2,
    eval_metric='logloss',
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=-1
)
xgb_early.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("=== XGBoost with Early Stopping ===")
print(f"Best Iteration:  {xgb_early.best_iteration}")
print(f"Best Eval Score: {xgb_early.best_score:.4f}")
print(f"Test Accuracy:   {xgb_early.score(X_test, y_test):.4f}")


# %% [markdown]
# ============================================================
# Section 10: XGBoost Loss Curves
# ============================================================
#
# This is the most informative diagnostic plot in XGBoost.
#
# We plot both training AND validation loss curve together.
# This is what GBM (Section 6) couldn't show us easily.
#
# Blue line -> Training loss (what model sees during learning)
# Red line  -> Validation loss (what model does on unseen data)
# Green line -> Where XGBoost decided to stop
#
# How to interpret:
#
# Both lines going down together:
#   -> Model is learning without overfitting. Good.
#
# Training goes down, Validation flattens:
#   -> Model is learning training data but not generalizing
#   -> Overfitting is starting
#   -> Early stopping should trigger here
#
# Training goes down, Validation goes UP:
#   -> Clear overfitting
#   -> Early stopping must trigger before this gets worse
#
# On a small dataset like Heart Disease, the validation loss
# may fluctuate more because each fold has fewer samples.
# On larger datasets, these curves are much smoother.
#
# ============================================================

# %%
results_hist = xgb_early.evals_result()

train_loss = results_hist['validation_0']['logloss']
val_loss = results_hist['validation_1']['logloss']

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss',   color='steelblue', lw=2)
plt.plot(val_loss,   label='Validation Loss', color='salmon',    lw=2)
plt.axvline(x=xgb_early.best_iteration, linestyle='--', color='green',
            label=f'Best Iteration: {xgb_early.best_iteration}')
plt.xlabel('Boosting Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost: Training vs Validation Loss\nHeart Disease Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('xgb_loss_curves.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Section 11: Regularization Effect in XGBoost
# ============================================================
#
# One of XGBoost's advantages over sklearn GBM
# is built-in regularization.
#
# Three parameters control regularization:
#
# gamma (min_split_loss):
#   -> A node is only split if the improvement > gamma
#   -> gamma=0: split on any improvement (like standard GBM)
#   -> gamma=1: only split if improvement > 1
#   -> Prunes the tree DURING construction, not after
#
# reg_lambda (L2 regularization):
#   -> Penalizes the square of leaf weight values
#   -> Encourages smaller, smoother weights
#   -> Default=1, larger = more regularization
#   -> Analogy: Ridge regression applied to the tree leaves
#
# reg_alpha (L1 regularization):
#   -> Penalizes the absolute value of leaf weights
#   -> Pushes small weights to exactly zero (sparse model)
#   -> Useful when many features are irrelevant
#   -> Analogy: Lasso regression applied to tree leaves
#
# When to use each:
# gamma -> when you want to control tree splits aggressively
# lambda -> general purpose, always leave at default (1)
#           increase when overfitting
# alpha -> when you have many features and suspect many are useless
#
# What to look for in the table:
# Train Acc vs Test Acc.
# A big gap means overfitting. More regularization should close it.
# On a small clean dataset, differences are small.
# On noisy real-world data, this matters a LOT more.
#
# ============================================================

# %%
print("=== Regularization Analysis ===\n")
print(f"{'Config':<35} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 57)

reg_configs = [
    {'gamma': 0, 'lambda': 1,   'alpha': 0,
        'label': 'Default (no extra reg)'},
    {'gamma': 1, 'lambda': 1,   'alpha': 0,
        'label': 'gamma=1 (split pruning)'},
    {'gamma': 0, 'lambda': 10,  'alpha': 0,   'label': 'lambda=10 (L2 heavy)'},
    {'gamma': 0, 'lambda': 1,   'alpha': 1,   'label': 'alpha=1 (L1)'},
    {'gamma': 1, 'lambda': 5,   'alpha': 0.5, 'label': 'Combined'},
]

for config in reg_configs:
    m = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        gamma=config['gamma'],
        reg_lambda=config['lambda'],
        reg_alpha=config['alpha'],
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=-1
    )
    m.fit(X_train, y_train)
    print(f"{config['label']:<35} "
          f"{m.score(X_train, y_train):>10.4f} "
          f"{m.score(X_test, y_test):>10.4f}")


# %% [markdown]
# ============================================================
# Section 12: XGBoost Feature Importance (Three Types)
# ============================================================
#
# XGBoost provides THREE different ways to measure feature importance.
# This is more than sklearn GBM (which only gives MDI).
#
# Weight (split count):
#   -> How many times this feature is used to split across all trees
#   -> Simple frequency count
#   -> Limitation: can favor continuous features that split many ways
#   -> A feature used 50 times in shallow splits may be less useful
#      than one used 10 times in highly informative splits
#
# Gain (average improvement per split):
#   -> Average improvement in loss each time this feature is used
#   -> More meaningful than weight
#   -> A feature used rarely but with huge gain = important
#   -> Recommended for feature selection decisions
#
# Cover (average samples per split):
#   -> Average number of patients affected by splits on this feature
#   -> Measures "reach" or "influence coverage"
#   -> High cover = feature affects decisions for many patients
#
# In practice:
# -> Use Gain for deciding which features to keep or remove
# -> Weight can be misleading (high count != high usefulness)
# -> Cover helps when you care about model fairness or coverage
#
# Look at which features rank differently across the 3 plots.
# A feature with high Weight but low Gain = used often but not helpful.
# A feature with high Gain but low Weight = rare but powerful splits.
#
# ============================================================

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

importance_types = ['weight', 'gain', 'cover']
titles = [
    'Weight\n(how often used for splits)',
    'Gain\n(average improvement per split)',
    'Cover\n(average patients per split)'
]

for ax, imp_type, title in zip(axes, importance_types, titles):
    imp_dict = xgb_model.get_booster().get_score(importance_type=imp_type)
    imp_df = (
        pd.DataFrame(list(imp_dict.items()), columns=['Feature', 'Score'])
        .sort_values('Score', ascending=False)
    )
    ax.barh(
        imp_df['Feature'].head(10)[::-1],
        imp_df['Score'].head(10)[::-1],
        color='steelblue', alpha=0.75
    )
    ax.set_xlabel(f'Importance ({imp_type})')
    ax.set_title(f'XGBoost Feature Importance\n{title}')
    ax.grid(True, alpha=0.3)

plt.suptitle('Three Views of XGBoost Feature Importance - Heart Disease Dataset',
             fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig('xgb_feature_importance_types.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Section 13: Hyperparameter Tuning with GridSearchCV
# ============================================================
#
# GridSearchCV tries every combination of hyperparameters
# and selects the best one using cross-validation.
#
# How it works:
# 1. You define a grid of values for each parameter.
# 2. GridSearchCV creates every possible combination.
# 3. For each combination, it runs 5-fold cross-validation.
# 4. It picks the combination with the best average CV score.
# 5. You then evaluate the winner on the test set.
#
# This grid covers:
# n_estimators   -> how many trees (100 or 200)
# learning_rate  -> step size (0.05 or 0.1)
# max_depth      -> tree depth (3, 4, or 6)
# subsample      -> row fraction (0.7 or 0.9)
# colsample_bytree -> column fraction (0.7 or 0.9)
#
# Total combinations here: 2 x 2 x 3 x 2 x 2 = 48
# With 5-fold CV: 48 x 5 = 240 model fits
# This can take a few minutes on a laptop.
#
# Tip for class time:
# Reduce the grid to fewer values per parameter.
# For example: n_estimators=[100], max_depth=[3,4] to run faster.
#
# Important note:
# best_score_ is the CV score on TRAINING data.
# Always check grid_search.score(X_test, y_test) separately.
# CV score and test score should be close.
# If test score >> CV score: something unusual happened.
# If test score << CV score: model may have overfit to CV folds.
#
# ============================================================

# %%
param_grid = {
    'n_estimators':     [100, 200],
    'learning_rate':    [0.05, 0.1],
    'max_depth':        [3, 4, 6],
    'subsample':        [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
}

xgb_base = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=SEED,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score:   {grid_search.best_score_:.4f}")
print(f"Test Accuracy:   {grid_search.score(X_test, y_test):.4f}")


# %% [markdown]
# ============================================================
# Section 14: SHAP Values for Model Explainability (Bonus)
# ============================================================
#
# Install: pip install shap
#
# SHAP = SHapley Additive exPlanations
#
# Standard feature importance (MDI, Weight, Gain) tells you:
# "Which features does the model rely on overall?"
#
# SHAP goes further and answers:
# "For THIS specific patient, WHY did the model predict disease?"
# "Did this feature push the prediction towards disease or away?"
#
# Two types of explanations:
#
# Global (summary_plot):
#   -> Shows all features ranked by average impact
#   -> X-axis = SHAP value = how much it pushes prediction + or -
#   -> Color = actual feature value (red=high, blue=low)
#   -> Example: high thalach (red) with negative SHAP
#               = high heart rate = LESS likely to have disease
#               This makes medical sense.
#
# Local (force_plot):
#   -> For ONE specific patient prediction
#   -> Shows exactly which features pushed prediction toward disease
#      and which pushed it away
#   -> Very useful for explaining a specific prediction to a doctor
#
# When is SHAP important in practice?
# -> Medical diagnosis: doctor needs to know WHY
# -> Credit decisions: regulations require explanation
# -> Any high-stakes decision where "black box" is not acceptable
#
# SHAP is the current gold standard for tree model explainability.
#
# ============================================================

# %%
try:
    import shap

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=FEATURE_NAMES,
        show=False
    )
    plt.title('SHAP Summary Plot - XGBoost\nHeart Disease Dataset')
    plt.tight_layout()
    plt.savefig('xgb_shap_summary.png', dpi=150)
    plt.show()

    print("SHAP installed and computed successfully!")
    print("\nHow to read the plot:")
    print("Each row = one feature")
    print("Each dot = one patient in the test set")
    print("X-axis position = SHAP value (positive = pushes toward Heart Disease)")
    print("Color = actual feature value (red=high, blue=low)")

except ImportError:
    print("SHAP not installed.")
    print("Run: pip install shap")
    print("Then re-run this cell.")
    print("\nSHAP provides per-prediction explanations beyond standard feature importance.")


# %% [markdown]
# ============================================================
# Section 15: Final Comparison - All Models
# ============================================================
#
# Now we compare all four models from Sessions 1 and 2:
#
# Decision Tree  -> single tree, interpretable
# Random Forest  -> bagging ensemble (Session 1)
# GBM            -> sequential boosting (sklearn)
# XGBoost        -> optimized boosting with regularization
#
# This is the payoff section.
# You can see exactly how much each approach gains
# over the previous one on the Heart Disease dataset.
#
# Metrics:
#
# Train Acc
#   -> Does the model fit training data? (high is expected)
#
# Test Acc
#   -> Does it generalize? (key metric)
#
# CV Mean
#   -> Average accuracy across 5 folds
#   -> More reliable than a single test split
#
# CV Std
#   -> Stability. Lower = more consistent model.
#   -> DT usually has highest std (most unstable)
#   -> Ensemble methods have lower std
#
# AUC-ROC
#   -> Probability separation quality
#   -> 0.5 = random, 1.0 = perfect
#   -> Better for comparing models than accuracy alone
#
# ============================================================

# %%
all_models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    'GBM':           GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=3, random_state=SEED),
    'XGBoost':       xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, eval_metric='logloss',
        random_state=SEED, n_jobs=-1),
}

comp_results = {}
for name, model in all_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    comp_results[name] = {
        'Train Acc': model.score(X_train, y_train),
        'Test Acc':  accuracy_score(y_test, y_pred),
        'CV Mean':   cv_acc.mean(),
        'CV Std':    cv_acc.std(),
        'AUC-ROC':   roc_auc_score(y_test, y_proba)
    }

print("\n" + "=" * 75)
print(f"{'Model':<18} {'Train':>7} {'Test':>7} {'CV Mean':>8} {'CV Std':>7} {'AUC-ROC':>8}")
print("=" * 75)
for name, res in comp_results.items():
    print(f"{name:<18} {res['Train Acc']:>7.4f} {res['Test Acc']:>7.4f} "
          f"{res['CV Mean']:>8.4f} {res['CV Std']:>7.4f} {res['AUC-ROC']:>8.4f}")
print("=" * 75)


# %% [markdown]
# ============================================================
# ROC Curves - All Models
# ============================================================
#
# Each curve represents one model's ability to separate
# heart disease patients from healthy ones.
#
# X-axis -> False Positive Rate
#   -> How often we wrongly label a healthy patient as "disease"
#   -> We want this LOW
#
# Y-axis -> True Positive Rate (Recall)
#   -> How often we correctly catch actual heart disease patients
#   -> We want this HIGH
#
# Perfect model -> jumps to top-left immediately (AUC = 1.0)
# Random model  -> follows the diagonal (AUC = 0.5)
#
# For medical diagnosis:
# We prioritize high TPR (don't miss disease cases)
# even at some cost of FPR (some healthy patients get false alarm)
#
# Look at which model stays closest to the top-left corner.
# That model best separates the two classes.
#
# ============================================================

# %%
plt.figure(figsize=(9, 7))
colors = ['royalblue', 'darkorange', 'green', 'red']

for (name, model), color in zip(all_models.items(), colors):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve Comparison - All Models\nHeart Disease Dataset', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_models_roc.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Performance Heatmap
# ============================================================
#
# Same numbers as the table above shown as a heatmap.
# Darker = better score.
#
# This makes it easy to spot at a glance:
# -> Which model dominates on which metric
# -> Whether any model has a weakness (pale cell in a row)
#
# The Decision Tree row usually has one obviously pale cell
# (either CV Std is high, or AUC-ROC is lower).
# Ensemble methods (RF, GBM, XGBoost) should be more uniform.
#
# ============================================================

# %%
metrics_df = pd.DataFrame(comp_results).T

plt.figure(figsize=(9, 5))
sns.heatmap(
    metrics_df[['Train Acc', 'Test Acc', 'CV Mean', 'AUC-ROC']],
    annot=True, fmt='.4f', cmap='YlGnBu',
    linewidths=0.5, vmin=0.75, vmax=1.0,
    cbar_kws={'label': 'Score'}
)
plt.title('Model Performance Heatmap - Heart Disease Dataset')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Cross-Validation Distribution - All Models
# ============================================================
#
# Boxplot of 10-fold CV scores for each model.
#
# What to look for:
#
# Narrow box, high median   -> stable and accurate (best)
# Narrow box, low median    -> stable but not accurate
# Wide box, high median     -> accurate but unreliable
# Wide box, low median      -> worst case
#
# Decision Tree (Default) typically has the widest box.
# It overfits some folds badly depending on which samples it gets.
#
# Random Forest and boosting methods should have narrower boxes
# because averaging/sequential corrections reduce fold-to-fold variance.
#
# Orange outlier dots = folds where the model struggled unusually.
#
# ============================================================

# %%
fig, ax = plt.subplots(figsize=(11, 5))

cv_box_data = {}
for name, model in all_models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_box_data[name] = scores

cv_box_df = pd.DataFrame(cv_box_data)
cv_box_df.boxplot(
    ax=ax, grid=False,
    boxprops=dict(color='steelblue', linewidth=2),
    medianprops=dict(color='red', linewidth=2.5),
    whiskerprops=dict(color='steelblue'),
    capprops=dict(color='steelblue'),
    flierprops=dict(marker='o', color='orange')
)
ax.set_ylabel('Accuracy (10-Fold CV)', fontsize=12)
ax.set_title(
    'Cross-Validation Score Distribution - All Models\nHeart Disease Dataset', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_models_cv_boxplot.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Confusion Matrices - All Models
# ============================================================
#
# Confusion matrices for all 4 models side by side.
#
# Reminder of what each cell means:
#
# Top-left:  True Negative  = correctly said "No Disease"
# Top-right: False Positive = wrongly said "Disease" (false alarm)
# Bot-left:  False Negative = MISSED a real case (dangerous)
# Bot-right: True Positive  = correctly said "Heart Disease"
#
# For this dataset, the bottom-left cell is the critical one.
# False Negatives = patients with heart disease who were missed.
# Compare which model has the fewest False Negatives.
# That model is safest for real clinical use.
#
# ============================================================

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, model) in zip(axes.flatten(), all_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap='Blues', colorbar=False
    )
    ax.set_title(f'{name}\nTest Acc: {accuracy_score(y_test, y_pred):.4f}')

plt.suptitle('Confusion Matrices - All Models\nHeart Disease Dataset',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('all_models_confusion_matrices.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Feature Importance Comparison - All Models
# ============================================================
#
# Each model's view of which features matter most.
#
# If all 4 models agree on the top 3-4 features:
# -> Those features are very reliably important.
# -> Strong evidence for clinical significance.
#
# If models disagree:
# -> Features may be correlated.
#    (e.g., thal and ca might both carry similar signal;
#     each model might pick one or the other)
# -> Each model captured slightly different patterns.
#
# For Heart Disease, check if medically significant features
# rank high across all models:
# -> thal (blood disorder type)
# -> ca (vessel blockage)
# -> cp (chest pain)
# -> thalach (max heart rate)
# -> oldpeak (ECG reading)
#
# Consistency across models = confidence in the finding.
#
# ============================================================

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax, (name, model) in zip(axes.flatten(), all_models.items()):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature':    FEATURE_NAMES,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    ax.barh(
        imp_df['Feature'][::-1],
        imp_df['Importance'][::-1],
        color='steelblue', alpha=0.8, edgecolor='white'
    )
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{name} - Top 10 Features')
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Feature Importance Across All Models\nHeart Disease Dataset',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('all_models_feature_importance.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# Session 2 Summary
# ============================================================
#
# What we learned:
#
# GBM (Gradient Boosting):
# -> Builds trees SEQUENTIALLY, each correcting previous errors
# -> Shallow trees (depth 3-5) work best
# -> learning_rate and n_estimators must be balanced
# -> More sensitive to hyperparameters than Random Forest
#
# XGBoost:
# -> Same idea as GBM but faster and with built-in regularization
# -> Second-order gradients = more precise updates
# -> gamma, reg_lambda, reg_alpha control complexity
# -> colsample_bytree adds extra randomization
# -> Early stopping finds optimal n_estimators automatically
# -> Three types of feature importance (Weight, Gain, Cover)
# -> SHAP provides per-prediction explanations
#
# Key comparison takeaways:
# -> Single Decision Tree: interpretable but unstable
# -> Random Forest: stable, good accuracy, less tuning needed
# -> GBM/XGBoost: highest accuracy, more tuning required
# -> No model is always best -> test on your actual data
#
# For Heart Disease specifically:
# -> Missing a disease case (False Negative) is dangerous
# -> Prioritize models with high Recall for the positive class
# -> Always look at confusion matrices, not just accuracy
#
# ============================================================

