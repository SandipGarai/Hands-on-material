# %% [markdown]
# ============================================================
# Session 2 - Hands-On Python
# Gradient Boosting & XGBoost
# ============================================================
#
# What we cover in this file:
#
# Section 4  -> Gradient Boosting (GBM)
# Section 6  -> XGBoost
# Section 7  -> Final model comparison (all models)
#
# We use the same Breast Cancer dataset from Session 1
# so you can directly compare results.
#
# ============================================================


# %% [markdown]
# ============================================================
# Setup
# ============================================================
#
# Same imports as Session 1, plus:
#
# GradientBoostingClassifier -> sklearn's GBM
# xgboost                    -> XGBoost library
# GridSearchCV               -> hyperparameter tuning
# DecisionTreeRegressor      -> used in the manual GBM demo
#
# ============================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score,
                             roc_curve, accuracy_score)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# 4.1 Load Dataset
# ============================================================
#
# Same dataset as Session 1.
#
# Breast Cancer Wisconsin dataset.
#
# Task:
# Classify tumor as malignant (0) or benign (1)
#
# Features:
# 30 numeric measurements of cell nuclei
# (radius, texture, perimeter, area, smoothness, etc.)
#
# We split into train/test with stratify=y
# so both sets have the same class ratio.
#
# ============================================================

# %%
cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target, name='target')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# %% [markdown]
# ============================================================
# 4.2 Manual Step-by-Step GBM (Conceptual Demo)
# ============================================================
#
# Before using sklearn's GBM, let's build it manually
# on a simple 1D regression problem.
#
# Why?
# Because GBM is often explained abstractly.
# This demo makes it concrete.
#
# The idea behind GBM:
#
# Round 0:
#   Start with a constant prediction = mean of y
#
# Round 1, 2, 3... (each round):
#   Compute residuals = actual - current prediction
#   Fit a small decision tree to those residuals
#   Update prediction by adding: learning_rate * tree_prediction
#
# Every round, the model corrects its own mistakes.
# This is what "boosting" means.
#
# --------------------------------------------------------
# Why learning rate matters here:
#
# If learning_rate = 1.0:
#   We trust each tree fully -> risk of overfitting
#
# If learning_rate = 0.1 or 0.5:
#   We take small steps -> more stable, less overfit
#   But we need more rounds to converge
#
# Watch how the red curve gets closer to the blue dots
# with each round in the plot below.
#
# ============================================================

# %%
np.random.seed(SEED)
X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
y_simple = np.sin(X_simple).ravel() + np.random.normal(0, 0.1, 100)

# Round 0: predict the mean
F0 = np.mean(y_simple)
predictions = np.full_like(y_simple, F0)

learning_rate = 0.5
n_rounds = 5

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

axes[0].scatter(X_simple, y_simple, alpha=0.5, label='True', color='steelblue')
axes[0].axhline(F0, color='red', label=f'F0 = {F0:.2f}')
axes[0].set_title('Round 0: Initial Prediction (Mean)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for m in range(1, n_rounds + 1):
    residuals = y_simple - predictions

    tree = DecisionTreeRegressor(max_depth=2, random_state=SEED)
    tree.fit(X_simple, residuals)

    predictions += learning_rate * tree.predict(X_simple)

    mse = np.mean((y_simple - predictions) ** 2)

    ax = axes[m]
    ax.scatter(X_simple, y_simple, alpha=0.3, label='True', color='steelblue')
    ax.plot(X_simple, predictions, color='red', lw=2, label=f'F{m}')
    ax.set_title(f'Round {m}: MSE = {mse:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('GBM: Each round corrects the residuals from the previous round',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('gbm_residual_demo.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 4.3 Train scikit-learn GradientBoostingClassifier
# ============================================================
#
# Now we switch to classification (cancer dataset).
#
# Key parameters explained:
#
# n_estimators = 100
#   -> number of trees (boosting rounds)
#   -> more rounds = more correction, but also slower
#
# learning_rate = 0.1
#   -> how much each tree contributes
#   -> small value = cautious, needs more trees
#
# max_depth = 3
#   -> GBM works best with shallow trees (depth 3-5)
#   -> deep trees + boosting = overfitting
#
# subsample = 0.8
#   -> use 80% of training samples per tree
#   -> adds randomness, reduces variance (Stochastic GBM)
#
# max_features = 'sqrt'
#   -> consider sqrt(n_features) at each split
#   -> same idea as Random Forest, adds randomness
#
# Expected output:
# Training accuracy will be high (model fits well).
# Test accuracy should be close to training.
# If there is a big gap -> overfitting.
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
print(classification_report(y_test, gbm.predict(X_test),
                            target_names=cancer.target_names))


# %% [markdown]
# ============================================================
# 4.4 Effect of Learning Rate and Number of Trees
# ============================================================
#
# This is the most important tradeoff in GBM.
#
# The rule of thumb:
#
# Small learning rate -> needs more trees
# Large learning rate -> needs fewer trees
#
# But large learning rate with few trees = underfit
# Large learning rate with many trees = overfit
#
# The product (learning_rate * n_estimators) roughly controls
# how much the model learns overall.
#
# Look at the table below:
#
# lr=1.0, n=10:
#   Very aggressive learning, very few trees.
#   High train accuracy but possibly poor test accuracy.
#
# lr=0.1, n=100:
#   Classic default config. Balanced.
#
# lr=0.01, n=500:
#   Very cautious, many trees. Slower but often more robust.
#
# In practice: start with lr=0.1, n=100.
# Then tune from there.
#
# ============================================================

# %%
configs = [
    {'lr': 1.0,  'n': 10,  'label': 'lr=1.0, n=10'},
    {'lr': 0.5,  'n': 50,  'label': 'lr=0.5, n=50'},
    {'lr': 0.1,  'n': 100, 'label': 'lr=0.1, n=100'},
    {'lr': 0.05, 'n': 200, 'label': 'lr=0.05, n=200'},
    {'lr': 0.01, 'n': 500, 'label': 'lr=0.01, n=500'},
]

results_lr = {}
for config in configs:
    gbm_temp = GradientBoostingClassifier(
        n_estimators=config['n'],
        learning_rate=config['lr'],
        max_depth=3,
        random_state=SEED
    )
    gbm_temp.fit(X_train, y_train)
    results_lr[config['label']] = {
        'train': gbm_temp.score(X_train, y_train),
        'test':  gbm_temp.score(X_test, y_test)
    }

print("\n=== Learning Rate vs n_estimators Tradeoff ===")
print(f"{'Config':<25} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 47)
for label, scores in results_lr.items():
    print(f"{label:<25} {scores['train']:>10.4f} {scores['test']:>10.4f}")


# %% [markdown]
# ============================================================
# 4.5 Training Deviance (Loss) Curve
# ============================================================
#
# This plot shows how the training loss changes
# with each boosting round.
#
# What is deviance?
# -> Log loss (for classification)
# -> Measures how wrong the model is
# -> Lower is better
#
# What to look for in this plot:
#
# Loss should keep going down during training.
# If it starts going back up -> overfitting.
#
# We also use early stopping here.
# n_iter_no_change = 20 means:
#   If loss doesn't improve for 20 rounds, stop.
#
# The green dashed line shows where it stopped.
# This is the best number of trees for this configuration.
#
# Note: sklearn's GBM only gives training deviance directly.
# For proper train vs validation tracking, use XGBoost
# (covered in Section 6.4).
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
plt.plot(train_deviance, label='Training Deviance', color='steelblue', lw=2)
plt.xlabel('Boosting Iterations (Trees)')
plt.ylabel('Log Loss (Deviance)')
plt.title('GBM: Training Loss over Boosting Rounds')
plt.axvline(x=gbm_curve.n_estimators_ - 1, color='green', linestyle='--',
            label=f'Early stop at round: {gbm_curve.n_estimators_}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gbm_loss_curve.png', dpi=150)
plt.show()

print(f"Stopped at iteration: {gbm_curve.n_estimators_}")
print(f"Test Accuracy:        {gbm_curve.score(X_test, y_test):.4f}")


# %% [markdown]
# ============================================================
# 4.6 GBM Feature Importance
# ============================================================
#
# GBM also provides feature importance scores.
# Same idea as Random Forest (Mean Decrease in Impurity (MDI) - impurity reduction).
#
# Each feature gets a score based on:
# How often it was used for splitting
# How much it reduced the loss on average
#
# Top features here should look similar to
# what we saw in Session 1 with Random Forest.
#
# If they are very different, that is worth investigating.
# Different models find different patterns.
#
# ============================================================

# %%
gbm_importances = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': gbm.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(data=gbm_importances.head(15),
            x='Importance', y='Feature',
            palette='magma')
plt.title('Top 15 Feature Importances - Gradient Boosting')
plt.xlabel('Importance Score (Gini Reduction)')
plt.tight_layout()
plt.savefig('gbm_feature_importance.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 6. XGBoost - Hands-On Python
# ============================================================
#
# XGBoost = eXtreme Gradient Boosting
#
# Same idea as GBM but with several improvements:
#
# 1. Regularization built in (L1 and L2)
#    -> helps control overfitting explicitly
#
# 2. Second-order gradients
#    -> more accurate updates than standard GBM
#
# 3. Parallel tree building
#    -> much faster training
#
# 4. Sparse data handling
#    -> handles missing values natively
#
# 5. Early stopping support
#    -> monitors validation loss, stops automatically
#
# New parameters compared to sklearn GBM:
#
# colsample_bytree
#   -> fraction of features used per tree (like max_features)
#
# gamma
#   -> minimum loss reduction required to split a node
#   -> higher gamma = more conservative splits
#
# reg_lambda
#   -> L2 regularization on leaf weights
#   -> default=1, increase to reduce overfitting
#
# reg_alpha
#   -> L1 regularization on leaf weights
#   -> pushes small weights to zero
#
# ============================================================


# %% [markdown]
# ============================================================
# 6.1 Install XGBoost
# ============================================================
#
# Run this in terminal before running the code:
#
# pip install xgboost
#
# ============================================================


# %% [markdown]
# ============================================================
# 6.2 Basic XGBoost Training
# ============================================================
#
# We start with a standard XGBoost setup.
#
# n_estimators = 200
#   -> we allow up to 200 trees
#
# learning_rate = 0.1
#   -> standard starting point
#
# max_depth = 4
#   -> slightly deeper than typical GBM
#   -> XGBoost handles this better due to regularization
#
# subsample = 0.8 and colsample_bytree = 0.8
#   -> use 80% of rows and 80% of features per tree
#   -> reduces overfitting
#
# gamma = 0
#   -> no pruning penalty (default)
#
# reg_lambda = 1
#   -> default L2 regularization
#
# eval_metric = 'logloss'
#   -> what XGBoost uses internally to measure error
#
# Compare these numbers with the GBM results from section 4.3.
# XGBoost usually gets similar or slightly better accuracy,
# and trains noticeably faster.
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
print(classification_report(y_test, xgb_model.predict(X_test),
                            target_names=cancer.target_names))


# %% [markdown]
# ============================================================
# 6.3 XGBoost with Early Stopping
# ============================================================
#
# We set n_estimators very high (1000)
# but let early stopping decide when to stop.
#
# early_stopping_rounds = 30 means:
#   If validation loss doesn't improve for 30 rounds, stop.
#
# We pass an eval_set with both train and test data.
# XGBoost monitors the last eval_set entry (test) for stopping.
#
# Why is this useful?
#
# Without early stopping, you need to guess n_estimators.
# With early stopping, XGBoost finds the right number for you.
#
# After running, check:
# best_iteration -> the round where it stopped
# best_score     -> the best validation score
#
# You can reuse best_iteration as n_estimators in production.
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

print(f"Best iteration: {xgb_early.best_iteration}")
print(f"Best eval score: {xgb_early.best_score:.4f}")
print(f"Test Accuracy:   {xgb_early.score(X_test, y_test):.4f}")

# %% [markdown]
# Best eval score: 0.0874 = minimum log loss achieved on the validation set

# %% [markdown]
# ============================================================
# 6.4 Plot Training History (Loss Curves)
# ============================================================
#
# This is one of the most useful diagnostic plots in XGBoost.
#
# Blue line -> training loss
# Red line  -> validation loss
#
# What to watch for:
#
# Both going down = model is learning
#
# Training goes down but validation flattens/goes up:
#   -> overfitting has started
#   -> the green line marks where XGBoost stopped
#
# Both lines going down together till the end:
#   -> could train more (increase n_estimators)
#
# This plot is more informative than sklearn GBM's deviance
# because we can see both train AND validation loss.
#
# Note: on a small dataset like this, the curves stabilize
# quickly. On larger datasets you'll see more movement.
#
# ============================================================

# %%
results_history = xgb_early.evals_result()

train_loss = results_history['validation_0']['logloss']
val_loss = results_history['validation_1']['logloss']

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss',   color='steelblue', lw=2)
plt.plot(val_loss,   label='Validation Loss', color='salmon',    lw=2)
plt.axvline(x=xgb_early.best_iteration, linestyle='--',
            color='green', label=f'Best Iteration: {xgb_early.best_iteration}')
plt.xlabel('Boosting Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost: Training vs Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('xgb_loss_curves.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 6.5 Regularization Effect on Performance
# ============================================================
#
# XGBoost has three regularization parameters:
#
# gamma (min_split_loss)
#   -> Controls whether a node gets split at all
#   -> gamma=0: split if any improvement (default)
#   -> gamma=1: only split if improvement > 1
#   -> Pruning happens during tree building itself
#
# reg_lambda (L2 regularization)
#   -> Penalizes large leaf weight values
#   -> Default = 1
#   -> Higher = smoother, more conservative model
#
# reg_alpha (L1 regularization)
#   -> Like L2 but pushes weights to exactly zero
#   -> Useful when you have many irrelevant features
#
# In the table below:
# Train Acc vs Test Acc tells you about overfitting.
#
# A big gap (train >> test) = overfitting
# Adding regularization should close that gap.
#
# On a small clean dataset like this, differences
# will be small. On noisy real-world data, regularization
# makes a much bigger difference.
#
# ============================================================

# %%
print("=== Regularization Analysis ===\n")
print(f"{'Config':<35} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 57)

reg_configs = [
    {'gamma': 0, 'lambda': 1,   'alpha': 0,
        'label': 'No extra reg (default)'},
    {'gamma': 1, 'lambda': 1,   'alpha': 0,
        'label': 'gamma=1 (node pruning)'},
    {'gamma': 0, 'lambda': 10,  'alpha': 0,   'label': 'lambda=10 (L2 heavy)'},
    {'gamma': 0, 'lambda': 1,   'alpha': 1,   'label': 'alpha=1 (L1)'},
    {'gamma': 1, 'lambda': 5,   'alpha': 0.5, 'label': 'Combined regularization'},
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
    print(f"{config['label']:<35} {m.score(X_train, y_train):>10.4f} "
          f"{m.score(X_test, y_test):>10.4f}")


# %% [markdown]
# ============================================================
# 6.6 XGBoost Feature Importance
# ============================================================
#
# XGBoost offers three different ways to measure importance.
# This is more than sklearn's GBM (which only gives MDI).
#
# Weight:
#   -> How many times a feature was used to split
#   -> Simple count
#   -> Can be biased toward features with many unique values
#
# Gain:
#   -> Average improvement in loss when this feature is used
#   -> More meaningful than weight
#   -> Recommended for most use cases
#
# Cover:
#   -> Average number of samples affected by splits on this feature
#   -> Tells you the "reach" of each feature
#
# Rule of thumb:
# Use Gain for feature selection decisions.
# Weight can be misleading.
# Cover is useful when you care about sample coverage.
#
# Notice how the rankings can differ across the three plots.
# A feature with high weight but low gain is used a lot
# but doesn't actually help much each time.
#
# ============================================================

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

importance_types = ['weight', 'gain', 'cover']
titles = ['Weight (split count)', 'Gain (avg improvement)',
          'Cover (avg coverage)']

for ax, imp_type, title in zip(axes, importance_types, titles):
    imp_dict = xgb_model.get_booster().get_score(importance_type=imp_type)
    imp_df = pd.DataFrame(list(imp_dict.items()),
                          columns=['Feature', 'Score']).sort_values('Score', ascending=False)

    ax.barh(imp_df['Feature'].head(10)[::-1],
            imp_df['Score'].head(10)[::-1],
            color='steelblue', alpha=0.7)
    ax.set_xlabel(f'Importance ({imp_type})')
    ax.set_title(f'XGBoost Feature Importance\n{title}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgb_feature_importance_types.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 6.7 Hyperparameter Tuning with GridSearchCV
# ============================================================
#
# We use GridSearchCV to search over multiple hyperparameters.
#
# GridSearchCV tries every combination in param_grid
# and picks the one with the best cross-validation score.
#
# This grid searches over:
#
# n_estimators    -> how many trees
# learning_rate   -> step size
# max_depth       -> tree depth
# subsample       -> row sampling
# colsample_bytree-> column sampling
#
# cv=5 means 5-fold cross-validation for each combination.
#
# Warning: this can take a few minutes.
# The comment "reduce param_grid for faster execution"
# is a hint to use fewer values during class time.
#
# After fitting, check:
#
# best_params_  -> the winning combination
# best_score_   -> its cross-validation accuracy
# Then evaluate on test set to confirm.
#
# Important: GridSearchCV result shows the best CV score,
# not the test score. Always evaluate on test set separately.
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
# 6.8 SHAP Values for Model Explainability (Bonus)
# ============================================================
#
# SHAP = SHapley Additive exPlanations
#
# Standard feature importance tells you which features
# the model used most. But it doesn't tell you:
#   - Does this feature push predictions up or down?
#   - How does it behave for a specific patient/sample?
#
# SHAP answers both questions.
#
# Global explanation (summary_plot):
#   -> Shows all features ranked by average impact
#   -> Color = feature value (red = high, blue = low)
#   -> X-axis = how much it pushes prediction + or -
#
# Local explanation (force_plot):
#   -> For a single prediction
#   -> Shows exactly why the model predicted what it did
#
# SHAP is the gold standard for explaining tree models.
# If you're building models that affect real decisions
# (medical, financial, etc.), SHAP is very important.
#
# Install: pip install shap
#
# ============================================================

# %%
try:
    import shap

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test,
                      feature_names=cancer.feature_names,
                      show=False)
    plt.title('SHAP Summary Plot - XGBoost')
    plt.tight_layout()
    plt.savefig('xgb_shap_summary.png', dpi=150)
    plt.show()

    print("SHAP installed and computed successfully!")

except ImportError:
    print("SHAP not installed. Run: pip install shap")
    print("SHAP gives both global and per-prediction explanations.")


# %% [markdown]
# ============================================================
# 7. Model Comparison - All Models Together
# ============================================================
#
# Now we compare all four models side by side:
#
# Decision Tree  -> single interpretable tree
# Random Forest  -> bagging + random features
# GBM            -> sequential boosting (sklearn)
# XGBoost        -> optimized boosting with regularization
#
# This is the final comparison that ties Sessions 1 and 2 together.
#
# What to look for:
#
# Train Acc vs Test Acc:
#   -> big gap = overfitting
#
# CV Mean and CV Std:
#   -> low std = stable model
#   -> high std = model is sensitive to data split
#
# AUC-ROC:
#   -> class separation quality
#   -> closer to 1.0 = better
#
# No single metric tells the full story.
# Use all of them together.
#
# ============================================================


# %% [markdown]
# ============================================================
# 7.1 Comprehensive Comparison: All Models
# ============================================================
#
# We train each model and collect:
#
# Train Accuracy  -> how well model fits training data
# Test Accuracy   -> how well model generalizes
# CV Mean         -> average cross-validation accuracy
# CV Std          -> stability across folds
# AUC-ROC         -> quality of probability predictions
#
# The Decision Tree row is our baseline.
# Each subsequent model should generally do better
# on CV Mean and AUC-ROC.
#
# ============================================================

# %%
all_models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    'GBM':           GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                max_depth=3, random_state=SEED),
    'XGBoost':       xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, eval_metric='logloss',
                                       random_state=SEED, n_jobs=-1),
}

comparison_results = {}

for name, model in all_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv_acc = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    comparison_results[name] = {
        'Train Acc': model.score(X_train, y_train),
        'Test Acc':  accuracy_score(y_test, y_pred),
        'CV Mean':   cv_acc.mean(),
        'CV Std':    cv_acc.std(),
        'AUC-ROC':   roc_auc_score(y_test, y_proba)
    }

print("\n" + "=" * 75)
print(f"{'Model':<18} {'Train':>7} {'Test':>7} {'CV Mean':>8} {'CV ±':>7} {'AUC-ROC':>8}")
print("=" * 75)
for name, res in comparison_results.items():
    print(f"{name:<18} {res['Train Acc']:>7.4f} {res['Test Acc']:>7.4f} "
          f"{res['CV Mean']:>8.4f} {res['CV Std']:>7.4f} {res['AUC-ROC']:>8.4f}")
print("=" * 75)


# %% [markdown]
# ============================================================
# 7.2 ROC Curves - All Models
# ============================================================
#
# Each curve shows how well the model separates classes.
#
# X-axis -> False Positive Rate
#   -> proportion of actual negatives predicted as positive
#
# Y-axis -> True Positive Rate (Recall)
#   -> proportion of actual positives correctly caught
#
# A perfect model goes straight to the top-left corner.
# The dashed line is a random classifier (no skill).
#
# AUC = area under the curve.
# Higher AUC = better separation = more useful model.
#
# For cancer detection specifically:
# We want high True Positive Rate (catch real cases).
# We can tolerate some False Positives.
# So the upper-left region of the plot matters most.
#
# ============================================================

# %%
plt.figure(figsize=(9, 7))

colors = ['royalblue', 'darkorange', 'green', 'red']
for (name, model), color in zip(all_models.items(), colors):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, lw=2, color=color,
             label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve Comparison - All Models', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('all_models_roc.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 7.3 Performance Heatmap
# ============================================================
#
# Same numbers as the table in 7.1, shown as a heatmap.
#
# Darker color = higher score.
#
# This makes it easy to spot:
# -> Which model is strongest on which metric
# -> Where Decision Tree falls short vs ensemble methods
# -> Whether boosting improves over bagging (RF) or not
#
# On the breast cancer dataset, all models tend to do well
# because the data is relatively clean and structured.
# The differences become more apparent on noisier datasets.
#
# ============================================================

# %%
metrics_df = pd.DataFrame(comparison_results).T

plt.figure(figsize=(8, 5))
sns.heatmap(
    metrics_df[['Train Acc', 'Test Acc', 'CV Mean', 'AUC-ROC']],
    annot=True, fmt='.4f', cmap='YlGnBu',
    linewidths=0.5, vmin=0.85, vmax=1.0,
    cbar_kws={'label': 'Score'}
)
plt.title('Model Performance Comparison Heatmap')
plt.tight_layout()
plt.savefig('performance_heatmap.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 7.4 Cross-Validation Distribution Comparison
# ============================================================
#
# This boxplot shows stability across 10 folds.
#
# Each fold is a different train/validation split.
# If the model scores are consistent across folds,
# the box will be narrow and the median will be stable.
#
# Wide box or outlier points = unstable model.
# That means the model is sensitive to which samples it sees.
#
# What we expect to see:
#
# Decision Tree -> widest box (most variance, least stable)
# Random Forest -> narrower (bagging reduces variance)
# GBM, XGBoost -> narrow and high (stable + accurate)
#
# A narrow box at 0.97 is better than
# a narrow box at 0.90. Both matter.
#
# ============================================================

# %%
fig, ax = plt.subplots(figsize=(10, 5))

cv_box_data = {}
for name, model in all_models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_box_data[name] = scores

cv_box_df = pd.DataFrame(cv_box_data)

cv_box_df.boxplot(ax=ax, grid=False,
                  boxprops=dict(color='steelblue', linewidth=2),
                  medianprops=dict(color='red', linewidth=2.5),
                  whiskerprops=dict(color='steelblue'),
                  capprops=dict(color='steelblue'),
                  flierprops=dict(marker='o', color='orange'))

ax.set_ylabel('Accuracy (10-Fold CV)', fontsize=12)
ax.set_title('Cross-Validation Score Distribution - All Models', fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_models_cv_boxplot.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 7.5 Feature Importance Comparison
# ============================================================
#
# Each model has its own view of feature importance.
#
# Look at the top features across models:
# -> Do they agree on the most important features?
# -> Or do they disagree?
#
# If all 4 models agree on the top 3-4 features,
# those features are very likely truly important.
#
# If models disagree a lot, it could mean:
# -> Features are correlated (model picks any one of them)
# -> Different models learned different patterns
#
# Decision Tree shows importance of the single split path.
# Random Forest averages over many trees.
# GBM/XGBoost track improvement over boosting rounds.
#
# This comparison gives you more confidence
# in feature selection than any single model alone.
#
# ============================================================

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax, (name, model) in zip(axes.flatten(), all_models.items()):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': cancer.feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)

    ax.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1],
            color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{name} - Top 10 Features')
    ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('Feature Importance Comparison Across Models',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('all_models_feature_importance.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 7.6 Confusion Matrices - All Models
# ============================================================
#
# Confusion matrix breaks down predictions into 4 categories:
#
# True Positive (TP):   predicted benign, actually benign
# True Negative (TN):   predicted malignant, actually malignant
# False Positive (FP):  predicted benign, actually malignant
# False Negative (FN):  predicted malignant, actually benign
#
# For cancer detection:
#
# False Negative is the dangerous error.
# -> We predicted "benign" but it was actually "malignant"
# -> Patient misses treatment
#
# False Positive is less dangerous.
# -> We predicted "malignant" but it was benign
# -> Patient gets unnecessary follow-up
#
# So in medical contexts, we prefer models with
# fewer False Negatives even at the cost of more False Positives.
#
# Check which model has the fewest cells in the off-diagonal
# (specifically the bottom-left cell = False Negatives).
#
# ============================================================

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, model) in zip(axes.flatten(), all_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=cancer.target_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name}\nTest Acc: {accuracy_score(y_test, y_pred):.4f}')

plt.suptitle('Confusion Matrices - All Models', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('all_models_confusion_matrices.png', dpi=150)
plt.show()

# %%
