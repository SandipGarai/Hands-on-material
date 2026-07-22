# %% [markdown]
# ============================================================
# Guided Hands-On: Predict Penguin Species with GBM & XGBoost
# ============================================================
#
# Goal:
#   Build a model that predicts the species of a penguin
#   (Adelie, Chinstrap, or Gentoo) from simple body measurements.
#
# Why penguins?
#   - The data is small and easy to understand.
#   - It has only 4 numeric features: bill length, bill depth,
#     flipper length, and body mass.
#   - It has some missing values, so we can show how XGBoost
#     handles them natively.
#
# What we cover:
#   1. Load and explore the data
#   2. Save / download the data as a CSV file
#   3. Train a Decision Tree and Random Forest (quick review)
#   4. Train a Gradient Boosting model (sklearn)
#   5. Train an XGBoost model with early stopping
#   6. Handle missing values in both GBM and XGBoost
#   7. Compare all models
#   8. Create a sample CSV file for practice
#   9. A copy-paste template for YOUR own CSV file
#
# By the end, you should feel confident replacing the penguin data
# with your own dataset and running the same steps.
#
# ============================================================


# %% [markdown]
# ============================================================
# 1. Setup
# ============================================================
#
# We use the same tools from Session 2, plus seaborn to load
# the penguins dataset.
#
# If any package is missing, run in your terminal:
#   pip install numpy pandas matplotlib seaborn scikit-learn xgboost
#
# ============================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)


# %% [markdown]
# ============================================================
# 2. Load and Explore the Data
# ============================================================
#
# We use seaborn's built-in "penguins" dataset.
# It contains measurements for 3 penguin species.
#
# The columns we care about:
#   bill_length_mm     -> length of the bill;
#   bill_depth_mm      -> depth of the bill;
#   flipper_length_mm  -> length of the flipper;
#   body_mass_g        -> body mass in grams;
#   species            -> Adelie, Chinstrap, Gentoo (our target)
#
# ============================================================

# %%
# Load the data
df = sns.load_dataset('penguins')

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nSpecies counts:")
print(df['species'].value_counts())


# %% [markdown]
# ============================================================
# 2.1 Save the Data to a CSV File
# ============================================================
#
# In a real project, you usually download or receive a CSV file.
# Here we save the seaborn penguins data to 'penguins.csv' so you
# can practice loading it yourself.
#
# This is also the file you would share with students or upload
# to a notebook environment (Google Colab, Jupyter, etc.).
#
# ============================================================

# %%
# Save the full raw data (including missing values) to a CSV file
df.to_csv('penguins.csv', index=False)
print("Saved penguins.csv")
print(f"File shape: {df.shape}")

# You can reload it anytime with:
# df = pd.read_csv('penguins.csv')


# %% [markdown]
# ============================================================
# 2.2 Why do we drop missing values here?
# ============================================================
#
# The penguin dataset has a few missing measurements.
# For this first hands-on, we drop rows with missing values
# to keep the workflow simple.
#
# IMPORTANT: In real projects, think before dropping.
# You can also:
#   - Fill missing values with the mean/median
#   - Use a model that handles missing values (XGBoost does this!)
#
# We will come back to missing values in Section 6.
#
# ============================================================

# %%
# Drop rows with missing values and select numeric features + target
df_clean = df.dropna().copy()

feature_cols = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
X = df_clean[feature_cols]
y = df_clean['species']

print(
    f"\nAfter dropping missing values: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Features: {list(feature_cols)}")
print(f"Target classes: {list(y.unique())}")


# %% [markdown]
# ============================================================
# 2.3 Train / Test Split
# ============================================================
#
# We split the data into training (80%) and test (20%) sets.
# stratify=y keeps the same proportion of each species in both sets.
#
# ============================================================

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# XGBoost needs numeric labels, so we encode species names
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
class_names = le.classes_

print(f"Train set: {X_train.shape}")
print(f"Test set:  {X_test.shape}")
print(f"\nTrain species distribution:\n{y_train.value_counts()}")
print(f"\nTest species distribution:\n{y_test.value_counts()}")
print(f"\nEncoded classes: {list(class_names)}")


# %% [markdown]
# ============================================================
# 2.4 Quick Visualization
# ============================================================
#
# A pairplot shows how the three species separate based on the
# four measurements. Notice that some pairs of species are very
# easy to separate, while others overlap.
#
# This is why tree-based models work well: they cut the feature
# space into simple rectangles.
#
# ============================================================

# %%
plt.figure(figsize=(8, 6))
sns.pairplot(df_clean, hue='species',
             vars=feature_cols, corner=True, height=1.8)
plt.suptitle('Penguin Measurements by Species', y=1.02, fontsize=13)
plt.savefig('penguins_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ============================================================
# 3. Baseline Models
# ============================================================
#
# Before boosting, let's quickly train two models from Session 1:
#   - Decision Tree (single tree, easy to interpret)
#   - Random Forest (bagging many trees)
#
# These give us a reference point. Boosting should beat or match
# them, especially on the test set.
#
# ============================================================

# %%
# Decision Tree baseline
dt = DecisionTreeClassifier(max_depth=4, random_state=SEED)
dt.fit(X_train, y_train_enc)

# Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train_enc)

print("=== Decision Tree ===")
print(f"Train Acc: {dt.score(X_train, y_train_enc):.4f}")
print(f"Test Acc:  {dt.score(X_test, y_test_enc):.4f}")

print("\n=== Random Forest ===")
print(f"Train Acc: {rf.score(X_train, y_train_enc):.4f}")
print(f"Test Acc:  {rf.score(X_test, y_test_enc):.4f}")


# %% [markdown]
# ============================================================
# 4. Gradient Boosting (sklearn)
# ============================================================
#
# We use GradientBoostingClassifier from scikit-learn.
# Remember the key idea from the lecture:
#
#   Trees are added one by one. Each new tree tries to correct
#   the mistakes of the existing ensemble.
#
# Key parameters:
#   n_estimators  -> number of trees (boosting rounds);
#   learning_rate -> how much each new tree contributes;
#   max_depth     -> depth of each tree (keep it small!);
#   subsample     -> fraction of rows used per tree (adds randomness)
#
# ============================================================

# %%
gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=SEED
)

gbm.fit(X_train, y_train_enc)

y_pred_gbm = gbm.predict(X_test)

print("=== Gradient Boosting ===")
print(f"Train Acc: {gbm.score(X_train, y_train_enc):.4f}")
print(f"Test Acc:  {gbm.score(X_test, y_test_enc):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred_gbm, target_names=class_names))


# %% [markdown]
# ============================================================
# 4.1 Learning Rate vs Number of Trees
# ============================================================
#
# This is the classic GBM tradeoff.
#
# Small learning_rate -> each tree contributes less -> need more trees.
# Large learning_rate -> each tree contributes more -> can overfit faster.
#
# Try the combinations below and watch the Test Acc.
#
# ============================================================

# %%
configs = [
    {'lr': 0.3,  'n': 50,  'label': 'lr=0.3, n=50'},
    {'lr': 0.1,  'n': 100, 'label': 'lr=0.1, n=100'},
    {'lr': 0.05, 'n': 200, 'label': 'lr=0.05, n=200'},
    {'lr': 0.01, 'n': 300, 'label': 'lr=0.01, n=300'},
]

print("\n=== Learning Rate vs n_estimators ===")
print(f"{'Config':<25} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 47)

for config in configs:
    model = GradientBoostingClassifier(
        n_estimators=config['n'],
        learning_rate=config['lr'],
        max_depth=3,
        random_state=SEED
    )
    model.fit(X_train, y_train_enc)
    print(f"{config['label']:<25} {model.score(X_train, y_train_enc):>10.4f} "
          f"{model.score(X_test, y_test_enc):>10.4f}")


# %% [markdown]
# ============================================================
# 4.2 GBM Feature Importance
# ============================================================
#
# Which measurement helped the model most?
# Feature importance tells us how much each feature reduced
# the loss across all trees.
#
# ============================================================

# %%
gbm_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gbm.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=gbm_importance, x='Importance',
            y='Feature', palette='viridis')
plt.title('GBM Feature Importance - Penguin Species')
plt.tight_layout()
plt.savefig('penguins_gbm_feature_importance.png', dpi=150)
plt.show()

print("\nFeature ranking:")
print(gbm_importance)


# %% [markdown]
# ============================================================
# 5. XGBoost
# ============================================================
#
# XGBoost does the same boosting idea but adds:
#   - Regularization (L1/L2 on leaf weights)
#   - Faster training (parallelized)
#   - Early stopping
#   - Native missing value handling
#
# For penguins (3 species), we use eval_metric='mlogloss'.
#
# ============================================================

# %%
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_lambda=1,
    reg_alpha=0,
    eval_metric='mlogloss',
    random_state=SEED,
    n_jobs=2
)

xgb_model.fit(X_train, y_train_enc)

y_pred_xgb = xgb_model.predict(X_test)

print("=== XGBoost ===")
print(f"Train Acc: {xgb_model.score(X_train, y_train_enc):.4f}")
print(f"Test Acc:  {xgb_model.score(X_test, y_test_enc):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred_xgb, target_names=class_names))


# %% [markdown]
# ============================================================
# 5.1 XGBoost with Early Stopping
# ============================================================
#
# Instead of guessing the right number of trees, let XGBoost decide.
#
# We set n_estimators very high (1000) and tell it:
#   "Stop if the validation loss doesn't improve for 30 rounds."
#
# This usually gives a better model and trains faster than
# manually tuning n_estimators.
#
# ============================================================

# %%
xgb_early = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_lambda=2,
    eval_metric='mlogloss',
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=2
)

xgb_early.fit(
    X_train, y_train_enc,
    eval_set=[(X_train, y_train_enc), (X_test, y_test_enc)],
    verbose=False
)

print("=== XGBoost with Early Stopping ===")
print(f"Best iteration:  {xgb_early.best_iteration}")
print(f"Best eval score: {xgb_early.best_score:.4f}")
print(f"Train Acc:       {xgb_early.score(X_train, y_train_enc):.4f}")
print(f"Test Acc:        {xgb_early.score(X_test, y_test_enc):.4f}")


# %% [markdown]
# ============================================================
# 5.2 Plot Training History
# ============================================================
#
# With early stopping, XGBoost saves the loss on train and test
# for every round. This plot is the best way to see overfitting.
#
# If the red line (validation) starts going up while the blue
# line (train) keeps going down, the model is overfitting.
# The green dashed line shows where early stopping kicked in.
#
# ============================================================

# %%
results = xgb_early.evals_result()

train_loss = results['validation_0']['mlogloss']
val_loss = results['validation_1']['mlogloss']

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss', color='steelblue', lw=2)
plt.plot(val_loss, label='Validation Loss', color='salmon', lw=2)
plt.axvline(x=xgb_early.best_iteration, linestyle='--', color='green',
            label=f'Best Iteration: {xgb_early.best_iteration}')
plt.xlabel('Boosting Iterations')
plt.ylabel('Multiclass Log Loss')
plt.title('XGBoost: Training vs Validation Loss (Early Stopping)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('penguins_xgb_loss_curves.png', dpi=150)
plt.show()


# %% [markdown]
# ============================================================
# 5.3 XGBoost Feature Importance
# ============================================================
#
# XGBoost gives us multiple ways to measure importance.
# We use 'gain' because it tells us which feature actually
# improves the model the most when it is used.
#
# ============================================================

# %%
imp_dict = xgb_model.get_booster().get_score(importance_type='gain')
imp_df = pd.DataFrame(list(imp_dict.items()),
                      columns=['Feature', 'Gain']).sort_values('Gain', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=imp_df, x='Gain', y='Feature', palette='magma')
plt.title('XGBoost Feature Importance (Gain) - Penguin Species')
plt.tight_layout()
plt.savefig('penguins_xgb_feature_importance.png', dpi=150)
plt.show()

print("\nFeature ranking by gain:")
print(imp_df)


# %% [markdown]
# ============================================================
# 5.4 Handling Missing Values: GBM vs XGBoost
# ============================================================
#
# sklearn's GradientBoostingClassifier CANNOT handle missing values.
# If we give it data with NaNs, it will throw an error.
#
# Solution for GBM: fill (impute) missing values.
# Here we use the column mean.
#
# XGBoost, on the other hand, handles missing values natively.
# It learns the best direction to send missing values during training.
#
# Let's compare both approaches on the original data with missing values.
#
# ============================================================

# %%
# Use the original data WITH missing values
X_full = df[feature_cols]
y_full = df['species']

X_train_miss, X_test_miss, y_train_miss, y_test_miss = train_test_split(
    X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full
)

# Encode labels for the missing-value version
y_train_miss_enc = le.fit_transform(y_train_miss)
y_test_miss_enc = le.transform(y_test_miss)

# --- GBM with imputation ---
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train_miss)
X_test_imp = imputer.transform(X_test_miss)

gbm_missing = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=SEED
)
gbm_missing.fit(X_train_imp, y_train_miss_enc)

print("=== GBM with Missing Values (mean imputation) ===")
print(f"Train Acc: {gbm_missing.score(X_train_imp, y_train_miss_enc):.4f}")
print(f"Test Acc:  {gbm_missing.score(X_test_imp, y_test_miss_enc):.4f}")
print(f"Missing values filled: {X_train_miss.isnull().sum().sum()}")

# --- XGBoost without imputation ---
xgb_missing = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='mlogloss',
    random_state=SEED,
    n_jobs=2
)

xgb_missing.fit(X_train_miss, y_train_miss_enc)

print("\n=== XGBoost with Missing Values (no imputation) ===")
print(f"Train Acc: {xgb_missing.score(X_train_miss, y_train_miss_enc):.4f}")
print(f"Test Acc:  {xgb_missing.score(X_test_miss, y_test_miss_enc):.4f}")
print(f"Missing values left as NaN: {X_train_miss.isnull().sum().sum()}")


# %% [markdown]
# ============================================================
# 6. Compare All Models
# ============================================================
#
# Now we put everything side by side:
#   Decision Tree, Random Forest, GBM, XGBoost.
#
# We use cross-validation (5-fold) to get a more reliable score
# than a single train-test split.
#
# ============================================================

# %%
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    'GBM': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=SEED),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.1,
                                 max_depth=3, eval_metric='mlogloss',
                                 random_state=SEED, n_jobs=2)
}

print("\n=== Model Comparison (5-Fold Cross-Validation) ===")
print(f"{'Model':<18} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 52)

for name, model in models.items():
    model.fit(X_train, y_train_enc)
    test_acc = model.score(X_test, y_test_enc)
    cv_scores = cross_val_score(
        model, X_train, y_train_enc, cv=5, scoring='accuracy')
    print(f"{name:<18} {test_acc:>10.4f} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f}")


# %% [markdown]
# ============================================================
# 6.1 Confusion Matrix for the Best Model
# ============================================================
#
# The confusion matrix shows WHICH species get confused with each other.
# This is often more useful than a single accuracy number.
#
# ============================================================

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{name}\nTest Acc: {accuracy_score(y_test_enc, y_pred):.4f}')

plt.suptitle('Confusion Matrices - All Models', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('penguins_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ============================================================
# 7. Create a Sample CSV File for Practice
# ============================================================
#
# Before you try your own data, you can practice the template
# on this small sample file: 'sample_penguin_data.csv'.
#
# It contains 20 real penguin rows with the same columns.
# After running this script, try loading it with the template
# in the next section.
#
# ============================================================

# %%
# Create a small sample file from the cleaned data
sample_df = df_clean.sample(n=20, random_state=SEED)[
    feature_cols + ['species']]
sample_df.to_csv('sample_penguin_data.csv', index=False)
print("Created sample_penguin_data.csv")
print(f"Sample shape: {sample_df.shape}")
print(sample_df.head())


# %% [markdown]
# ============================================================
# 8. Template: Use This with Your Own Data
# ============================================================
#
# You can replace the penguin data with your own CSV file.
# The steps stay exactly the same.
#
# How to upload your file:
#   - In Jupyter/VS Code: put the CSV in the same folder as this script.
#   - In Google Colab: run the cell below and upload via the file picker.
#
# A sample file 'sample_penguin_data.csv' is also provided.
# Try loading it with the template below.
#
# ============================================================

# %%
# TEMPLATE:

# # If using Google Colab, uncomment the next two lines to upload:
# # from google.colab import files
# # uploaded = files.upload()   # then use the uploaded filename

# # 1. Load your data (replace 'your_data.csv' with your filename)
# df = pd.read_csv('your_data.csv')
# # df = pd.read_csv('sample_penguin_data.csv')  # sample file
#
# # 2. Choose features and target
# feature_cols = ['feature_1', 'feature_2', 'feature_3']
# target_col = 'target'
#
# X = df[feature_cols]
# y = df[target_col]
#
# # 3. Optional: handle categorical features
# # X = pd.get_dummies(X, drop_first=True)
#
# # 4. Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#
# # 5. Encode target labels (XGBoost needs numbers, not strings)
# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train)
# y_test_enc = le.transform(y_test)
#
# # 6. Train XGBoost with early stopping
# model = xgb.XGBClassifier(
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=3,
#     eval_metric='mlogloss',  # or 'logloss' for binary classification
#     early_stopping_rounds=30,
#     random_state=42,
#     n_jobs=2
# )
#
# model.fit(
#     X_train, y_train_enc,
#     eval_set=[(X_train, y_train_enc), (X_test, y_test_enc)],
#     verbose=False
# )
#
# print(f"Best iteration: {model.best_iteration}")
# print(f"Test accuracy:  {model.score(X_test, y_test_enc):.4f}")
# print(f"Classes:        {le.classes_}")


# %% [markdown]
# ============================================================
# 9. Student Checklist
# ============================================================
#
# After this hands-on, you should be able to:
#
#  [ ] Load a dataset and split it into train/test.
#  [ ] Train a Decision Tree, Random Forest, GBM, and XGBoost model.
#  [ ] Explain what boosting does in one sentence.
#  [ ] Tune learning_rate and n_estimators.
#  [ ] Use early stopping in XGBoost.
#  [ ] Read a feature importance plot.
#  [ ] Handle missing values in GBM (impute) and XGBoost (native).
#  [ ] Replace the penguin data with your own CSV.
#
# ============================================================
# %%
# Practice:

# 1. Load your data (replace 'your_data.csv' with your filename)
df = pd.read_csv('data.csv')
# df = pd.read_csv('sample_penguin_data.csv')  # sample file

# 2. Choose features and target
feature_cols = ['feature_1', 'feature_2', 'feature_3']
target_col = 'target'

X = df[feature_cols]
y = df[target_col]

# 3. Optional: handle categorical features
# X = pd.get_dummies(X, drop_first=True)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Encode target labels (XGBoost needs numbers, not strings)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 6. Train XGBoost with early stopping
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    eval_metric='mlogloss',  # or 'logloss' for binary classification
    early_stopping_rounds=30,
    random_state=42,
    n_jobs=2
)

model.fit(
    X_train, y_train_enc,
    eval_set=[(X_train, y_train_enc), (X_test, y_test_enc)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Test accuracy:  {model.score(X_test, y_test_enc):.4f}")
print(f"Classes:        {le.classes_}")
