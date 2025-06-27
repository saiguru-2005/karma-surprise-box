import json
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# Update the input file path to your new dataset
INPUT_JSON_FILE = os.getenv("INPUT_JSON_FILE", "training_data.json")
MODEL_OUTPUT_FILENAME = "classifier_new.pkl"
FEATURES_OUTPUT_FILENAME = "feature_names_new.json"
TEST_SET_SIZE = 0.25
RANDOM_STATE = 42

# Step 1: Load the dataset
print(f" Loading data from: {INPUT_JSON_FILE}")
with open(INPUT_JSON_FILE, 'r') as f:
    all_data_points = json.load(f)
print(f" Loaded {len(all_data_points)} data points.")

# Step 2: Extract features and labels
EXPECTED_MODEL_FEATURE_KEYS = [
    "login_streak", "posts_created", "comments_written",
    "upvotes_received", "quizzes_completed", "buddies_messaged",
    "karma_spent", "karma_earned_today"
]
X_all = pd.DataFrame([{key: dp['features'].get(key, 0) for key in EXPECTED_MODEL_FEATURE_KEYS} for dp in all_data_points])
# Convert boolean 'surprise_unlocked' to integer (true -> 1, false -> 0)
y_all = np.array([1 if dp['surprise_unlocked'] else 0 for dp in all_data_points])

print(f"\n X shape: {X_all.shape}, y shape: {y_all.shape}")

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y_all
)
print(f"\n Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
print("\n Training label distribution:")
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"Label {label}: {count} samples ({count/len(y_train)*100:.2f}%)")

# Step 4: Compute Sample Weights
sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)
print(f"\n Sample weights computed for {len(sample_weights_train)} training samples.")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# Step 4.5: Manual K-Fold Cross-validation to print classification reports per fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\nClassification Report for Each Fold:")
fold_idx = 1
for train_idx, val_idx in kfold.split(X_train, y_train):
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Compute sample weights for this fold
    fold_weights = compute_sample_weight(class_weight='balanced', y=y_fold_train)

    # Train model on fold
    fold_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **{
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'criterion': 'gini'
    })
    fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)

    # Predict and report
    y_fold_pred = fold_model.predict(X_fold_val)
    print(f"\nFold {fold_idx} Classification Report:")
    print(classification_report(y_fold_val, y_fold_pred, target_names=["No Reward (0)", "Reward (1)"]))
    fold_idx += 1

# Step 5: Grid Search Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\n GridSearchCV will evaluate {total_combinations} hyperparameter combinations...")

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("\n Training model with GridSearchCV...")
grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)
best_rf = grid_search.best_estimator_
print(f"\n Best Hyperparameters: {grid_search.best_params_}")

# Step 6: Evaluate the Model
print("\n Evaluating on test set...")
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"\n Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Reward (0)", "Reward (1)"]))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\n ROC AUC Score: {roc_auc:.4f}")

# Step 7: Feature Importances
print("\nTop 10 Feature Importances:")
importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)
print(importances.head(10).to_string(index=False))

# Step 8: Save Artifacts
print(f"\nSaving model to '{MODEL_OUTPUT_FILENAME}'...")
joblib.dump(best_rf, MODEL_OUTPUT_FILENAME)
print(f"Saving feature names to '{FEATURES_OUTPUT_FILENAME}'...")
with open(FEATURES_OUTPUT_FILENAME, 'w') as f:
    json.dump(EXPECTED_MODEL_FEATURE_KEYS, f)
print(f"\nModel and features successfully saved.")