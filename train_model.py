import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import numpy as np

# Load your dataset
data = pd.read_csv('Data.csv')  # Update this with your dataset path

# Ensure that 'fluid_overload.1' is included as a feature
if 'fluid_overload.1' not in data.columns:
    data['fluid_overload.1'] = 0  # Default value for missing feature

# Define your features and target variable
X = data.drop(columns=['prognosis'])  # All columns except 'prognosis' are features
y = data['prognosis']  # The target variable

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier without hyperparameter tuning
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)

# Calculate and print metrics
print("Random Forest (Normal):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=0))

# Hyperparameter tuning for Random Forest using RandomizedSearchCV
param_distributions = {
    'n_estimators': np.arange(50, 500, 50),  # Number of trees
    'max_depth': [None] + list(np.arange(5, 20, 5)),  # Maximum depth of the trees
    'min_samples_split': np.arange(2, 10, 2),  # Minimum number of samples to split a node
    'min_samples_leaf': np.arange(1, 5),  # Minimum number of samples at a leaf node
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                      param_distributions, n_iter=10, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)

# Best model from random search
best_model_rf = random_search_rf.best_estimator_
y_pred_rf = best_model_rf.predict(X_test)

# Calculate and print metrics for the best model
print("\nRandom Forest (With Randomized Hyperparameter Tuning):")
print("Best Parameters:", random_search_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted', zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted', zero_division=0))

# Optionally save the best model
joblib.dump(best_model_rf, 'model.pkl')