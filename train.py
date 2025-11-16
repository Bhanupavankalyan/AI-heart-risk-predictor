# ===============================
# Heart Disease Model Training Script
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv("heart.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(rf, param_grid=params, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Trained Successfully!")
print(f"🎯 Accuracy: {acc*100:.2f}%")
print("Best Parameters:", grid.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(best_model, "optimized_heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("💾 Model and Scaler saved successfully!")
