''''
Develop a predictive model using a healthcare dataset containing patient data such as age, blood pressure, BMI, and symptoms.
- Use "Linear Regression" to predict continuous values like medical costs.
- Apply "Logistic Regression" to classify whether a patient is at risk of a specific disease (yes/no).
- Implement a "Decision Tree" to classify patient health levels into multiple categories like low, moderate, or high risk.
- Compare the performance of these models and analyze their strengths in different prediction tasks.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, classification_report,
    confusion_matrix
)

# Load the dataset
df = pd.read_csv("healthcare_data.csv")
print("=== Sample Data ===")
print(df.head(), "\n")

# Encode HealthLevel for Decision Tree
level_map = {"Low": 0, "Moderate": 1, "High": 2}
reverse_map = {v: k for k, v in level_map.items()}
df["HealthLevelEncoded"] = df["HealthLevel"].map(level_map)

features = ["Age", "BloodPressure", "BMI", "SymptomScore"]

# ---------------------- Linear Regression -----------------------
X_cost = df[features]         # input
y_cost = df["MedicalCost"]    # output

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cost, y_cost, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(Xc_train, yc_train)
yc_pred = lin_reg.predict(Xc_test)

print("=== Linear Regression (Medical Costs Prediction) ===")
print("Predicted:", yc_pred.round(2))
print("Actual   :", list(yc_test))
print("MSE      :", round(mean_squared_error(yc_test, yc_pred), 2))
print("R² Score :", round(r2_score(yc_test, yc_pred), 2))
print()

# 📊 Visualization: Predicted vs Actual Costs
plt.figure(figsize=(6, 4))
plt.scatter(yc_test, yc_pred, color='blue', label='Predictions')
plt.plot([yc_test.min(), yc_test.max()], [yc_test.min(), yc_test.max()], 'r--', label='Perfect Fit')
plt.xlabel("Actual Medical Cost")
plt.ylabel("Predicted Medical Cost")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------- Logistic Regression ---------------------
X_disease = df[features]
y_disease = df["DiseaseRisk"]

# stratify ensures equal ratio of Yes/No in both train and test.
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_disease, y_disease, test_size=0.2, random_state=42, stratify=y_disease
)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(Xd_train, yd_train)
yd_pred = log_reg.predict(Xd_test)

print("=== Logistic Regression (Disease Risk Classification) ===")
print("Predicted:", yd_pred)
print("Actual   :", list(yd_test))
print("Accuracy :", round(accuracy_score(yd_test, yd_pred), 2))
print(classification_report(yd_test, yd_pred, zero_division=0))
print()

# 📊 Visualization: Confusion Matrix
cm = confusion_matrix(yd_test, yd_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------- Decision Tree ---------------------------
X_health = df[features]
y_health = df["HealthLevelEncoded"] #y is numerical version of health level (0, 1, 2)

Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_health, y_health, test_size=0.3, random_state=42, stratify=y_health
)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(Xh_train, yh_train)
yh_pred = tree.predict(Xh_test)

# Converts prediction numbers back to words (Low, Moderate, High)
decoded_pred = [reverse_map[i] for i in yh_pred]
decoded_actual = [reverse_map[i] for i in yh_test]

print("=== Decision Tree (Health Level Classification) ===")
print("Predicted:", decoded_pred)
print("Actual   :", decoded_actual)
print("Accuracy :", round(accuracy_score(yh_test, yh_pred), 2))
print(classification_report(
    yh_test, yh_pred,
    labels=[0, 1, 2],
    target_names=["Low", "Moderate", "High"],
    zero_division=0
))

# 📊 Visualization: Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(tree, feature_names=features, class_names=["Low", "Moderate", "High"],
          filled=True, rounded=True)
plt.title("Decision Tree for Health Risk Level")
plt.show()

# 📊 Visualization: Health Level Prediction Confusion Matrix
cm = confusion_matrix(yh_test, yh_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Low", "Moderate", "High"],
            yticklabels=["Low", "Moderate", "High"])
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
