import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# New patient prediction
new_patient = np.array([[
    14.5, 20.2, 95.3, 650.1, 0.10,
    0.13, 0.18, 0.09, 0.18, 0.06,
    0.40, 1.20, 2.80, 45.0, 0.006,
    0.03, 0.04, 0.02, 0.1, 0.003,
    16.0, 25.0, 110.0, 850.0, 0.14,
    0.30, 0.40, 0.15, 0.30, 0.08
]])

new_patient_scaled = scaler.transform(new_patient)
prediction = knn.predict(new_patient_scaled)

if prediction[0] == 0:
    print("\nResult: Malignant (Cancer)")
else:
    print("\nResult: Benign (Non-cancer)")
