from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train classifiers with different random_state values
clf1 = RandomForestClassifier(criterion='log_loss', random_state=42)
clf2 = RandomForestClassifier(criterion='gini', random_state=24)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)

# Predict and evaluate
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)

print(f"Accuracy with random_state=42: {accuracy1:.2f}")
print(f"Accuracy with random_state=24: {accuracy2:.2f}")
