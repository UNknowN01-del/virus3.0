import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load & prepare
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train multiclass logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Evaluate on test set
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Single‑sample prediction without warning
sample_df = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)
pred_class = model.predict(sample_df)[0]
print(f"Input {sample_df.values[0].tolist()} → Predicted class: {iris.target_names[pred_class]}")
