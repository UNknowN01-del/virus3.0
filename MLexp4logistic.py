import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load the dataset
import seaborn as sns
titanic = sns.load_dataset('titanic')

# 2. Preprocess
#   - Drop columns we won’t use
#   - Fill or drop missing values
#   - Encode categoricals
titanic = titanic.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])
titanic['age'].fillna(titanic['age'].mean(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
titanic.dropna(subset=['fare'], inplace=True)  # just in case

# Map categories to numbers
titanic.replace({
    'sex': {'male': 0, 'female': 1},
    'embarked': {'S': 0, 'C': 1, 'Q': 2}
}, inplace=True)

# 3. Split features & target
X = titanic.drop(columns=['survived'])
y = titanic['survived']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# 5. Fit logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Inspect coefficients
print("Intercept:", model.intercept_[0])
print("Coefficients:")
for feat, coef in zip(X.columns, model.coef_[0]):
    print(f"  {feat}: {coef:.4f}")

# 7. Predict & evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
