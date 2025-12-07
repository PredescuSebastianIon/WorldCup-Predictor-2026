import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

DATA_DIR = "../data"

matches = pd.read_csv(f"{DATA_DIR}/matches-v1.1.csv")
matches["Result"] = matches["Result"].map({"win": 2, "draw": 1, "lose": 0})

X = matches.drop(columns=['Result', 'HomeGoals', 'AwayGoals', 'TotalGoals'])
X = pd.get_dummies(X, columns=["HomeTeam", "AwayTeam"])
y = matches['Result']
feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

logistic_model = LogisticRegression(max_iter=30000)
logistic_model.fit(X_train, y_train)

y_predict = logistic_model.predict(X_test)

print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_predict) * 100:.2f}%")

def predict_match_with_logistic_regression(HomeTeam, AwayTeam, Year):
    new_match = pd.DataFrame({
        'HomeTeam': [HomeTeam], 
        'AwayTeam': [AwayTeam], 
        'Year': [Year]
    })
    new_encoded = pd.get_dummies(new_match, columns=["HomeTeam", "AwayTeam"])
    new_encoded = new_encoded.reindex(columns=feature_columns, fill_value=0)

    result_code = logistic_model.predict(new_encoded)[0]
    probabilities = logistic_model.predict_proba(new_encoded)[0]

    labels = {0: "lose", 1: "draw", 2: "win"}
    print(f"\nPredicted result {new_match['HomeTeam'][0]} vs {new_match['AwayTeam'][0]} with LogisticRegression: {labels[result_code].upper()}")
    print(f"Probabilities (lose/draw/win): {np.round(probabilities, 2)}")

    return labels[result_code].upper(), np.round(probabilities, 2)
# , np.round(probabilities, 2)
