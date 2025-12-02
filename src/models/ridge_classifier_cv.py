import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn import metrics

DATA_DIR = "../data"

matches = pd.read_csv(f"{DATA_DIR}/matches-v1.1.csv")
matches["Result"] = matches["Result"].map({"win": 2, "draw": 1, "lose": 0})

X = matches.drop(columns=["HomeGoals", "AwayGoals", "TotalGoals", "Result"])
X = pd.get_dummies(X, columns=["HomeTeam", "AwayTeam"])
y = matches["Result"]
feature_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
# X_train = matches.drop(columns=["HomeGoals", "AwayGoals", "TotalGoals", "Result"])
# X_test = matches.drop(columns=["HomeGoals", "AwayGoals", "TotalGoals", "Result"])

ridge_model = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_model.fit(X_train, y_train)

y_predict = ridge_model.predict(X_test)
print(f"Ridge model accuracy: {metrics.accuracy_score(y_test, y_predict) * 100:.2f}%")

def predict_match_with_ridge_classifier(home_team, away_team, year):
    new_match = pd.DataFrame({
        "HomeTeam": [home_team],
        "AwayTeam": [away_team],
        "Year": [year]
    })
    new_encoded = pd.get_dummies(new_match, columns=["HomeTeam", "AwayTeam"])
    new_encoded = new_encoded.reindex(columns=feature_columns, fill_value=0)

    result_code = ridge_model.predict(new_encoded)[0]
    labels = {0: "lose", 1: "draw", 2: "win"}
    print(f"\nPredicted result {new_match['HomeTeam'][0]} vs {new_match['AwayTeam'][0]} with RidgeClassifier: {labels[result_code].upper()}")

    return labels[result_code].upper()

# predict_match_with_ridge_classifier("Brazil", "France", 2026)
