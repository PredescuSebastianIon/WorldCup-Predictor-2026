import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("../../data/processed/all_matches_relevant_teams.csv")
df = pd.read_csv("../data/processed/all_matches_relevant_teams.csv")

def result(row):
    if row["home_score"] > row["away_score"]:
        return 1
    elif row["home_score"] < row["away_score"]:
        return -1
    else:
        return 0

df["match_result"] = df.apply(result, axis = 1)
df["year"] = pd.to_datetime(df["date"]).dt.year

team_cols = ["home_team", "away_team"]
label_encoders = {}
# List all teams that could possibly be encountered for home or away teams,
# including those that might not appear in all_matches_relevant_teams.csv.
# This list is based on your traceback error showing 'Korea Republic' missing.
all_possible_teams = list(df["home_team"].unique()) + list(df["away_team"].unique())
# Add 'Korea Republic' and ensure uniqueness, just in case:
all_possible_teams = list(set(all_possible_teams + ['Korea Republic', 'South Korea',
                                                    'Serbia', 'Côte d\'Ivoire',
                                                    'Cabo Verde', 'Chile', 'Iceland'])) # Add any other missing teams here!

for col in team_cols:
    le = LabelEncoder()
    # Fit the LabelEncoder on the full, comprehensive list of teams.
    le.fit(all_possible_teams)

    # Transform the DataFrame columns using the newly fitted encoder.
    df[col] = le.transform(df[col])
    label_encoders[col] = le
# for col in team_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     label_encoders[col] = le

features = [
    "home_team", "away_team",
    "home_rank", "away_rank",
    "home_points", "away_points",
    "rank_diff", "points_diff",
    "same_confed", "neutral",
    "year"
]

X = df[features]
y = df["match_result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

forest_model.fit(X_train, y_train)

y_pred = forest_model.predict(X_test)

print(f"Random Forest model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
# print(classification_report(y_test, y_pred))

# importances = pd.Series(forest_model.feature_importances_, index=features)
# print(importances.sort_values(ascending=False))

def predict_match_with_random_forest(home_team, away_team, year):
    numeric_cols = [
        "home_rank", "away_rank",
        "home_points", "away_points",
        "rank_diff", "points_diff",
        "same_confed", "neutral"
    ]
    home_stats = df[df["home_team"] == label_encoders["home_team"].transform([home_team])[0]][numeric_cols].mean()
    away_stats = df[df["away_team"] == label_encoders["away_team"].transform([away_team])[0]][numeric_cols].mean()

    new_match = pd.DataFrame({
        "home_team": [label_encoders["home_team"].transform([home_team])[0]],
        "away_team": [label_encoders["away_team"].transform([away_team])[0]],
        "home_rank": [home_stats["home_rank"]],
        "away_rank": [away_stats["away_rank"]],
        "home_points": [home_stats["home_points"]],
        "away_points": [away_stats["away_points"]],
        "rank_diff": [home_stats["rank_diff"]],
        "points_diff": [home_stats["points_diff"]],
        "same_confed": [1 if home_stats["same_confed"] > 0.5 else 0],
        "neutral": [0],
        "year": [year]
    })

    result_code = forest_model.predict(new_match)[0]
    probabilities = forest_model.predict_proba(new_match)[0]

    # labels = {-1: "AWAY WIN", 0: "DRAW", 1: "HOME WIN"}
    labels = {-1: "LOSE", 0: "DRAW", 1: "WIN"}
    print(f"\nPrediction {home_team} vs {away_team} ({year}): {labels[result_code]}")
    print("Probabilities:", probabilities)
    return labels[result_code], probabilities

# OLD DATASET
# teams = pd.read_csv("../data/processed/merged_data.csv")
# matches = pd.read_csv("../data/matches-v1.1.csv")

# teams['year'] = pd.to_datetime(teams['date']).dt.year

# matches = matches.merge(
#     teams[['country', 'year', 'rank', 'points', 'confed']],
#     left_on=['HomeTeam', 'Year'],
#     right_on=['country', 'year'],
#     how='left'
# ).rename(columns={
#     'rank': 'home_rank',
#     'points': 'home_points',
#     'confed': 'home_confed'
# }).drop(columns=['country', 'year'])

# matches = matches.merge(
#     teams[['country', 'year', 'rank', 'points', 'confed']],
#     left_on=['AwayTeam', 'Year'],
#     right_on=['country', 'year'],
#     how='left'
# ).rename(columns={
#     'rank': 'away_rank',
#     'points': 'away_points',
#     'confed': 'away_confed'
# }).drop(columns=['country', 'year'])

# def match_outcome(row):
#     if row['HomeGoals'] > row['AwayGoals']:
#         return 1    # home win
#     elif row['HomeGoals'] < row['AwayGoals']:
#         return -1   # away win
#     else:
#         return 0    # draw

# matches['result'] = matches.apply(match_outcome, axis=1)

# matches['rank_diff'] = matches['away_rank'] - matches['home_rank']
# matches['points_diff'] = matches['home_points'] - matches['away_points']
# matches['home_advantage'] = 1

# feature_cols = [
#     'home_rank', 'away_rank',
#     'home_points', 'away_points',
#     'rank_diff', 'points_diff',
#     'home_advantage'
# ]

# X = matches[feature_cols]
# y = matches['result']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# forest_model = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=None,
#     min_samples_split=2,
#     random_state=42,
#     n_jobs=-1
# )

# forest_model.fit(X_train, y_train)

# y_pred = forest_model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# importances = pd.Series(forest_model.feature_importances_, index=feature_cols)
# print(importances.sort_values(ascending=False))
