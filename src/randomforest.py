import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

teams = pd.read_csv("../data/processed/merged_data.csv")
matches = pd.read_csv("../data/matches-v1.1.csv")

teams['year'] = pd.to_datetime(teams['date']).dt.year

matches = matches.merge(
    teams[['country', 'year', 'rank', 'points', 'confed']],
    left_on=['HomeTeam', 'Year'],
    right_on=['country', 'year'],
    how='left'
).rename(columns={
    'rank': 'home_rank',
    'points': 'home_points',
    'confed': 'home_confed'
}).drop(columns=['country', 'year'])

matches = matches.merge(
    teams[['country', 'year', 'rank', 'points', 'confed']],
    left_on=['AwayTeam', 'Year'],
    right_on=['country', 'year'],
    how='left'
).rename(columns={
    'rank': 'away_rank',
    'points': 'away_points',
    'confed': 'away_confed'
}).drop(columns=['country', 'year'])

def match_outcome(row):
    if row['HomeGoals'] > row['AwayGoals']:
        return 1    # home win
    elif row['HomeGoals'] < row['AwayGoals']:
        return -1   # away win
    else:
        return 0    # draw

matches['result'] = matches.apply(match_outcome, axis=1)

matches['rank_diff'] = matches['away_rank'] - matches['home_rank']
matches['points_diff'] = matches['home_points'] - matches['away_points']
matches['home_advantage'] = 1

feature_cols = [
    'home_rank', 'away_rank',
    'home_points', 'away_points',
    'rank_diff', 'points_diff',
    'home_advantage'
]

X = matches[feature_cols]
y = matches['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False))