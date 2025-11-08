import pandas as pd
import os
from scraper import years

DATA_DIR = '../data'

df_matches = pd.read_csv(f'{DATA_DIR}/matches-v1.1.csv')
df_predict = pd.read_csv(f'{DATA_DIR}/matches-predict.csv')

# Clean Data
df_matches.dropna(inplace=True)
df_matches.drop_duplicates(inplace=True)
# Specific w/o match
delete_index = df_matches[df_matches['home'].str.contains('Sweden') 
                   & df_matches['away'].str.contains('Austria')].index
if not delete_index.empty:
    df_matches.drop(index=delete_index, inplace=True)
df_matches['score'] = df_matches['score'].replace(
    {r'[–—−]': '-'}, regex=True
)
df_matches['score'] = df_matches['score'].str.replace(r'[^\d-]', '', regex=True)
# Drop score column and get home and away score
print(df_matches['score'].str.split('-'))
score = df_matches['score'].str.split('-', expand=True)
# print(score[0])
df_matches['HomeGoals'], df_matches['AwayGoals'] = score[0], score[1]
df_matches.drop(columns='score', inplace=True)

df_matches.rename(columns={
    'home': 'HomeTeam',
    'away': 'AwayTeam',
    'year': 'Year'
}, inplace=True)
df_matches = df_matches.astype({
    'HomeGoals': int,
    'AwayGoals': int,
    'Year': int
})
df_matches['TotalGoals'] = df_matches['HomeGoals'] + df_matches['AwayGoals']
print(df_matches.dtypes)

# Predict data
df_predict.dropna(inplace=True)
df_predict.drop_duplicates(inplace=True)
df_predict.rename(columns={
    'home': 'HomeTeam',
    'away': 'AwayTeam',
    'year': 'Year'
}, inplace=True)


# Save dataframes
df_matches.to_csv(f'{DATA_DIR}/matches-v1.1.csv', index=False)
df_predict.to_csv(f'{DATA_DIR}/matches-predict.csv', index=False)

# Verifications
for y in years:
    print(y, len(df_matches[df_matches['Year'] == y]))
