import pandas as pd
from bs4 import BeautifulSoup
import requests

DATA_DIR = '../data'

years = [
    1930, 1934, 1938, 1950, 1954, 
    1958, 1962, 1966, 1970, 1974, 
    1978, 1982, 1986, 1990, 1994, 
    1998, 2002, 2006, 2010, 2014, 
    2018, 2022]

def get_matches(year):
    web_page = f'https://en.wikipedia.org/wiki/{year}_FIFA_World_Cup'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 "
        "Safari/537.36"
    }
    response = requests.get(web_page, headers=headers)
    content = response.text
    # print(content)
    soup = BeautifulSoup(content, 'lxml')

    matches = soup.find_all('div', class_='footballbox')

    home = []
    score = []
    away = []

    for match in matches:
        home.append(match.find('th', class_='fhome').get_text().strip())
        score.append(match.find('th', class_='fscore').get_text().strip())
        away.append(match.find('th', class_='faway').get_text().strip())

    if year == 1990:
        tables = soup.find_all('table', style='width:100%')
        for table in tables:
            rows = table.find_all('tr', style='font-size:90%')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 3:
                    continue
                home_td = cols[0]
                score_td = cols[1]
                away_td = cols[2]

                home_team_tag = home_td.find('a')
                away_team_tag = away_td.find('a')
                score_val = score_td.get_text().strip()

                if home_team_tag and away_team_tag:
                    home.append(home_team_tag.get_text().strip())
                    score.append(score_val)
                    away.append(away_team_tag.get_text().strip())
            

    df = pd.DataFrame({
        'home': home,
        'score': score,
        'away': away
    })
    df['year'] = year
    return df

matches_list = [get_matches(year) for year in years]
for i in range(0, len(years)):
    print(years[i], int(matches_list[i].size / 4))
df_matches = pd.concat(matches_list, ignore_index=True)
print(df_matches)
df_matches.to_csv(f'{DATA_DIR}/matches-v1.0.csv')

print(get_matches(1990))

# PREDICT FOR 2026
df_predict = get_matches(2026)
print(df_predict)
df_predict.to_csv(f'{DATA_DIR}/matches-predict.csv')
