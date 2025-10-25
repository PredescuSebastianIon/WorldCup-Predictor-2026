import sqlite3
import os
import pandas as pd

DATA_BASE_FILE = "../data/winner_predictions.db"
# implement later with os for docker

def update_procentage():
    winner_predictions_db = sqlite3.connect(DATA_BASE_FILE)
    winner_predictions_cursor = winner_predictions_db.cursor()
    fetch_command = fetch_command = """
        SELECT 
            team, 
            COUNT(team) as cnt, 
            (COUNT(team) * 1.0 / SUM(COUNT(team)) OVER ())
        FROM winner
        GROUP BY team
        """
    winner_predictions_cursor.execute(fetch_command)
    data = winner_predictions_cursor.fetchall()
    panda_data = pd.DataFrame(data, columns=['team', 'cnt', 'percentage'])
    return panda_data
    print(data)

def select_winner(user, team):
    winner_predictions_db = sqlite3.connect(DATA_BASE_FILE)
    winner_predictions_cursor = winner_predictions_db.cursor()
    insert_command = f"""INSERT INTO winner (user, team)
    VALUES ('{user}', '{team}')
    """
    winner_predictions_cursor.execute(insert_command)
    winner_predictions_db.commit()
    return update_procentage()

def setup_db():
    winner_predictions_db = sqlite3.connect(DATA_BASE_FILE)
    winner_predictions_cursor = winner_predictions_db.cursor()

    # create table
    create_table_command = """
    CREATE TABLE IF NOT EXISTS
    winner(user VARCHAR(50), team VARCHAR(50));
    """
    winner_predictions_cursor.execute(create_table_command)

setup_db()
