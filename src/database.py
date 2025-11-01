import os
import pandas as pd
import sqlite3

DATA_BASE_FILE = "../data/winner_predictions.db"
# implement later with os for docker

def update_procentage(winner_predictions_cursor):
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

def select_winner(user, team, winner_predictions_cursor, winner_predictions_db):
    insert_command = f"""
        INSERT INTO winner (user, team)
        VALUES ('{user}', '{team}')
    """
    winner_predictions_cursor.execute(insert_command)
    winner_predictions_db.commit()
    return update_procentage(winner_predictions_cursor)

# SETUP DATABASE
def setup_db():
    # create table
    winner_predictions_db = sqlite3.connect(DATA_BASE_FILE, check_same_thread=False)
    winner_predictions_cursor = winner_predictions_db.cursor()
    create_table_command = """
        CREATE TABLE IF NOT EXISTS
        winner(user VARCHAR(50), team VARCHAR(50));
    """
    winner_predictions_cursor.execute(create_table_command)

    return winner_predictions_db, winner_predictions_cursor

winner_predictions_db, winner_predictions_cursor = setup_db()
