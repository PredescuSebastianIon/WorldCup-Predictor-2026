import pandas as pd
import os
import gradio as gr
from countries import countries
from database import select_winner, update_procentage
from database import winner_predictions_db, winner_predictions_cursor
import plotly.express as px
from models import logistic_model, ridge_model, forest_model
from models import predict_match_with_logistic_regression, predict_match_with_ridge_classifier, predict_match_with_random_forest
import random

# --- TOURNAMENT SIMULATION CODE (New Addition) ---

# Synthetic Group Definitions for 48 teams (12 Groups of 4)
# NOTE: The actual draw will follow complex FIFA ranking and geographical rules.
SYNTHETIC_GROUPS = {
    'A': ['Canada', 'Argentina', 'Senegal', 'New Zealand'],
    'B': ['Mexico', 'Netherlands', 'Morocco', 'Panama'],
    'C': ['United States', 'Portugal', 'Poland', 'Ghana'],
    'D': ['Brazil', 'Croatia', 'South Korea', 'Saudi Arabia'],
    'E': ['England', 'Germany', 'Serbia', 'Peru'],
    'F': ['France', 'Uruguay', 'Japan', 'Cameroon'],
    'G': ['Spain', 'Colombia', 'Iran', 'Haiti'],
    'H': ['Belgium', 'Switzerland', 'Australia', 'Romania'],
    'I': ['Italy', 'Denmark', 'Tunisia', 'Ecuador'],
    'J': ['Chile', 'Sweden', 'Nigeria', 'Egypt'],
    'K': ['Ukraine', 'Czech Republic', 'Qatar', 'Venezuela'],
    'L': ['Turkey', 'Norway', 'Finland', 'Honduras'],
}

ACTUAL_GROUPS = {
    'A': ['Mexico', 'South Africa', 'Korea Republic', 'Ukraine'],
    'B': ['Canada', 'Qatar', 'Switzerland', 'Italy'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Romania'],
    'E': ['Germany', 'Curacao', 'Côte d\'Ivoire', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Tunisia', 'Sweden'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cabo Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Norway', 'Chile'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'Uzbekistan', 'Colombia', 'Iceland'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

def predict_match(home_team, away_team, predict_func_state, model_name):

    # Uses the actual prediction function passed from the Gradio state
    if predict_func_state:
        # Predict_func_state returns 'Team A', 'Team B', or 'Draw'
        if model_name == "LogisticRegression" or model_name == "RandomForest":
            state, probabilities = predict_func_state(home_team, away_team, 2026)
        else:
            state = predict_func_state(home_team, away_team, 2026)
            probabilities = None
        if state == "WIN":
            return home_team, probabilities
        elif state == "LOSE":
            return away_team, probabilities
        elif state == "DRAW":
            return state, probabilities
    # Fallback if no model is trained
    return random.choice([home_team, away_team]), None

def run_knockout_match(team1, team2, predict_func_state, model_name):
    # Knockout matches cannot end in a draw.
    while True:
        winner, probabilities = predict_match(team1, team2, predict_func_state, model_name)
        if winner != 'DRAW':
            return winner, probabilities
        # Simulate penalty shootout if a draw is predicted
        return random.choice([team1, team2]), probabilities

def simulate_group_stage(groups, predict_func_state, model_name):
    standings = {group: {team: {'P': 0, 'GD': 0, 'Pts': 0} for team in teams} for group, teams in groups.items()}
    
    for group, teams in groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                team_a, team_b = teams[i], teams[j]
                result = predict_match(team_a, team_b, predict_func_state, model_name)[0]
                
                if result == team_a:
                    standings[group][team_a]['Pts'] += 3
                elif result == team_b:
                    standings[group][team_b]['Pts'] += 3
                else: # Draw
                    standings[group][team_a]['Pts'] += 1
                    standings[group][team_b]['Pts'] += 1
                    
                standings[group][team_a]['P'] += 1
                standings[group][team_b]['P'] += 1
                    
    qualified_teams = []
    third_place_teams = []

    # Get top two from each group (24 teams)
    for group in groups:
        group_standings_list = sorted(
            standings[group].items(),
            key=lambda x: (x[1]['Pts'], x[1]['GD'], random.random()), 
            reverse=True
        )
        qualified_teams.append(group_standings_list[0][0]) 
        qualified_teams.append(group_standings_list[1][0]) 
        third_place_teams.append((group_standings_list[2][0], group_standings_list[2][1]['Pts'], group_standings_list[2][1]['GD'])) 
        
    # Get 8 best third-place teams (8 teams)
    best_thirds = sorted(
        third_place_teams,
        key=lambda x: (x[1], x[2], random.random()), 
        reverse=True
    )[:8]
    
    qualified_teams.extend([team[0] for team in best_thirds])
    random.shuffle(qualified_teams)

    return qualified_teams

def simulate_knockout_stage(qualified_teams, predict_func_state, stage_name, model_name):
    winners = []
    stage_output = f"### ➡️ {stage_name} (Total Matches: {len(qualified_teams)//2})\n"
    
    for i in range(0, len(qualified_teams), 2):
        team1 = qualified_teams[i]
        team2 = qualified_teams[i+1]

        winner, probabilities = run_knockout_match(team1, team2, predict_func_state, model_name)
        winners.append(winner)
        stage_output += f"* **{team1}** vs **{team2}** &rArr; Winner: **{winner}**"
        if probabilities is not None:
            stage_output += f" with the probabilities of lose-{probabilities[0]*100:.2f}%, \
                                                        draw-{probabilities[1]*100:.2f}%, \
                                                        win-{probabilities[2]*100:.2f}%\n"
        else:
            stage_output += f"\n"

    return winners, stage_output

def run_full_tournament_simulation(predict_func_state, model_name):
    if predict_func_state is None:
        return "⚠️ Please train a model first using the 'Train your model' button!"

    # 1. Group Stage
    qualified_32 = simulate_group_stage(ACTUAL_GROUPS, predict_func_state, model_name)
    
    # 2. Knockout Stages (Simulated in a single pass)
    # The results from simulate_knockout_stage are lists of winners and formatted output strings
    round_of_16_teams, output_r32 = simulate_knockout_stage(qualified_32, predict_func_state, "Round of 32 (1/16 Finals)", model_name)
    quarter_final_teams, output_r16 = simulate_knockout_stage(round_of_16_teams, predict_func_state, "Round of 16 (Octavos)", model_name)
    semi_final_teams, output_qf = simulate_knockout_stage(quarter_final_teams, predict_func_state, "Quarter-Finals (Cuartos)", model_name)
    finalists, output_sf = simulate_knockout_stage(semi_final_teams, predict_func_state, "Semi-Finals (Semifinales)", model_name)
    champion, output_final = simulate_knockout_stage(finalists, predict_func_state, "The FINAL", model_name)
    
    # Get the final champion name
    champion_name = champion[0]

    # Use a single triple-quoted f-string for the entire output
    full_output = f"""
# 🗺️ World Cup 2026 Simulation Bracket

---

## ✅ Qualified for Round of 32:
* {", ".join(qualified_32)}

---

## 🏟️ Knockout Stages
{output_r32}
---
{output_r16}
-----
{output_qf}
---
{output_sf}
---
{output_final}

---

# 👑 WORLD CUP 2026 CHAMPION: **{champion_name}**
"""
    # Note: Using textwrap.dedent could clean up the leading whitespace if needed, 
    # but for simple Gradio output, this is usually acceptable.
    
    return full_output

# Data Visualization Functions
def create_pie_chart(panda_data):
    """
    Creates a pie chart showing winner prediction percentages.
    
    Args:
        panda_data: DataFrame with 'team' and 'percentage' columns
        
    Returns:
        Plotly pie chart figure
    """
    if panda_data.empty:
        return px.pie(title="Winner predictions (No data yet)")
    
    return px.pie(
        panda_data,
        names='team',  
        values='percentage', 
        title="Winner predictions"
    )

def get_and_create_pie():
    """Fetches latest prediction data and generates pie chart."""
    panda_data = update_procentage(winner_predictions_cursor) 
    return create_pie_chart(panda_data)

# User Prediction Functions
def select_and_update(user, team):
    """
    Records user's winner prediction and updates the visualization.
    
    Args:
        user: Username string
        team: Selected team name
        
    Returns:
        Tuple of (updated pie chart, cleared username field, cleared dropdown)
    """
    if user and team:
        select_winner(user, team, winner_predictions_cursor, winner_predictions_db)
    
    return get_and_create_pie(), gr.update(value=""), gr.update(value=None)

# Model Training Functions
def train_model(model_name, current_model, current_predict_function):
    """
    Initializes the selected ML model and its prediction function.
    
    Args:
        model_name: Name of the model to train ("LogisticRegression" or "RidgeClassifierCV")
        current_model: Current model state (unused, maintained for state management)
        current_predict_function: Current prediction function state
        
    Returns:
        Tuple of (updated dropdown, model object, prediction function)
    """
    if model_name == "LogisticRegression":
        model = logistic_model
        predict_function = predict_match_with_logistic_regression
        
    elif model_name == "RidgeClassifierCV":
        model = ridge_model
        predict_function = predict_match_with_ridge_classifier

    elif model_name == "RandomForest":
        model = forest_model
        predict_function = predict_match_with_random_forest
        
    else:
        return gr.update(value=model_name), current_model, current_predict_function

    return gr.update(value=model_name), model, predict_function

# Match Prediction Functions
def run_prediction(home_team, away_team, predict_func_state, model_name):
    """
    Predicts the outcome of a match between two teams.
    """
    if predict_func_state is None:
        return "⚠️ Please train a model first!" 
    probabilities = None
    
    if model_name == "LogisticRegression" or model_name == "RandomForest":
        predicted_winner, probabilities = predict_func_state(home_team, away_team, 2026) 
    else:
        predicted_winner = predict_func_state(home_team, away_team, 2026) 
    
    output_text = f"**{home_team}** vs **{away_team}**\n\n"
    output_text += f"🏆 Predicted Result: **{predicted_winner}**\n"

    if model_name == "LogisticRegression" or model_name == "RandomForest":
        output_text += f"{probabilities}"

    return output_text

# Gradio Interface
with gr.Blocks() as page:

    # State management for model and prediction function
    model_state = gr.State(value=None)
    predict_function_state = gr.State(value=None)
    probabilities_state = gr.State(value=None)

    # Header Section
    gr.Markdown("# 🏆 Welcome to World Cup 2026 Prediction")
    
    # User Prediction Section
    winner_dropdown = gr.Dropdown(
        countries, 
        label="Winner", 
        info="Who do you think will win this World Cup?",
        value=None
    )
    user = gr.Textbox(label="Username", placeholder="Enter your name")
    submit = gr.Button("Submit", variant="primary")
    plot = gr.Plot(label="Top picks")

    # Model Training Section
    gr.Markdown("---")
    gr.Markdown("## 🤖 Model Training")
    
    model_dropdown = gr.Dropdown(
        ["LogisticRegression", "RidgeClassifierCV", "RandomForest"],
        label="Choose your model"
    )
    train = gr.Button("Train your model", variant="primary")
    
    train.click(
        fn=train_model,
        inputs=[model_dropdown, model_state, predict_function_state], 
        outputs=[model_dropdown, model_state, predict_function_state] 
    )

    # Match Prediction Section
    gr.Markdown("---")
    gr.Markdown("## ⚽ Predict a Match")
    
    with gr.Row():
        HomeTeam = gr.Dropdown(
            countries, 
            label="Home Team"
        )
        AwayTeam = gr.Dropdown(
            countries,
            label="Away Team"
        )
    
    predict = gr.Button("Predict", variant="primary")
    prediction_output = gr.Markdown(label="Match Prediction Result") 
    
    predict.click(
        fn=run_prediction, 
        inputs=[HomeTeam, AwayTeam, predict_function_state, model_dropdown], 
        outputs=[prediction_output]
    )
    
    # --- FULL TOURNAMENT SIMULATION SECTION (New Addition) ---
    gr.Markdown("---")
    gr.Markdown("## 🌐 Full Tournament Simulation")
    
    simulate_btn = gr.Button("Run World Cup 2026 Simulation", variant="secondary")
    simulation_output = gr.Markdown(label="Tournament Bracket & Winner")
    
    simulate_btn.click(
        fn=run_full_tournament_simulation, 
        inputs=[predict_function_state, model_dropdown], 
        outputs=[simulation_output]
    )
    # --- END NEW SECTION ---

    # Event Handlers
    submit.click(
        fn=select_and_update, 
        inputs=[user, winner_dropdown], 
        outputs=[plot, user, winner_dropdown]
    )

    page.load(
        fn=get_and_create_pie, 
        inputs=None, 
        outputs=plot
    )

# Application Entry Point
if __name__ == "__main__":
    page.launch(share=True)
