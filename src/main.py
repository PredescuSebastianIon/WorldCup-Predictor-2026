import pandas as pd
import os
import gradio as gr
from countries import countries
from database import select_winner, update_procentage
from database import winner_predictions_db, winner_predictions_cursor
import plotly.express as px
from models import logistic_model, ridge_model
from models import predict_match_with_logistic_regression, predict_match_with_ridge_classifier

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
    
    # Clear inputs and refresh chart
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
        
    else:
        # No valid model selected, keep current state
        return gr.update(value=model_name), current_model, current_predict_function

    return gr.update(value=model_name), model, predict_function

# Match Prediction Functions
def run_prediction(home_team, away_team, predict_func_state):
    """
    Predicts the outcome of a match between two teams.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        predict_func_state: Trained prediction function from State
        
    Returns:
        Tuple of (formatted prediction text, probability array)
    """
    if predict_func_state is None:
        return "⚠️ Please train a model first!", None 
    
    # Run prediction for World Cup 2026
    # predicted_winner, probabilities = predict_func_state(home_team, away_team, 2026) 
    
    # # Format output with markdown styling
    # output_text = f"**{home_team}** vs **{away_team}**\n\n"
    # output_text += f"🏆 Predicted Winner: **{predicted_winner}**\n"
    # output_text += f"📊 Probabilities (Loss/Draw/Win): **{probabilities}**"

    # return output_text, probabilities

    # Run prediction for World Cup 2026
    predicted_winner = predict_func_state(home_team, away_team, 2026) 
    
    # Format output with markdown styling
    output_text = f"**{home_team}** vs **{away_team}**\n\n"
    output_text += f"🏆 Predicted Result: **{predicted_winner}**\n"

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
        ["LogisticRegression", "RidgeClassifierCV"],
        label="Choose your model"
    )
    train = gr.Button("Train your model", variant="primary")
    
    # Train button updates model and prediction function states
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
    
    # Predict button runs prediction with selected teams
    predict.click(
        fn=run_prediction, 
        inputs=[HomeTeam, AwayTeam, predict_function_state], 
        outputs=[prediction_output]
        # probabilities_state
    )

    # Event Handlers
    # Submit user prediction and refresh chart
    submit.click(
        fn=select_and_update, 
        inputs=[user, winner_dropdown], 
        outputs=[plot, user, winner_dropdown]
    )

    # Load initial pie chart on page load
    page.load(
        fn=get_and_create_pie, 
        inputs=None, 
        outputs=plot
    )

# Application Entry Point
if __name__ == "__main__":
    page.launch(share=False)