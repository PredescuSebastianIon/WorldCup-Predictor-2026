import pandas as pd
import os
import gradio as gr
from countries import countries
from database import select_winner, update_procentage 
import plotly.express as px

def create_pie_chart(panda_data):
    if panda_data.empty:
        return px.pie(title="Winner predictions (No data yet)")
    return px.pie(
        panda_data,
        names='team',  
        values='percentage', 
        title="Winner predictions"
    )

def get_and_create_pie():
    panda_data = update_procentage()
    return create_pie_chart(panda_data)

def select_and_update(user, team):
    if user and team:
        select_winner(user, team)
    return get_and_create_pie()

# Gradio App
with gr.Blocks() as page:
    gr.Markdown("Welcome to WorldCup 2026 prediction")
    winner_dropdown = gr.Dropdown(
        countries, 
        label="Winner", 
        info="Who do you think will win this WorldCup?",
        value = None
    )
    user = gr.Textbox(label="Username")
    submit = gr.Button("Submit", variant="primary")

    plot = gr.Plot(label="Top picks")

    submit.click(
        fn=select_and_update, 
        inputs=[user, winner_dropdown], 
        outputs=plot
    )

    page.load(
        fn=get_and_create_pie, 
        inputs=None, 
        outputs=plot
    )

# Entrypoint
if __name__ == "__main__":
    page.launch(share = False)
