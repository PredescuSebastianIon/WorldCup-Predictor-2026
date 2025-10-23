import pandas as pd
import os
import gradio as gr
from countries import countries

with gr.Blocks() as page:
    gr.Markdown("Welcome to WorldCup 2026 prediction")
    with gr.Row():
        winner_dropdown = gr.Dropdown(
            countries, 
            label="Winner", 
            info="Who do you think will win this WorldCup?",
            value = None
        ),

# ---------- Entrypoint ----------
if __name__ == "__main__":
    page.launch(share = False)
