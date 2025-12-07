# 💻 App Design: World Cup 2026 Prediction

## 🚀 Technologies Breakdown

This application is built as a **full-stack Machine Learning project** leveraging Python and its ecosystem. It encompasses **data scraping**, **data processing**, **Machine Learning model training**, a **database** for user input, and a **web interface**.

---

### I. 🌐 Frontend & Interface

The application's interface is built entirely using **Gradio**.

* **Gradio (`gradio as gr`)**: Used to create the web application (`main.py`) with minimal effort. It handles the layout (`gr.Blocks`, `gr.Row`), input components (`gr.Dropdown`, `gr.Textbox`, `gr.Button`), and output displays (`gr.Markdown`, `gr.Plot`). This allows the Python logic to be directly exposed as an interactive web UI.
* **Plotly Express (`plotly.express as px`)**: Used for generating the **"Top picks" pie chart** visualization within the Gradio interface, making the user predictions visually engaging.

---

### II. 🧠 Backend & Machine Learning

The core functionality involves data handling and predictive modeling, implemented in various Python scripts under the `src/` and `src/models/` directories.

* **Pandas (`import pandas as pd`)**: The fundamental library for **data manipulation, cleaning, and preparation** (used in `scraper.py`, `cleaning.py`, `data_process.py`, and all model scripts).
* **Scikit-learn (sklearn)**: The primary library for building and evaluating ML models.
    * **Logistic Regression (`LogisticRegression`)**: Used for one of the predictive models (`logistic_regression.py`).
    * **Ridge Classifier CV (`RidgeClassifierCV`)**: Used for the second predictive model (`ridge_classifier_cv.py`).
    * **Random Forest (`RandomForestClassifier`)**: Included in the repository for comparison or alternative use (`random_forest.py`).
* **Tournament Simulation**: The logic for running the **Group Stage** and **Knockout Stages** is custom Python code within `main.py`, relying on the trained ML prediction functions.

---

### III. 💾 Data & Database Management

Data is sourced from the web and stored locally.

* **Requests/BeautifulSoup (`import requests`, `from bs4 import BeautifulSoup`)**: Used in `scraper.py` to **fetch and parse HTML** from Wikipedia pages to gather historical World Cup match results.
* **SQLite (via standard Python libraries)**: Used in `database.py` to manage `winner_predictions.db`. This database stores user winner predictions and is used to calculate and update the percentages for the pie chart.
* **FIFA API/Requests**: Used in `fifa_rankings.py` to fetch the **latest FIFA rankings** for use in model feature engineering.

---

### IV. 🛠️ Development & Task Automation

The development workflow is managed using the `invoke` library.

* **Invoke (`from invoke import task`)**: Used in `tasks.py` to create a command-line interface (CLI) for common development tasks, such as:
    * **`scrape`**: Run data collection and cleaning.
    * **`model`**: Train the Logistic or Ridge models.
    * **`build`**: Start the Gradio web application.
    * **`clean` / `stop`**: Maintain the environment and server processes.
