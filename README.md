# :soccer: World Cup 2026 Predictor

**Authors**:
- Predescu Sebastian-Ion
- Tudorache Stefan
- Iosif Andrei Constantin

-----

## Overview

### :thinking: What Is This Project?

The **World Cup 2026 Predictor** is a sophisticated tool designed to forecast the results of the upcoming FIFA World Cup. It goes beyond simple intuition or expert opinion by using the power of **data analysis** and **machine learning** to simulate the entire tournament, from the first group match right up to the final.

Think of it as having a **data-driven supercomputer** dedicated entirely to predicting the beautiful game. Our aim is to provide an objective, statistically backed view of the tournament's prospects, eliminating human bias.

### :sparkles: Key Features

This project provides detailed insights and forecasts for the 2026 tournament:

  * ### Match Outcome Predictions

    Forecast the likely winner (or a draw) for every single game in the tournament, offering a probability score to show the confidence in the prediction.

  * ### Historical Learning Engine

    The core system has been rigorously trained on vast amounts of historical World Cup and international match data. This allows the model to deeply understand team strengths, dynamics, and historical trends that often repeat themselves.

  * ### Full Tournament Simulation

    Beyond individual matches, the predictor simulates the complete progression of the competition, showing how teams advance through the:

      * **Group Stages** (Predicting qualifiers and final group standings)
      * **Knockout Rounds** (Round of 32, 16, Quarter-finals, etc.)

  * ### Final Champion Forecast

    Identify the most probable teams to lift the trophy based on thousands of independent simulations run by the algorithm.

-----

### :gear: How It Works

This predictor uses smart computer algorithms to make its forecasts:

1.  **Data Collection & Cleaning:** We start by gathering a large, clean database of past international matches, team rankings, and performance statistics.
2.  **Training the Algorithm:** The system is fed this historical data and learns to recognize complex patterns that led to wins, losses, or draws. The current version uses three different models that can predict the outcome.
3.  **Forecasting:** Once trained, the system analyzes the current teams competing in the 2026 World Cup and applies the learned patterns to predict the probability of every possible match result.
4.  **Simulation & Output:** We run the entire World Cup schedule (all 104 matches) many times over. The final output provides the most likely outcomes based on the majority consensus of these intensive simulations.

## :books: Further Reading

For a detailed, technical deep-dive into the specific algorithms used, feature engineering, and performance metrics, please refer to the dedicated design documentation:

  * **[design.md](https://github.com/PredescuSebastianIon/WorldCup-Predictor-2026/blob/main/Design.md)**

## Project Setup for devs :gear:

Please read [Dev Manual](DeveloperSetup.md) for detailed intructions.

## :rocket: Usage / Live Demo

You can try the live demo of this project **[here](https://test.com)**.

### How to Use

1. **Make Your Pick**  
    Enter your name and choose the team you believe will win the World Cup.

2. **View Live Prediction Stats**  
   Instantly see updated community prediction statistics based on all user picks.

3. **Explore the Model Training Section**  
   Scroll down to the **Model Training** area to select the machine learning model you want to train.

4. **Run Predictions or Simulations**  
   After selecting a model, you can:
   - Predict the result of a specific match, **or**
   - Run a full tournament simulation and see the predicted champion.

## License

This project is under [MIT License](LICENSE).
