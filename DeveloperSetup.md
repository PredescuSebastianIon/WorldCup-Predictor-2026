# Dev manual

This guide provides the necessary steps to clone, install, and run the project.

## Environment setup

- Clone repository
- Install python3.12 (App is not compatible with python3.14)
- Create an virtual envinroment
- Activate envinroment
- Install all dependencies

```bash
git clone https://github.com/PredescuSebastianIon/WorldCup-Predictor-2026.git
# Install python3.12
cd WorldCup-Predictor-2026
python -m venv .worldcup
source .worldcup/bin/activate
pip install -r requirements.txt
```

## Build and Run

Go in `src` project. You are going to find a file named `tasks.py`. <br>
This project is automated with `PyInvoke`. You can see all commands with:
* inv --list
* inv --help [task] - manual for each task

## Supported Commands

Below are all automation commands available in this project via PyInvoke.

## Server & Browser

#### `inv build [--background]` Start the server (`main.py`). Runs in the foreground by default. Use `--background` to run it as a detached process.

#### `inv see` Open the application in the default web browser at `http://127.0.0.1:7860`.

#### `inv stop` Stop the server if it is running in background mode.

## Cleaning

#### `inv clean` Remove all cache directories and delete stored match data. Automatically triggers `inv stop` before cleaning.


## Scraping & Rankings

#### `inv scrape` Run the scraping workflow to retrieve and clean match data.

#### `inv fifa-latest` Fetch the latest FIFA rankings and write `fifa_latest.csv`.

#### `inv merge` Merge historical and latest FIFA rankings into `merged_data.csv`.

## Data Preparation

#### `inv filter-matches [--min-year=<year>]` Filter `all_matches.csv` to keep matches from the specified year onward. Default year: `1993`.

#### `inv enrich` Enrich filtered matches using FIFA rankings and write `all_matches_enriched.csv`.

#### `inv wc-filter` Filter matches to include only teams relevant to the World Cup.

#### `inv split` Perform a time-based split of the processed dataset into training, validation, and test sets.

## Pipeline Tasks

#### `inv build-data` Execute the full preprocessing pipeline:

1. Merge FIFA rankings
2. Filter matches
3. Enrich with rankings
4. Filter WC-relevant teams
5. Perform train/validation/test split

#### `inv all` Run the complete development workflow in sequence:

1. Stop server
2. Clean cache
3. Scrape match data
4. Fetch latest FIFA rankings
5. Merge and preprocess data
6. Open browser
7. Start server

## Model Training

#### `inv model -m <model_name>` Train a machine learning model. Supported values: `logistic`, `ridge`, `forest`, `poisson`.


**Also, you can check [PyInvoke documentation](https://docs.pyinvoke.org/en/stable/index.html)**.
