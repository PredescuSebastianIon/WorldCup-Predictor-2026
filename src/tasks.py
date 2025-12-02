from invoke import task
import time
from scrape.fifa_rankings import fetch_latest
from data_process import (
    merge as merge_data,
    filter_all_matches,
    enrich_dataset,
    filter_relevant_teams,
    split_dataset,
)
DATA_DIR = "../data"
SCRAPE_DIR = "./scrape"
MODELS_DIR = "./models"

@task(name = "see")
def see_app(c):
    """
    Open app in default browser
    """
    c.run(f"open http://127.0.0.1:7860")

@task(name = "build", 
      help = {
          "background": "Add --background for background process (default is foreground)"
    })
def build(c, background = False):
    """
    Start the server
    """
    background = bool(background)
    if background:
        c.run("python main.py", disown=background)
    else:
        c.run("python main.py", disown=background, pty=True)

@task(name = "stop")
def stop(c):
    """
    Stop server if running in background (otherwise it does nothing)
    """
    c.run("lsof -i tcp:7860 | grep python | awk '{print $2}' | uniq | xargs kill -9", warn = True)

@task(name = "clean", pre=[stop])
def clean(c):
    """
    Delete cache and data matches for new scraping
    """
    print("Deleting all cache")
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +")
    print("Deleting data matches")
    c.run(f"rm -rf {DATA_DIR}/matches*.csv")

@task(name = "scrape")
def scrape(c):
    """
    Get data matches
    """
    c.run(f"python {SCRAPE_DIR}/scraper.py")
    c.run(f"python {SCRAPE_DIR}/cleaning.py")

@task(name = "fifa-latest")
def fifa_latest(c):
    fetch_latest()

@task(name = "merge")
def merge_task(c):
    merge_data()

@task(name = "model")
def create_models(c, model_name):
    """
    Create models: Logistic Regression and Ridge Classifier CV
    Usage: inv model -m <model_name>
    <model_name> = {logistic, ridge}
    """
    if model_name == "logistic":
        c.run(f"python {MODELS_DIR}/logistic_regression.py")
    
    if model_name == "ridge":
        c.run(f"python {MODELS_DIR}/ridge_classifier_cv.py")

@task(name="filter-matches")
def filter_matches_task(c, min_year=1993):
    """
    Filter all_matches.csv to keep only matches from min_year onwards.
    Writes data/processed/all_matches_filtered.csv
    """
    # Invoke passes CLI args as strings; make sure it's int
    min_year = int(min_year)
    filter_all_matches(min_year=min_year)


@task(name="enrich")
def enrich_task(c):
    """
    Enrich filtered matches with FIFA rankings.
    Uses merged_data.csv and all_matches_filtered.csv
    Writes data/processed/all_matches_enriched.csv
    """
    enrich_dataset()


@task(name="wc-filter")
def wc_filter_task(c):
    """
    Keep only matches where both teams are WC-relevant
    (qualified / hosts / still able to qualify) based on teams_master.csv.
    Writes data/processed/all_matches_relevant_teams.csv
    """
    filter_relevant_teams()


@task(name="split")
def split_task(c):
    """
    Time-based train / val / test split on all_matches_relevant_teams.csv
    Writes matches_train.csv, matches_val.csv, matches_test.csv
    """
    split_dataset()


@task(name="build-data")
def build_data_task(c):
    """
    Full data pipeline (rankings + matches) for ML
      1) merge FIFA rankings -> merged_data.csv
      2) filter all_matches.csv -> all_matches_filtered.csv
      3) enrich with rankings -> all_matches_enriched.csv
      4) filter to WC-relevant teams -> all_matches_relevant_teams.csv
      5) time-based split -> matches_train/val/test.csv
    """
    print("Step 1: merge FIFA rankings...")
    merge_data()

    print("Step 2: filter matches (>= 1993)...")
    filter_all_matches(min_year=1993)

    print("Step 3: enrich matches with rankings...")
    enrich_dataset()

    print("Step 4: filter WC-relevant teams...")
    filter_relevant_teams()

    print("Step 5: split into train / val / test...")
    split_dataset()

    print("Build-data pipeline completed.")


@task(name = "all", pre = [clean])
def all(c):
    """
    Run stop, clean, scrape, see and build in this order
    """
    print("Getting data for all matches")
    fetch_latest()
    scrape(c)
    print("Merging FIFA rankings (merged_data.csv)...")
    merge_data()

    print("Filtering & enriching matches...")
    filter_all_matches(min_year=1993)
    enrich_dataset()

    print("Filtering to WC-relevant teams...")
    filter_relevant_teams()

    print("Time-based split into train / val / test...")
    split_dataset()
    print("Please return to terminal to see when app is ready")
    print("When app is ready, refresh")
    time.sleep(2)
    see_app(c)
    build(c)
