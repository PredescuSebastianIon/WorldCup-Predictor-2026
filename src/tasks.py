from invoke import task
import time
from scrape.fifa_rankings import fetch_latest

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

@task(name = "all", pre = [clean])
def all(c):
    """
    Run stop, clean, scrape, see and build in this order
    """
    print("Getting data for all matches")
    fetch_latest()
    scrape(c)
    print("Please return to terminal to see when app is ready")
    print("When app is ready, refresh")
    time.sleep(2)
    see_app(c)
    build(c)
