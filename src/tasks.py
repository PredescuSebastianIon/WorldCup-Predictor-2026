from invoke import task
import time

DATA_DIR = "../data"
SCRAPE_DIR = "./scrape"

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

@task(name = "all", pre = [clean])
def all(c):
    """
    Run stop, clean, scrape, see and build in this order
    """
    print("Getting data for all matches")
    scrape(c)
    print("Please return to terminal to see when app is ready")
    print("When app is ready, refresh")
    time.sleep(2)
    see_app(c)
    build(c)
