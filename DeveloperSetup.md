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

Supported commands:
* inv build $-$ start the server (default is in foreground)
* inv see $-$ open the server in your default browser
* inv stop $-$ stop the server (if running in background)
* inv clean $-$ clean all cache data
* inv all $-$ stop server, clean, scrape, see, build

**Also, you can check [PyInvoke documentation](https://docs.pyinvoke.org/en/stable/index.html)**.
