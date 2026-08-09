# ml-06-serving

[![Workflow Guide](https://img.shields.io/badge/Pro--Guide-pro--analytics--02-green)](https://denisecase.github.io/pro-analytics-02/workflow-b-apply-example-project/)
[![Python 3.14](https://img.shields.io/badge/python-3.14%2B-blue?logo=python)](./pyproject.toml)
[![MIT](https://img.shields.io/badge/license-see%20LICENSE-yellow.svg)](./LICENSE)

> Professional Python project: deploying and serving machine learning models.

## Publishing Predictive Engines

A machine learning model learns patterns from data.
But, once trained, the model might just sit on a single computer.

**Serving** a model means wrapping it in a small web service
so anyone can send it a data request over the internet
and get a prediction back.

In this project, we train a model that
identifies penguin species from physical measurements,
then deploy it so you can ask it
**what is the most likely penguin species**
(given the measurements you provided in the request)
from anywhere in the world.

## Project Description

This project focuses on learning to deploy a trained model so others can use it.

We learn to:

- save and load a trained model
- wrap a model in a simple API or script
- validate inputs and handle errors gracefully
- think about drift, versioning, and monitoring

## Project Dependencies

This project needs additional dependencies

```toml
    "fastapi[standard]", # for serving - a web framework for building APIs
    "uvicorn",           # for serving - ASGI server for FastAPI
    "joblib",            # for model serialization (saving and loading models)
```

## Project Process

A `.joblib` file is a serialized Python object that holds
the trained model frozen to disk.

The package `joblib` converts the in-memory **RandomForestClassifier**
(with all its learned decision trees and their weights)
into bytes and writes them to a file.

Loading it back gives us the same trained model
without having to retrain.

This is how serving a trained model works:
train once, save once, load once at startup,
then predict on every incoming request.

## Example Notebook
Links:

- [ml_06_serve_model](notebooks/ml_06_serve_model.ipynb)



## Command Reference

<details>
<summary>Show command reference</summary>

### In a machine terminal (open in your `Repos` folder)

After you get a copy of this repo in your own GitHub account,
open a machine terminal in your `Repos` folder:

```shell
# Replace username with YOUR GitHub username.
git clone https://github.com/ajaneh/ml-06-serving

cd ml-06-serving
code .
```

### In a VS Code terminal

These are listed for convenience.
For best results, follow the detailed instructions in
[pro-analytics-02 guide](https://denisecase.github.io/pro-analytics-02/).

```shell
uv self update
uv python pin 3.14
uv lock --upgrade
uv sync --extra dev --extra docs --upgrade

uvx pre-commit install
uvx pre-commit autoupdate

git add -A
uvx pre-commit run --all-files
# repeat if changes were made
uvx pre-commit run --all-files

# run the example module to verify the environment (.venv/)
uv run python -m mlstudio.app_case

# TASK 1: train the example model and save it to artifacts/model.joblib.
uv run python -m mlstudio.model_builder_case


#In a new terminal, serve the api
uv run fastapi dev src/mlstudio/serve_alex.py

#Follow visual intructions below to test it out

#Ctrl+C to cancel server instance when finished

# run common chores
uv run ruff format .
uv run ruff check . --fix
uv run python -m pyright
uv run python -m pytest
uv run python -m zensical build

# save progress
git add -A
git commit -m "update"
git push -u origin main
```

</details>


## Alternative Test Route
-Download postman
-POST to `http://127.0.0.1:8000/predict`
-Clear parameters
-Edit body to accept RAW and return JSON
-Copy paste the following to test:
```shell
 {
  "bill_length_mm": 39.1,
  "bill_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750
}
```

## Try a Web-based ML Penguin Predictor on Render


```shell
# PowerShell
curl -X POST https://ml-06-serving.onrender.com/predict
     -H "Content-Type: application/json" `
     -d '{"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750}'

```



## Findings and Visuals

Project was hosted on Render
![Render](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.31.36.png?raw=true)

![Custom Error](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.33.37.png?raw=true)
Returns 422 and detail if the request is bad
![Detailed Errors](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.27.46.png?raw=true)

![Probability Added](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.34.13.png?raw=true)
Predictions now come with model probability 

## Citation

[CITATION.cff](./CITATION.cff)

## License

[MIT](./LICENSE)
