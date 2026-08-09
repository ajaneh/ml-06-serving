# Project Documentation


## Serving a Model (Hosting an Endpoint)

- [**Online Predictions**](./api-predict.md)

## Deployment Options

- [**Render**](./render.md) - free, easier, CC required

## Phase 4. Technical Modification

In phase 4 I made a fairly easy change, I wanted endpoint predictions to persist, so I added a logging statement to the API request, now the input features, and the prediction are logged.

## Phase 5. Custom Project

### Basis and Data

The model trained and served in this project predicts 1 of 3 penguins species, a Random Tree Classifier was trained on Seaborn's Penguins dataset.

### Summary, Example Model and Serving Approach

For phase 5, I didn't alter the model. I altered the endpoint. "serve_alex.py" is now used to host the model and prediction API.

The API was modified to
- Return probability along with prediction
- Return proper response codes
- Provide details about why a request failed

Project was hosted on Render
![Render](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.31.36.png?raw=true)

![Custom Error](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.33.37.png?raw=true)


Returns 422 and detail if the request is bad


![Detailed Errors](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.27.46.png?raw=true)




![Probability Added](https://github.com/ajaneh/ml-06-serving/blob/main/docs/images/Screenshot%202026-08-09%2014.34.13.png?raw=true)


Predictions now come with model probability 
