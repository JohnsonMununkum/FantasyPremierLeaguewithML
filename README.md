# FantasyPremierLeaguewithML
An AI-powered Fantasy Premier League prediction and squad optimization web application. The system uses a trained Random Forest machine learning model to predict player points for the upcoming gameweek and applies Linear Programming to generate optimal 15-player squads within FPL constraints.

The application is built with Flask and deployed on Render, with automated data updates every 6 hours using GitHub Actions.

Live Application: https://fpl-intelligence-web.onrender.com/

Screencast Demo: https://youtu.be/8K4ZN0d-I4c

## Features 
- Player Predictions - Predicts Fantasy Premier League points for all players for the upcoming gameweek using the Random Forest model (R² = 0.503, MAE = 0.73)

- Top & Bottom Players - Displays the top 10 & bottom 10 predicted players with position filtering

- Player Profiles - Individual player pages showing season statistics, predicted points, goals, assists, bonus points & ownership percentage.

- Squad Optimization - Generates 3 distinct optimal 15-player squads using PuLP Linear Programming, within FPL constraints (£100m budget, position constraints(2 GK, 5 DEF, 5 MID, 3 FWD), & the 3 player per team limit)

- Captain Recommendation - Picks the highest predicted points player in each squad as captain & doubles their points

- Auto Data Updates - GitHub Actions workflow gets fresh FPL API data every 6 hours & recalculates all features & predictions

- Dark Mode - Toggle between light and dark themes

- Real-Time Search - Search and filter players by name and position on the predictions page

## Technology Stack
- Backend - Python, Flask
- Machine Learning - scikit-learn (Random Forest)
- Optimization - PuLP (Linear Programming)
- Database - SQLite
- Data Source - Official FPL REST API
- Frontend - HTML, CSS, Bootstrap 5, JavaScript
- Deployment - Render
- CI/CD - GitHub Actions

## Project Structure
```
    FantasyPremierLeaguewithML/
    ├── .github/
    │   └── workflows/
    │       └── update-fpl-data.yml
    ├── flask_app/
    │   ├── models/
    │   │   ├── fpl_data.db
    │   │   ├── fpl_features.csv
    │   │   └── fpl_predictor_model.pkl
    │   ├── static/
    │   │   ├── css/
    │   │   └── js/
    │   ├── templates/
    │   │   ├── base.html
    │   │   ├── index.html
    │   │   ├── player.html
    │   │   ├── predictions.html
    │   │   └── squads.html
    │   ├── utils/
    │   │   ├── data_fetcher.py
    │   │   ├── optimizer.py
    │   │   ├── predictor.py
    │   │   └── training.py
    │   ├── app.py
    │   ├── Procfile
    │   ├── requirements.txt
    │   └── scheduler.py
    ├── Pulp/
    ├── queries/
    ├── bestmodel.ipynb
    ├── collectingdata.ipynb
    ├── featureengineering.ipynb
    ├── fpl_data.db
    ├── fpl_features.csv
    ├── fpl_predictor_model.pkl
    ├── mltraining.ipynb
    ├── model_summary.txt
    ├── render.yaml
    └── README.md
```

## Machine Learning Model
The prediction model is a Random Forest Regressor trained on historical FPL gameweek data across all available gameweeks this season. The features used are:

- minutes - Minutes played in recent gameweeks
- rolling_avg_points - Rolling average of FPL points
- price - current FPL price (£)
- clean_sheets_rolling_avg - Rolling average of clean sheets (useful for defenders & goalkeepers)
- opponent_difficulty - Fixture Difficulty rating based on opponent difficulty (1-10)
- is_home - Home or away fixture
- position_encoded - Encoded position (GK/DEF/MID/FWD)
- form - FPL API form rating
- goals_scored - Season goals scored
- assists - season assists

Model performance: R² = 0.503, MAE = 0.73

## Running Locally
### Prerequisites
- Python 3.10 or higher
- pip

### Installation
1. Clone the repository:
```
git clone https://github.com/JohnsonMununkum/FantasyPremierLeaguewithML.git
cd FantasyPremierLeaguewithML/flask_app
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Start the Flask development server
```
python app.py
```

4. Open your browser and go to (http://localhost:5000)

### Populating the Database
If the database is empty, trigger a data update by visiting.
```
http://localhost:5000/api/trigger-update
```
This gets the latest player data from the FPL API, recalculates all the features, and prepares predictions.

### Retraining the Model
To retrain the model on the latest season data, run the training script from the flask_app directory:
```
python utils/training.py
```
This gets the full gameweek history for all players from the FPL API, engineers features, trains the Random Forest Model, and saves it to models/fpl_predictor_model.pkl. The training data is saved to a separate table training_features table and does not affect the production database.

## Deployment on Render
The application is deployed on Render as a web service.

### Manual Deployment
1. Fork or push the repository to GitHub.
2. Create a new Web Service on Render and connect your GitHub repository.
3. Set the following configuration:
    - Root Directory: flask_app
    - Build Command: pip install -r requirements.txt
    - Start Command: python app.py
4. Add the following environment variables in Render:
    - RENDER_API_KEY - Your Render API key
    - RENDER_SERVICE_ID - Your Render service ID

### Automated Updates (GitHub Actions)
The workflow in .github/workflows/update-fpl-data.yml automatically calls the /api/trigger-update endpoint every 6 hours to keep the predictions current. Make sure your RENDER_API_KEY and RENDER_SERVICE_ID are set as GitHub repository secrets.

## API Endpoints
- /api/predictions/top - GET - Top 10 predicted players
- /api/predictions/bottom - GET - Bottom 10 predicted players
- /api/predictions/all-players - GET - All player predictions
- /api/player/<id> - GET - Individual player details and prediction
- /api/optimize - GET - Single optimal squad
- /api/optimize/multiple - GET - Three distinct optimal squads
- /api/trigger-update - GET/POST - Trigger FPL data refresh
- /health - GET - Application health check 

## Author
Johnson Mununkum
BSc (Hons) Computing in Software Development 
Student ID: G00419319