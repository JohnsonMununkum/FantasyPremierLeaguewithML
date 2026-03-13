# Gets current season gameweek data from the fpl api API all players
# rebuilds the training database, and retrains the model
import requests
import pandas as pd
import numpy as np
import sqlite3
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

# Constants for API endpoints, database path, and model path
BASE_URL = 'https://fantasy.premierleague.com/api'
DB_PATH = 'models/fpl_data.db'
MODEL_PATH = 'models/fpl_predictor_model.pkl'

# Getting the current season's data from the fpl api & building a training dataset with engineered features
# then training a Random Forest model to predict player points based on historical performance & upcoming fixtures.
data = requests.get(f'{BASE_URL}/bootstrap-static/', timeout=30).json()

# Getting players & teams data from the api resoponse and putting it into a dataframe
players_df = pd.DataFrame(data['elements'])
teams_df = pd.DataFrame(data['teams'])
team_map = dict(zip(teams_df['id'], teams_df['name']))

# Mapping position ids to position names and calculating player price in millions
position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
players_df['team_name'] = players_df['team'].map(team_map)
players_df['position'] = players_df['element_type'].map(position_map)
players_df['price'] = players_df['now_cost'] / 10.0

# Displaying basic info about the data
print(f"{len(players_df)} players, {len(teams_df)} teams")

# Getting upcoming fixtures to calculate opponent difficulty for the next gameweek
print("Getting gameweek history for each player")
all_history = []

# Looping through each player to get their gameweek history data from the api
for i, row in players_df.iterrows():
    player_id = row['id']
    try:
        url = f'{BASE_URL}/element-summary/{player_id}/'
        resp = requests.get(url, timeout=15).json()
        history = resp.get('history', [])
        # For each gameweek in the player's history, storing relevant data along with player info
        for gw in history:
            all_history.append({
                'player_id': player_id,
                'name': row['first_name'] + ' ' + row['second_name'],
                'team': row['team_name'],
                'position': row['position'],
                'price': row['price'],
                'gameweek': gw['round'],
                'minutes': gw['minutes'],
                'goals_scored': gw['goals_scored'],
                'assists': gw['assists'],
                'clean_sheets': gw['clean_sheets'],
                'bonus': gw['bonus'],
                'total_points': gw['total_points'],
                'opponent_team': gw['opponent_team'],
                'was_home': int(gw['was_home']),
            })
    except Exception as e:
        print(f"No gameweek history for {player_id}: {e}")
        continue

    # Printing progress for every 100 players to see how many players have been processed and how many are left
    if i % 100 == 0:
        print(f"{i}/{len(players_df)} players fetched")

# Converting the collected history data into a DataFrame for feature engineering and model training
df = pd.DataFrame(all_history)
print(f"{len(df)} gameweek records fetched")
print(f"Gameweeks: {df['gameweek'].min()} to {df['gameweek'].max()}")

# Calculating opponent difficulty ratings based on the average points scored against each team in the current season
# Using a scale of 1 - 10, where 1 is easiest and 10 is hardest
team_avg_points = df.groupby('team')['total_points'].mean()
min_pts = team_avg_points.min()
max_pts = team_avg_points.max()
team_difficulty = ((max_pts - team_avg_points) / (max_pts - min_pts)) * 9 + 1

# Mapping opponent team id to team name then to difficulty
team_id_to_name = dict(zip(teams_df['id'], teams_df['name']))
df['opponent_name'] = df['opponent_team'].map(team_id_to_name)
df['opponent_difficulty'] = df['opponent_name'].map(team_difficulty).fillna(5.0)

# Engineering features for the model, rolling average points, opponent difficulty, home/away, price, position encoding, and clean sheet rolling average
df = df.sort_values(['player_id', 'gameweek'])

# Rolling avg points (last 5 games)
df['rolling_avg_points'] = df.groupby('player_id')['total_points'].transform(
    lambda x: x.rolling(window=5, min_periods=5).mean().shift(1)
).fillna(0)

# Clean sheets rolling avg (last 5 games)
df['clean_sheets_rolling_avg'] = df.groupby('player_id')['clean_sheets'].transform(
    lambda x: x.rolling(window=5, min_periods=5).mean().shift(1)
).fillna(0)

# Clean sheets are only relevant for defenders and goalkeepers, so setting it to 0 for midfielders and forwards to avoid confusion for the model
df.loc[~df['position'].isin(['GK', 'DEF']), 'clean_sheets_rolling_avg'] = 0

# is_home
df['is_home'] = df['was_home']

# Position one-hot encoding
df['pos_GK'] = (df['position'] == 'GK').astype(int)
df['pos_DEF'] = (df['position'] == 'DEF').astype(int)
df['pos_MID'] = (df['position'] == 'MID').astype(int)
df['pos_FWD'] = (df['position'] == 'FWD').astype(int)

# Saving to database
print(f"Saving to {DB_PATH}")
conn = sqlite3.connect(DB_PATH)
# Saving to features table, replacing it if it already exists, and not including the index as a column in the database
df.to_sql('features', conn, if_exists='replace', index=False)
conn.close()

# Training the model using the engineered features
print("Training model")

# Only using records where the player played
df_played = df[df['minutes'] > 0].copy()
print(f"  Training on {len(df_played)} records (players who played)")

# Defining the features to be used for training the model
features = [
    'rolling_avg_points',
    'opponent_difficulty',
    'minutes',
    'is_home',
    'price',
    'pos_GK',
    'pos_DEF',
    'pos_MID',
    'pos_FWD',
    'clean_sheets_rolling_avg'
]

# Preparing the training data by selecting the defined features and filling any missing values with 0
# The target variable is the total points scored by the player in that gameweek
X = df_played[features].fillna(0)
y = df_played['total_points']

# Splitting the data into training and testing sets to evaluate the model's performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest Regressor with specified hyperparameters to predict player points based on the engineered features
# Using random forest even though predicitions are capped even if a player scored 20 points last week
# as it gives out a better R² and MAE compared to XGBoost, which seems to be overfitting with the current hyperparameters and data size
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Training an XGBoost Regressor with specified hyperparameters to predict player points based on the engineered features
# Seeing how different the performance is compared to the Random Forest model, and if it provides better predictions for the optimizer to work with
# as saw that random forest predictions are capped even if a player scored 20 points last week
# the prediction would be around the mean points for that player, which is around 5-6 points, and it would never predict a very high score for a player even if they have the potential to score big in a given week
# MAE is 1.8, R² is 0.203 with XGBoost so the model is overfitting with these hyperparameters
# model = XGBRegressor(
    # n_estimators=100,
    # max_depth=4,
    # learning_rate=0.1,
    # subsample=0.7,
    # colsample_bytree=0.7,
    # min_child_weight=10,
    # reg_alpha=0.1,
    # reg_lambda=1.0,
    # random_state=42,
    # n_jobs=-1
# )
# Fitting the model to the training data
model.fit(X_train, y_train)

# Evaluating the models performance
preds = model.predict(X_test)
print(f"\nModel Performance:")
print(f"MAE:  {mean_absolute_error(y_test, preds):.3f}")
print(f"R²:   {r2_score(y_test, preds):.3f}")
print(f"Prediction range: {preds.min():.1f} - {preds.max():.1f}")

# Feature importance
print(f"\nFeature Importance:")
fi = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
for feat, imp in fi:
    print(f"  {feat}: {imp*100:.1f}%")

# Saving the trained model to a pickel file
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
print(f"\nModel saved to {MODEL_PATH}")