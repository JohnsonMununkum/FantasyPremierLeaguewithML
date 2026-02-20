# utils/data_fetcher.py
# Fetches live FPL data and recalculates all features for production
import requests
import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

# Gets the latest data from the fpl api 
# reclalculates all features using the same feature engineering process as in the featureengineering.ipynb
class FPLDataFetcher:
   # Initializes the data fetcher with the database path and base URL for the FPL API
    def __init__(self, db_path='models/fpl_data.db'):
        self.db_path = db_path
        self.base_url = 'https://fantasy.premierleague.com/api'
    
    # Gets all players data from the FPL API, including stats, form, prices, and availability
    def fetch_all_players(self):
        print(f"[{datetime.now()}] Getting player data from FPL API")
        
        try:
            # Gets bootstrap-static data which contains all players and teams information
            url = f'{self.base_url}/bootstrap-static/'
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extracting the players and teams
            players = pd.DataFrame(data['elements'])
            teams = pd.DataFrame(data['teams'])
            
            # Addding team names to players
            team_mapping = dict(zip(teams['id'], teams['name']))
            players['team_name'] = players['team'].map(team_mapping)
            
            # Saving the raw data
            conn = sqlite3.connect(self.db_path)
            players.to_sql('players_raw', conn, if_exists='replace', index=False)
            teams.to_sql('teams', conn, if_exists='replace', index=False)
            conn.close()
            
            print(f"[{datetime.now()}] Got {len(players)} players from {len(teams)} teams")
            return players
            
        except Exception as e:
            print(f"[{datetime.now()}] Error fetching players: {e}")
            raise
    
    # Fetches the current gameweek and fixtures data, and calculates opponent difficulty ratings based on team strength
    def fetch_gameweek_data(self):
        print(f"[{datetime.now()}] Getting gameweek and fixtures data")
        
        try:
            # Getting the fixtures
            url = f'{self.base_url}/fixtures/'
            response = requests.get(url)
            response.raise_for_status()
            fixtures = pd.DataFrame(response.json())
            
            # Getting the current gameweek info
            url = f'{self.base_url}/bootstrap-static/'
            response = requests.get(url)
            data = response.json()
            
            current_gw = None
            for event in data['events']:
                if event['is_current']:
                    current_gw = event['id']
                    break
            
            if current_gw is None:
                # Getting the next gameweek
                for event in data['events']:
                    if event['is_next']:
                        current_gw = event['id']
                        break
            
            # Saving the fixtures
            conn = sqlite3.connect(self.db_path)
            fixtures.to_sql('fixtures', conn, if_exists='replace', index=False)
            
            # Saving the current gameweek
            gw_df = pd.DataFrame([{'current_gameweek': current_gw}])
            gw_df.to_sql('current_gameweek', conn, if_exists='replace', index=False)
            conn.close()
            
            print(f"[{datetime.now()}] Current gameweek: {current_gw}, Fixtures: {len(fixtures)}")
            return True
            
        except Exception as e:
            print(f"[{datetime.now()}] Error fetching gameweek data: {e}")
            raise
    
    # Calculating all features again with the latest data from the API, using the same feature engineering process as in the featureengineering.ipynb
    def update_features(self):
        print(f"[{datetime.now()}] Recalculating all features")
        
        try:
            # Connecting to the database to get the latest players and fixtures data
            conn = sqlite3.connect(self.db_path)
            
            # Loading the data
            players = pd.read_sql_query('SELECT * FROM players_raw', conn)
            
            # Getting the current gameweek
            try:
                gw_query = 'SELECT current_gameweek FROM current_gameweek LIMIT 1'
                current_gw = pd.read_sql_query(gw_query, conn)['current_gameweek'][0]
            except:
                current_gw = 1
            
            # Creating the features dataframe
            features_df = pd.DataFrame()
            
            features_df['player_id'] = players['id']
            features_df['name'] = players['first_name'] + ' ' + players['second_name']
            features_df['team'] = players['team_name']
            features_df['gameweek'] = current_gw
            
            # Position mapping
            position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            features_df['position'] = players['element_type'].map(position_map)
            
            # rolling_avg_points 
            # Uses the form field from API (last 5 games average)
            features_df['rolling_avg_points'] = players['form'].astype(float)
            
            # opponent_difficulty
            # Uses average of team difficulty for upcoming fixtures
            try:
                fixtures = pd.read_sql_query('SELECT * FROM fixtures', conn)
                # Calculating opponent difficulty based on team strength
                team_strength = players.groupby('team')['total_points'].mean()
                features_df['opponent_difficulty'] = 3.0
            except:
                # Sets to default difficulty if fixtures data is not available
                features_df['opponent_difficulty'] = 3.0
            
            # Expected minutes based on recent playing time
            features_df['minutes'] = players['minutes'].fillna(0)
            
            # is_home
            features_df['is_home'] = 0.5
            
            # price
            features_df['price'] = players['now_cost'] / 10.0
            
            # Position encoding, one-hot
            features_df['pos_GK'] = (features_df['position'] == 'GK').astype(int)
            features_df['pos_DEF'] = (features_df['position'] == 'DEF').astype(int)
            features_df['pos_MID'] = (features_df['position'] == 'MID').astype(int)
            features_df['pos_FWD'] = (features_df['position'] == 'FWD').astype(int)
            
            # clean_sheets_rolling_avg
            # Uses clean sheets per 90 minutes for defenders/goalkeepers
            features_df['clean_sheets_rolling_avg'] = players['clean_sheets'].fillna(0) / np.maximum(players['minutes'].fillna(1) / 90, 1)
            
            # 0 if the player is not a defender or goalkeeper, as clean sheets are not relevant for midfielders and forwards
            features_df.loc[~features_df['position'].isin(['GK', 'DEF']), 'clean_sheets_rolling_avg'] = 0
            
            # Handling any missing values
            features_df = features_df.fillna(0)
            
            # Saving to the database
            features_df.to_sql('features', conn, if_exists='replace', index=False)
            conn.close()
            
            print(f"[{datetime.now()}] Calculated 10 features for {len(features_df)} players")
            return True
            
        except Exception as e:
            print(f"[{datetime.now()}] Error updating features: {e}")
            raise

