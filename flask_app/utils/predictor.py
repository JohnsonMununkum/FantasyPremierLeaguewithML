# FPL Predictor using a pre-trained model to predict player points based on the features implemented
# Loads the trained Random Forest model created in ml_training
import pickle
import pandas as pd
import numpy as np

class FPLPredictor:
    def __init__(self, model_path='models/fpl_predictor_model.pkl'):
        # Loads the random Forest Model from the pickel file created in ml_training
        self.model = pickle.load(open(model_path, 'rb'))
        # Defining the features used for predictions
        # Features were all engineered in the featureengineering file
        self.features = [
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
    
    # Predicts points for a given player data DataFrame using the loaded model and defined features
    # added a fallback mechanism to use rolling average points if the model predictions are not working properly
    def predict_points(self, player_data):
        # Extracting the features
        X = player_data[self.features]
        
        # handling mising values
        X = X.fillna(0)
        
        try:
            # Getting the model predictions
            predictions = self.model.predict(X)
            
            # Clipping negative predictions to 0
            predictions = np.maximum(predictions, 0)
            
            # Checking if the models predictions are too uniform
            unique_predictions = len(np.unique(predictions))

            # If model is only predicting a few unique values
            # it is broken so using rolling average points as a fallback instead
            if unique_predictions < 5:
                print(f"Model only predicts {unique_predictions} unique values")
                print("Using rolling_avg_points as fallback")
                
                # Using rolling average as prediction
                fallback_predictions = player_data['rolling_avg_points'].fillna(0)
                return fallback_predictions.round(1)
            
            # If model is working using the models predictions
            return predictions.round(1)
            
        except Exception as e:
            # Error using the model for predictions, using rolling average points as a fallback instead
            print(f"Prediction failed: {e}")
            print("Using rolling_avg_points as fallback")
            fallback_predictions = player_data['rolling_avg_points'].fillna(0)
            return fallback_predictions.round(1)
    
    # Predicts points for players in a specific position and returns the top 10 players based on predicted points
    def predict_by_position(self, df, position):
        # Filter the dataframe to only include players from the specified position
        pos_df = df[df['position'] == position].copy()
        # generates predictions for this positions players
        pos_df['predicted_points'] = self.predict_points(pos_df)
        # Returns the top 10 players in the specified position based on predicted points, including their name, team, price, and predicted points
        return pos_df.nlargest(10, 'predicted_points')[
            ['name', 'team', 'price', 'predicted_points']
        ].to_dict('records')
