# FPL Predictor using a pre-trained model to predict player points based on the features implemented
# Loads the trained Random Forest model created in ml_training
import pickle
import pandas as pd

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
    def predict_points(self, player_data):
        # Extracts the relevant features from the player data and uses the model to predict points
        X = player_data[self.features]
        # uses the trained model to predict points & returns an array of the predicted points
        return self.model.predict(X)
    
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
