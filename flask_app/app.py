# app.py
# Flask API for FPL Intelligence System
# Provides endpoints for predictions and team optimization
import os
from flask import Flask, jsonify, render_template
import pandas as pd
import sqlite3
from datetime import datetime
from utils.predictor import FPLPredictor
from utils.optimizer import FPLOptimizer
from utils.data_fetcher import FPLDataFetcher

app = Flask(__name__)

# Initializes the predictor and optimizer on startup
print("Initializing FPL Intelligence API")
predictor = FPLPredictor('models/fpl_predictor_model.pkl')
optimizer = FPLOptimizer()
print("Model and optimizer loaded")

# Loads the latest gameweek data from the database
def load_latest_data():
    conn = sqlite3.connect('models/fpl_data.db')
    query = 'SELECT * FROM features WHERE gameweek = (SELECT MAX(gameweek) FROM features)'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# HTML Routes
# Renders the home page with an overview of the project and its features
@app.route('/')
def home():
    # Getting the last update timestamp and calculate time ago
    last_update = "Unknown"
    
    # Getting the last update timestamp from the database
    try:
        conn = sqlite3.connect('models/fpl_data.db')
        query = "SELECT last_update FROM last_update LIMIT 1"
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        #  If the last update timestamp is available, then calculates how long ago it was 
        if len(result) > 0:
            last_update_dt = datetime.fromisoformat(result['last_update'][0])
            
            # Calculates time difference from current time to last update time
            now = datetime.now()
            diff = now - last_update_dt
            
            # Formats as "X time ago"
            if diff.days > 0:
                last_update = f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                last_update = f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                last_update = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
            else:
                last_update = "Just now"
    except Exception as e:
        last_update = "Active"
    
    # Getting player count
    # Setting to 752 as default
    player_count = 752 
    # Getting the player count from the database to display on the home page
    try:
        conn = sqlite3.connect('models/fpl_data.db')
        result = pd.read_sql_query("SELECT COUNT(*) as count FROM players_raw", conn)
        player_count = result['count'][0]
        conn.close()
    except:
        pass

    # Renders the home page template and passes the last update time and player count to be displayed on the page
    return render_template('index.html', last_update=last_update, player_count=player_count)

# Predictions page showing top and bottom players based on the latest model predictions
@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

# Squad builder page where users can see 3 different optimal squad options based on the latest predictions and constraints
@app.route('/squads')
def squads():
    return render_template('squads.html')

# Player Statitics page
@app.route('/player/<int:player_id>')
def player_detail(player_id):
    return render_template('player.html', player_id=player_id)

# API Endpoints
@app.route('/api/info')
def index():
    return jsonify({
        'name': 'FPL Intelligence API',
        'version': '1.0.0',
        'status': 'running',
        'model': {
            'type': 'Random Forest',
            'r_squared': 0.508,
            'mae': 0.78,
            'features': 10
        },
        'endpoints': {
            'health': '/health',
            'predictions_top': '/api/predictions/top',
            'predictions_bottom': '/api/predictions/bottom',
            'predictions_all': '/api/predictions/all',
            'optimize_single': '/api/optimize',
            'optimize_multiple': '/api/optimize/multiple',
            'trigger_update': '/api/trigger-update (POST/GET)',
            'player_detail': '/api/player/<id>'
        }
    })

# Health check endpoint for monitoring
# Shows that the model is loaded and database is accessible
@app.route('/health')
def health():
    try:
        df = load_latest_data()
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'database': 'connected',
            'data_rows': len(df)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

# Endpoint to get top 10 predicted players for next gameweek
# Returns the players with highest predicted points
@app.route('/api/predictions')
@app.route('/api/predictions/top')
def api_predictions_top():
    try:
        # Loads the latest data and generates predictions
        df = load_latest_data()
        df['predicted_points'] = predictor.predict_points(df)
        
        # Gets the top 10 by predicted points
        top_10 = df.nlargest(10, 'predicted_points')[
            ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
        ].to_dict('records')
        
        return jsonify({
            'status': 'success',
            'type': 'top',
            'count': len(top_10),
            'description': 'Players with highest predicted points',
            'predictions': top_10
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint to get bottom 10 predicted players for next gameweek
# Returns the players with lowest predicted points
@app.route('/api/predictions/bottom')
def api_predictions_bottom():
    try:
        # Loads the latest data and generates predictions
        df = load_latest_data()
        df['predicted_points'] = predictor.predict_points(df)
        
        # Gets the bottom 10 by predicted points
        bottom_10 = df.nsmallest(10, 'predicted_points')[
            ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
        ].to_dict('records')
        
        return jsonify({
            'status': 'success',
            'type': 'bottom',
            'count': len(bottom_10),
            'description': 'Players with lowest predicted points',
            'predictions': bottom_10
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint to get both top 10 and bottom 10 predictions in one response
@app.route('/api/predictions/all')
def api_predictions_all():
    try:
        # Loads the latest data and generates predictions
        df = load_latest_data()
        df['predicted_points'] = predictor.predict_points(df)
        
        # Gets the top 10 and bottom 10
        top_10 = df.nlargest(10, 'predicted_points')[
            ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
        ].to_dict('records')
        
        bottom_10 = df.nsmallest(10, 'predicted_points')[
            ['player_id', 'name', 'position', 'team', 'price', 'predicted_points']
        ].to_dict('records')
        
        return jsonify({
            'status': 'success',
            'top_predictions': {
                'count': len(top_10),
                'description': 'Best players to pick',
                'players': top_10
            },
            'bottom_predictions': {
                'count': len(bottom_10),
                'description': 'Worst players to avoid',
                'players': bottom_10
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint to get a single optimal squad of 15 players based on predicted points and constraints
@app.route('/api/optimize')
def api_optimize():
    try:
        # Loada the data and generates predictions
        df = load_latest_data()
        df['predicted_points'] = predictor.predict_points(df)
        
        # Gets the optimal squad
        squad = optimizer.optimize_squad(df, budget=100.0)
        
        return jsonify(squad)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint to get multiple optimal squad options
@app.route('/api/optimize/multiple')
def api_optimize_multiple():
    try:
        # Loads the latest data and generates predictions
        df = load_latest_data()
        df['predicted_points'] = predictor.predict_points(df)
        
        # Generates 3 different optimal squads
        squads = optimizer.optimize_multiple_squads(df, num_squads=3, budget=100.0)
        
        return jsonify({
            'status': 'success',
            'count': len(squads),
            'message': f'Generated {len(squads)} optimal squad options',
            'squads': squads
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint to get information on a specific player
@app.route('/api/player/<int:player_id>')
def api_player_detail(player_id):
    try:
        # Loads the player data
        conn = sqlite3.connect('models/fpl_data.db')
        
        # Getting the specific player from the features table
        query = f"SELECT * FROM features WHERE player_id = {player_id}"
        player_df = pd.read_sql_query(query, conn)
        
        if len(player_df) == 0:
            conn.close()
            return jsonify({
                'status': 'error',
                'message': 'Player not found'
            }), 404
        
        # Getting the predicted points for the player
        player_df['predicted_points'] = predictor.predict_points(player_df)
        player_data = player_df.iloc[0].to_dict()
        
        # Getting raw data for stats
        raw_query = f"SELECT * FROM players_raw WHERE id = {player_id}"
        raw_df = pd.read_sql_query(raw_query, conn)
        conn.close()
        
        if len(raw_df) > 0:
            raw_data = raw_df.iloc[0].to_dict()
            # Adding additional stats to the player data
            player_data['total_points'] = raw_data.get('total_points', 0)
            player_data['goals_scored'] = raw_data.get('goals_scored', 0)
            player_data['assists'] = raw_data.get('assists', 0)
            player_data['bonus'] = raw_data.get('bonus', 0)
            player_data['selected_by_percent'] = raw_data.get('selected_by_percent', '0')
            player_data['transfers_in'] = raw_data.get('transfers_in', 0)
            player_data['transfers_out'] = raw_data.get('transfers_out', 0)
            player_data['photo'] = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{raw_data.get('code', '')}.png"
        
        return jsonify({
            'status': 'success',
            'player': player_data
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# update endpoint that is called by GitHub Actions every 6 hours to update FPL data
# gets the latest player data, gameweek info, and recalculates features to keep predictions current
@app.route('/api/trigger-update', methods=['POST', 'GET'])
def trigger_update():
    try:
        print(f"Updating Data: {datetime.now()}")
        
        # Initializes the data fetcher
        fetcher = FPLDataFetcher()
        
        # Gets and updates all player data
        print("Getting the player data from FPL API")
        fetcher.fetch_all_players()
        
        print("Getting gameweek and fixtures data")
        fetcher.fetch_gameweek_data()
        
        print("Recalculating features based on the latest data")
        fetcher.update_features()

        print("Saving update timestamp")
        fetcher.save_update_timestamp()
        
        print(f"Data updated successfully")

         # Getting the current gameweek number from database
        current_gw = "Unknown"
        try:
            conn = sqlite3.connect('models/fpl_data.db')
            gw_query = 'SELECT current_gameweek FROM current_gameweek LIMIT 1'
            gw_result = pd.read_sql_query(gw_query, conn)
            if len(gw_result) > 0:
                current_gw = gw_result['current_gameweek'][0]
            conn.close()
        except Exception as gw_error:
            print(f"Could not fetch gameweek: {gw_error}")
            current_gw = "Unknown"
        
        return jsonify({
            'status': 'success',
            'message': 'Data updated successfully',
            'timestamp': datetime.now().isoformat(),
            'updated': {
                'players': 'fetched from FPL API',
                'gameweek': f'Gameweek {current_gw}' if current_gw != "Unknown" else 'updated',
                'features': 'recalculated (10 features)',
                'predictions': 'ready'
            }
        })
        
    except Exception as e:
        print(f"ERROR during data update: {e}\n")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
