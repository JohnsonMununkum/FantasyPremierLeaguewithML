# app.py
# Flask API for FPL Intelligence System
# Provides endpoints for predictions and team optimization
import os
from flask import Flask, jsonify
import pandas as pd
import sqlite3
from utils.predictor import FPLPredictor
from utils.optimizer import FPLOptimizer

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

# API Endpoints
@app.route('/')
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
            'optimize_multiple': '/api/optimize/multiple'
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
            ['name', 'position', 'team', 'price', 'predicted_points']
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
    """
    Get bottom 10 predicted players for next gameweek.
    
    Returns players with lowest predicted points - these are the
    worst performers to avoid or consider transferring out.
    """
    try:
        # Loads the latest data and generates predictions
        df = load_latest_data()
        df['predicted_points'] = predictor.predict_points(df)
        
        # Gets the bottom 10 by predicted points
        bottom_10 = df.nsmallest(10, 'predicted_points')[
            ['name', 'position', 'team', 'price', 'predicted_points']
        ].to_dict('records')
        
        return jsonify({
            'status': 'success',
            'type': 'bottom',
            'count': len(bottom_10),
            'description': 'Players with lowest predicted points - consider avoiding',
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
            ['name', 'position', 'team', 'price', 'predicted_points']
        ].to_dict('records')
        
        bottom_10 = df.nsmallest(10, 'predicted_points')[
            ['name', 'position', 'team', 'price', 'predicted_points']
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
