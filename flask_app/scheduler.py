# scheduler.py
# Background worker that runs 24/7 on Render
# Gets the FPL data every 6 hours to keep predictions current
import schedule
import time
from datetime import datetime
from utils.data_fetcher import FPLDataFetcher

# Scheduled task to fetch latest FPL data and update predictions
# Runs every 6 hours to ensure predictions stay current as player prices, availability, form, and fixtures change
def fetch_and_update():
    print(f"\n{'='*80}")
    print(f"Scheduler run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*80)
    
    try:
        # Initializes the data fetcher
        fetcher = FPLDataFetcher()
        
        # Gets all the player data from FPL API
        fetcher.fetch_all_players()
        
        # Updates the gameweek and fixture data
        fetcher.fetch_gameweek_data()
        
        # Calculates all the engineered features again based on the latest data
        fetcher.update_features()
        
        print(f"{'='*80}")
        print(f"Success, Next run in 6 hours")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"Error in scheduler run: {e}\n")

# Scheduled the task to run every 6 hours
schedule.every(6).hours.do(fetch_and_update)

print("FPL Intelligence Scheduler Starting...")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Schedule: Every 6 hours")
print("Running initial data fetch\n")

# Runs immediately on startup dosent wait for the first 6 hour to run
fetch_and_update()

# While true keeps the scheduler running forever
# Checks every 60 seconds if it's time to run scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(60)
