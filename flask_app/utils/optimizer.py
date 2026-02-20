# FPLOptimizer using Linear Programming (PuLP) to select optimal FPL squads
# Uses the PuLP library to solve the optimization problem
from pulp import *

class FPLOptimizer:
    # Initializes the optimizer with the necessary data and parameters
    # Generates a single 15-player squad using Linear Programming
    def optimize_squad(self, players_df, budget=100.0):
        # Extracts data from the dataframe
        player_ids = players_df.index.tolist()
        predicted_points = players_df['predicted_points'].to_dict()
        prices = players_df['price'].to_dict()
        positions = players_df['position'].to_dict()
        names = players_df['name'].to_dict()
        teams = players_df['team'].to_dict() if 'team' in players_df.columns else {}
        
        # Puts the players into groups by position for constraints
        gk_ids = [i for i in player_ids if positions[i] == 'GK']
        def_ids = [i for i in player_ids if positions[i] == 'DEF']
        mid_ids = [i for i in player_ids if positions[i] == 'MID']
        fwd_ids = [i for i in player_ids if positions[i] == 'FWD']
        
        # Creates the Linear Programming problem 
        problem = LpProblem("FPL_Squad", LpMaximize)
        
        # 1 if player is selected, 0 if not
        player_vars = LpVariable.dicts("player", player_ids, cat='Binary')
        
        # Maximize the total predicted points
        problem += lpSum([predicted_points[i] * player_vars[i] for i in player_ids])
        
        # Constraints
        # Budget constraint, total cost must not be greater than the budget
        problem += lpSum([prices[i] * player_vars[i] for i in player_ids]) <= budget
        
        # Squad size, Exactly 15 players
        problem += lpSum([player_vars[i] for i in player_ids]) == 15
        
        # Position constraints, 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs
        problem += lpSum([player_vars[i] for i in gk_ids]) == 2
        problem += lpSum([player_vars[i] for i in def_ids]) == 5
        problem += lpSum([player_vars[i] for i in mid_ids]) == 5
        problem += lpSum([player_vars[i] for i in fwd_ids]) == 3
        
        # Team limit constraint, Max of 3 players from same team
        if teams:
            for team in set(teams.values()):
                team_players = [i for i in player_ids if teams.get(i) == team]
                if team_players:
                    problem += lpSum([player_vars[i] for i in team_players]) <= 3
        
        # Solve the optimization problem
        status = problem.solve(PULP_CBC_CMD(msg=0))
        
        # Checking to see if a solution was found
        if status != 1:
            return {'status': 'failed', 'message': 'No optimal solution found'}
        
        # Extracting the selected players
        selected_ids = [i for i in player_ids if player_vars[i].value() == 1]
        
        # Building a squad dictionary organized by position
        squad = {
            'status': 'success',
            'total_cost': sum(prices[i] for i in selected_ids),
            'total_predicted': sum(predicted_points[i] for i in selected_ids),
            'goalkeepers': [],
            'defenders': [],
            'midfielders': [],
            'forwards': []
        }
        
        # Organizing players by position
        for pid in selected_ids:
            player_info = {
                'name': names[pid],
                'team': teams.get(pid, 'Unknown'),
                'price': prices[pid],
                'predicted_points': predicted_points[pid]
            }
            
            if positions[pid] == 'GK':
                squad['goalkeepers'].append(player_info)
            elif positions[pid] == 'DEF':
                squad['defenders'].append(player_info)
            elif positions[pid] == 'MID':
                squad['midfielders'].append(player_info)
            elif positions[pid] == 'FWD':
                squad['forwards'].append(player_info)
        
        # Sorting each position by predicted points in descending order
        for key in ['defenders', 'midfielders', 'forwards']:
            squad[key].sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Player with the highest predicted points in the squad is chosen as captain
        all_players = squad['goalkeepers'] + squad['defenders'] + squad['midfielders'] + squad['forwards']
        captain = max(all_players, key=lambda x: x['predicted_points'])
        squad['captain'] = captain['name']
        squad['captain_points'] = captain['predicted_points']
        
        # Calculating the total with captain bonus
        # Captains points are doubled
        squad['total_with_captain'] = squad['total_predicted'] + captain['predicted_points']
        
        return squad
    
    # Generating multiple optimal squads by excluding previously selected players
    # Gives users options to pick from
    def optimize_multiple_squads(self, players_df, num_squads=3, budget=100.0):
        squads = []
        # used to keep track of players already selected in previous squads to ensure different squads
        excluded_players = set()
        
        for i in range(num_squads):
            # Creating a dataframe excluding players from previous squads
            available_df = players_df[~players_df.index.isin(excluded_players)].copy()
            
            # Checking to see if there are enough players left to form another squad
            if len(available_df) < 15:
                break
            
            # Getting an optimal squad from the players left
            squad = self.optimize_squad(available_df, budget=budget)
            
            if squad['status'] == 'success':
                squads.append({
                    'squad_number': i + 1,
                    'squad_name': f'Squad Option {i + 1}',
                    'description': f'Optimal squad #{i + 1} maximizing predicted points',
                    **squad
                })
                
                # Marking all selected players as used
                all_selected = (squad['goalkeepers'] + squad['defenders'] + 
                              squad['midfielders'] + squad['forwards'])
                
                for player in all_selected:
                    player_indices = available_df[available_df['name'] == player['name']].index
                    excluded_players.update(player_indices)
            else:
                # if no more optimal squads can be generated the process is stopped
                break
        
        return squads
