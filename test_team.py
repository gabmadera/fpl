#!/usr/bin/env python3

import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Now import with relative imports working
from src.team_selector import TeamSelector

def main():
    print("Testing team selection...")
    
    # Load predictions
    df = pd.read_csv('data/processed/predictions_current.csv')
    print(f'Loaded {len(df)} players')
    
    # Debug: check forward distribution
    fwd_count = len(df[df['position'] == 'FWD'])
    fwd_available = len(df[(df['position'] == 'FWD') & (~df['fpl_status'].isin(['i', 's']))])
    fwd_affordable = len(df[(df['position'] == 'FWD') & (df['price'] <= 10.0)])
    print(f'FWD players: {fwd_count} total, {fwd_available} available, {fwd_affordable} under £10m')
    
    # Test team selection  
    selector = TeamSelector()
    print('Initialized team selector')
    
    try:
        team = selector.select_optimal_team(df, gameweek=1)
        print(f'Team selected successfully!')
        print(f'Formation: {team.formation}')
        print(f'Total Cost: £{team.total_cost:.1f}m')
        print(f'Budget Remaining: £{100.0 - team.total_cost:.1f}m')
        print(f'Predicted Points: {team.predicted_points:.1f}')
        print(f'Chip Recommendation: {team.chip_recommendation or "None"}')
        
        # Debug formation requirements
        def_count, mid_count, fwd_count = map(int, team.formation.split('-'))
        total_starters_needed = 1 + def_count + mid_count + fwd_count  # 1 GKP + formation
        print(f'Formation {team.formation} needs {total_starters_needed} starters (1 GKP + {def_count} DEF + {mid_count} MID + {fwd_count} FWD)')
        
        # Check position counts in starting XI
        starter_positions = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in team.starters:
            starter_positions[player['position']] += 1
        print(f'Actual starters by position: {starter_positions}')
        
        print(f'\nStarting XI ({len(team.starters)} players):')
        for player in team.starters:
            captain_mark = ' (C)' if player['player_id'] == team.captain_id else ''
            vice_mark = ' (VC)' if player['player_id'] == team.vice_captain_id else ''
            print(f"  {player['position']:3} {player['name']:20} £{player['price']:4.1f}m {player['predicted_points']:5.1f}pts{captain_mark}{vice_mark}")
        
        print(f'\nBench ({len(team.bench)} players):')
        bench_positions = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for player in team.bench:
            bench_positions[player['position']] += 1
            print(f"  {player['position']:3} {player['name']:20} £{player['price']:4.1f}m {player['predicted_points']:5.1f}pts")
        print(f'Bench positions: {bench_positions}')
        
        # Total check
        total_positions = {pos: starter_positions[pos] + bench_positions[pos] for pos in starter_positions}
        print(f'Total squad positions: {total_positions}')
        total_players = sum(total_positions.values())
        print(f'Total squad size: {total_players} (should be 15)')
        
        # Test all formations
        print(f'\n=== Testing All FPL Formations ===')
        for formation in selector.formations:
            def_count, mid_count, fwd_count = map(int, formation.split('-'))
            total = 1 + def_count + mid_count + fwd_count
            valid = (def_count >= 3 and fwd_count >= 1 and total == 11)
            status = "✅ VALID" if valid else "❌ INVALID"
            print(f'{formation}: 1 GKP + {def_count} DEF + {mid_count} MID + {fwd_count} FWD = {total} total {status}')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()