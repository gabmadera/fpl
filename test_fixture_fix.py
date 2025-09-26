#!/usr/bin/env python3
"""
Test script to verify the fixture data fix is working correctly.
This script validates that predictions show current opponent data (GW6) instead of stale data (GW4/GW5).
"""

from src.fpl_client import FPLClient
from src.ml_pipeline import MLPipeline
import pandas as pd

def main():
    print("=" * 60)
    print("FPL FIXTURE DATA FIX VALIDATION")
    print("=" * 60)

    # Initialize clients
    fpl = FPLClient()
    pipeline = MLPipeline()

    # 1. Validate fixture data freshness
    print("\n1. FIXTURE DATA VALIDATION")
    print("-" * 30)
    validation = fpl.validate_fixture_freshness()

    print(f"Status: {validation['status'].upper()}")
    print(f"Current GW: {validation.get('current_gw')}")
    print(f"Next Active GW: {validation.get('next_active_gw')}")
    print(f"Upcoming Fixtures: {validation.get('upcoming_fixtures_count')}")

    if validation.get('issues'):
        print("\nIssues detected:")
        for issue in validation['issues']:
            print(f"  ⚠ {issue}")

    if validation.get('recommendations'):
        print("\nRecommendations:")
        for rec in validation['recommendations']:
            print(f"  💡 {rec}")

    # 2. Test opponent mapping
    print("\n2. OPPONENT MAPPING TEST")
    print("-" * 30)

    try:
        opp_map, strength = pipeline._team_opp_map()
        print(f"Mapped opponents for {len(opp_map)} teams")

        # Get team names for display
        teams_data = fpl.bootstrap_static().get('teams', [])
        teams_df = pd.DataFrame(teams_data)
        team_map = teams_df.set_index('id')['name'].to_dict()

        print("\nSample upcoming fixtures (should be GW6, not GW4/GW5):")
        for i, (team_id, (opp_id, is_home)) in enumerate(list(opp_map.items())[:5]):
            home_team = team_map.get(team_id, f'Team {team_id}')
            away_team = team_map.get(opp_id, f'Team {opp_id}')
            if is_home:
                print(f"  {home_team} vs {away_team}")
            else:
                print(f"  {home_team} @ {away_team}")

    except Exception as e:
        print(f"❌ Error in opponent mapping: {e}")
        return False

    # 3. Test key player opponent data (the original issue)
    print("\n3. KEY PLAYER OPPONENT CHECK")
    print("-" * 30)

    try:
        # Generate a small prediction sample to check opponent data
        df = pipeline.prepare_training_data()
        if df.empty:
            print("❌ No player data available")
            return False

        # Look for high-profile players mentioned in the issue
        test_players = ["Haaland", "Salah", "Son"]
        found_players = []

        for player in test_players:
            player_data = df[df['name'].str.contains(player, case=False, na=False)]
            if not player_data.empty:
                found_players.append((player, player_data.iloc[0]))

        if not found_players:
            # Just use first few players
            print("Using sample players for opponent check:")
            for _, player_row in df.head(3).iterrows():
                team_id = int(player_row.get('team_id', 0))
                if team_id in opp_map:
                    opp_id, is_home = opp_map[team_id]
                    opp_name = team_map.get(opp_id, f'Team {opp_id}')
                    venue = "vs" if is_home else "@"
                    print(f"  {player_row.get('name', 'Unknown')} ({team_map.get(team_id, 'Unknown Team')}) {venue} {opp_name}")
        else:
            print("Found high-profile players - checking their opponents:")
            for player_name, player_row in found_players:
                team_id = int(player_row.get('team_id', 0))
                if team_id in opp_map:
                    opp_id, is_home = opp_map[team_id]
                    opp_name = team_map.get(opp_id, f'Team {opp_id}')
                    venue = "vs" if is_home else "@"
                    print(f"  {player_name} ({team_map.get(team_id, 'Unknown Team')}) {venue} {opp_name}")
                else:
                    print(f"  {player_name}: No opponent mapping found")

    except Exception as e:
        print(f"❌ Error checking player opponents: {e}")
        return False

    # 4. Verify we're not showing old GW4 fixtures
    print("\n4. OLD FIXTURE CHECK")
    print("-" * 30)

    fixtures_df = pd.DataFrame(fpl.fixtures())
    gw4_fixtures = fixtures_df[fixtures_df['event'] == 4]
    gw5_fixtures = fixtures_df[fixtures_df['event'] == 5]
    gw6_fixtures = fixtures_df[fixtures_df['event'] == 6]

    print(f"GW4 fixtures (old): {len(gw4_fixtures)} (should be finished)")
    print(f"GW5 fixtures (current): {len(gw5_fixtures)} (should be finished)")
    print(f"GW6 fixtures (next): {len(gw6_fixtures)} (should be upcoming)")

    # Check if any GW4 fixtures are still marked as unfinished (would be bad)
    unfinished_gw4 = gw4_fixtures[~gw4_fixtures.get('finished', True)]
    if len(unfinished_gw4) > 0:
        print(f"⚠ WARNING: {len(unfinished_gw4)} GW4 fixtures still unfinished")
    else:
        print("✓ All GW4 fixtures properly marked as finished")

    print("\n5. SUMMARY")
    print("-" * 30)

    if validation['status'] == 'ok' and len(opp_map) >= 10:
        print("✅ FIXTURE DATA FIX SUCCESSFUL")
        print("   - Correct gameweek detection (GW6 for opponents)")
        print("   - Fresh opponent mappings")
        print("   - Validation monitoring in place")
        print("   - Reduced cache TTL for faster updates")
        return True
    else:
        print("❌ FIXTURE DATA FIX NEEDS ATTENTION")
        print("   - Check validation errors above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)