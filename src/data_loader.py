import requests
import pandas as pd
import time
from utils import logger

class FPLDataLoader:
    """Load data from Fantasy Premier League API"""
    
    BASE_URL = "https://fantasy.premierleague.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        
    def fetch_bootstrap_data(self):
        """Fetch main FPL data (players, teams, gameweeks)"""
        url = f"{self.BASE_URL}/bootstrap-static/"
        logger.info(f"Fetching from: {url}")
        
        response = self.session.get(url)
        response.raise_for_status()
        
        data = response.json()
        logger.info("✓ Successfully fetched bootstrap data")
        
        return data
    
    def parse_players(self, bootstrap_data):
        """Parse player data from bootstrap"""
        players_df = pd.DataFrame(bootstrap_data['elements'])
        logger.info(f"✓ Found {len(players_df)} players")
        return players_df
    
    def parse_teams(self, bootstrap_data):
        """Parse team data from bootstrap"""
        teams_df = pd.DataFrame(bootstrap_data['teams'])
        logger.info(f"✓ Found {len(teams_df)} teams")
        return teams_df
    
    def fetch_player_history(self, player_id):
        """Fetch historical gameweek data for a specific player"""
        url = f"{self.BASE_URL}/element-summary/{player_id}/"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Get gameweek history
            history = data.get('history', [])
            return pd.DataFrame(history)
            
        except Exception as e:
            logger.warning(f"Could not fetch history for player {player_id}: {e}")
            return pd.DataFrame()
    
    def load_all_player_history(self, players_df, max_players=None):
        """
        Load gameweek history for all players
        
        Args:
            players_df: DataFrame of players
            max_players: Maximum number of players to load (None = all)
        """
        all_history = []
        
        # Limit players if specified
        if max_players is not None:
            players_to_load = players_df.head(max_players)
        else:
            players_to_load = players_df
        
        total_players = len(players_to_load)
        logger.info(f"Progress: 0/{total_players} players loaded")
        
        for idx, (_, player) in enumerate(players_to_load.iterrows()):
            player_id = player['id']
            player_name = player['web_name']
            
            # Fetch history
            history_df = self.fetch_player_history(player_id)
            
            if not history_df.empty:
                # Add player metadata
                history_df['player_id'] = player_id
                history_df['player_name'] = player_name
                history_df['element_type'] = player['element_type']
                
                all_history.append(history_df)
            
            # Progress update every 50 players
            if (idx + 1) % 50 == 0:
                logger.info(f"Progress: {idx+1}/{total_players} players loaded")
            
            # Rate limiting (be nice to FPL servers)
            time.sleep(0.1)
        
        # Combine all history
        if all_history:
            combined_df = pd.concat(all_history, ignore_index=True)
            logger.info(f"✓ Loaded {len(combined_df)} gameweek records")
            return combined_df
        else:
            logger.warning("No history data loaded")
            return pd.DataFrame()


def main():
    """Main data loading function"""
    logger.info("\n=== Loading FPL Data ===")
    
    # Initialize loader
    loader = FPLDataLoader()
    
    # Fetch bootstrap data
    bootstrap = loader.fetch_bootstrap_data()
    
    # Parse players and teams
    players = loader.parse_players(bootstrap)
    teams = loader.parse_teams(bootstrap)
    
    # Load history (ALL 755 players - this will take 1-1.5 hours!)
    logger.info("Loading player histories (this will take 1-1.5 hours)...")
    history_df = loader.load_all_player_history(players, max_players=None)
    
    return {
        'players': players,
        'teams': teams,
        'history': history_df
    }

if __name__ == '__main__':
    data = main()

