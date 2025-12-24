import pandas as pd
import numpy as np
import time
import unicodedata
import difflib
from datetime import datetime

# Importazioni ufficiali nba_api
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonallplayers, leaguedashteamstats
from requests.exceptions import ReadTimeout

class NBADataCollector:
    """
    Raccolta dati NBA Potenziata.
    Stagione aggiornata: 2025-26.
    Include gestione avanzata dei nomi, lista dinamica e 
    ANALISI DIFENSIVA AVVERSARIA (DefRating + Pace).
    """
    
    def __init__(self, api_key: str = None):
        self.cache = {}
        self.player_id_cache = {}
        self.active_players_df = None 
        self.team_stats_cache = None
        self.teams_map = self._get_teams_map() # Mappa ABBR -> ID
        print("   ✅ Inizializzato NBADataCollector (Stagione 2025-26)")
        
    def _get_teams_map(self):
        """Crea mappa per convertire Abbreviazioni (es. LAL) in ID"""
        try:
            nba_teams = teams.get_teams()
            return {t['abbreviation']: t['id'] for t in nba_teams}
        except Exception:
            return {}

    def _get_team_stats(self, season='2025-26'): 
        """Scarica statistiche difensive (con fallback robusto e parametro corretto)"""
        if self.team_stats_cache is not None:
            return self.team_stats_cache
            
        try:
            print(f"   🛡️ Scaricamento statistiche difensive squadre ({season})...")
            stats = pd.DataFrame()
            
            # TENTATIVO 1: Richiedi statistiche AVANZATE con il parametro corretto
            try:
                stats = leaguedashteamstats.LeagueDashTeamStats(
                    season=season,
                    measure_type_detailed_defense='Advanced', 
                    timeout=30
                ).get_data_frames()[0]
            except Exception as e:
                # Fallback silenzioso su statistiche base se fallisce Advanced
                try:
                    stats = leaguedashteamstats.LeagueDashTeamStats(
                        season=season,
                        timeout=30
                    ).get_data_frames()[0]
                except Exception:
                    pass

            # Se il download è fallito del tutto
            if stats.empty:
                 print("   ⚠️ Impossibile scaricare stats squadre. Uso valori medi.")
                 return pd.DataFrame()

            # Normalizza colonne in maiuscolo
            stats.columns = [c.upper() for c in stats.columns]
            
            # Se mancano colonne Advanced (perché siamo finiti sul fallback Base), le calcoliamo/stimiamo
            if 'DEF_RATING' not in stats.columns:
                stats['DEF_RATING'] = 112.0 # Media lega stimata 2025
            
            if 'PACE' not in stats.columns:
                stats['PACE'] = 99.0 

            clean_stats = stats[['TEAM_ID', 'TEAM_NAME', 'DEF_RATING', 'PACE']].copy()
            clean_stats.columns = ['opp_team_id', 'opp_name', 'opp_def_rating', 'opp_pace']
            
            self.team_stats_cache = clean_stats
            print("   ✅ Statistiche difensive pronte.")
            return clean_stats
            
        except Exception as e:
            print(f"⚠️ Errore critico team stats: {e}")
            return pd.DataFrame()

    def _normalize_name(self, name: str) -> str:
        """Normalizza stringhe per confronto (rimuove accenti, punti, case)"""
        if not name: return ""
        n = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
        return n.replace('.', '').replace("'", "").lower().strip()

    def _get_active_players_dynamic(self):
        """Scarica la lista aggiornata dei giocatori attivi"""
        if self.active_players_df is not None:
            return self.active_players_df
            
        try:
            # print("   ↻ Scaricamento lista giocatori aggiornata da NBA.com...")
            resp = commonallplayers.CommonAllPlayers(is_only_current_season=1, timeout=10)
            self.active_players_df = resp.get_data_frames()[0]
            return self.active_players_df
        except Exception as e:
            return pd.DataFrame()

    def get_player_id(self, player_name: str) -> int:
        """Trova l'ID giocatore con strategia a 3 livelli"""
        if player_name in self.player_id_cache:
            return self.player_id_cache[player_name]
            
        search_name = self._normalize_name(player_name)
        
        # 1. Lista statica
        nba_players = players.get_players()
        for p in nba_players:
            if self._normalize_name(p['full_name']) == search_name:
                self.player_id_cache[player_name] = p['id']
                return p['id']
        
        # 2. Lista dinamica
        active_df = self._get_active_players_dynamic()
        if not active_df.empty:
            found = active_df[active_df['DISPLAY_FIRST_LAST'].apply(self._normalize_name) == search_name]
            if not found.empty:
                pid = found.iloc[0]['PERSON_ID']
                self.player_id_cache[player_name] = pid
                return pid

        # 3. Fuzzy Match
        all_names = {self._normalize_name(p['full_name']): p['id'] for p in nba_players}
        if not active_df.empty:
            for _, row in active_df.iterrows():
                all_names[self._normalize_name(row['DISPLAY_FIRST_LAST'])] = row['PERSON_ID']
        
        matches = difflib.get_close_matches(search_name, all_names.keys(), n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            pid = all_names[best_match]
            print(f"   ✓ Match Fuzzy trovato: '{player_name}' -> ID {pid} (match con '{best_match}')")
            self.player_id_cache[player_name] = pid
            return pid
            
        return None

    def get_player_game_logs_extended(self, player_name: str, 
                                     season: str = "2025-26", 
                                     n_games: int = 30) -> pd.DataFrame:
        """Recupera i game logs e calcola le features + STATS AVVERSARIO"""
        cache_key = f"{player_name}_{season}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        player_id = self.get_player_id(player_name)
        if not player_id:
            return pd.DataFrame()
        
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season,
                timeout=30
            )
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.rename(columns={
                'GAME_DATE': 'date', 'MATCHUP': 'matchup', 'WL': 'result', 'MIN': 'minutes',
                'PTS': 'points', 'REB': 'rebounds', 'AST': 'assists', 'STL': 'steals',
                'BLK': 'blocks', 'TOV': 'turnovers', 'FG3M': 'threes', 'FGM': 'fgm',
                'FGA': 'fga', 'FG3A': 'fg3a', 'FTM': 'ftm', 'FTA': 'fta', 'PLUS_MINUS': 'plus_minus'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            cols_numeric = ['points', 'rebounds', 'assists', 'threes', 'minutes', 'plus_minus', 'fga', 'fgm', 'fta']
            for col in cols_numeric:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['is_home'] = df['matchup'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
            
            # --- INTEGRAZIONE STATS AVVERSARIO ---
            team_stats = self._get_team_stats(season)
            
            def extract_opponent(matchup):
                try:
                    clean = matchup.replace(' vs. ', ' ').replace(' @ ', ' ')
                    parts = clean.split(' ')
                    return parts[1] if len(parts) > 1 else None
                except: return None

            df['opp_abbr'] = df['matchup'].apply(extract_opponent)
            
            # Merge sicuro: se team_stats è vuoto o mancano ID, non crasha
            if not team_stats.empty and self.teams_map:
                df['opp_id'] = df['opp_abbr'].map(self.teams_map)
                df = df.merge(team_stats, left_on='opp_id', right_on='opp_team_id', how='left')
                
                # Riempie buchi (es. squadre non trovate) con default
                df['opp_def_rating'] = df['opp_def_rating'].fillna(112.0) 
                df['opp_pace'] = df['opp_pace'].fillna(99.0)
            else:
                # Fallback totale se API stats fallisce
                df['opp_def_rating'] = 112.0
                df['opp_pace'] = 99.0

            # --- ORDINAMENTO E FEATURES ---
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            df = self._engineer_features(df)
            df = df.sort_values('date', ascending=False).reset_index(drop=True).head(n_games)
            
            self.cache[cache_key] = df
            time.sleep(0.6) 
            return df
            
        except Exception as e:
            return pd.DataFrame()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering per ML (Include Impact Score Difensivo)"""
        if df.empty: return df
        
        for stat in ['points', 'rebounds', 'assists', 'threes']:
            if stat not in df.columns: continue
            
            df[f'{stat}_ma3'] = df[stat].rolling(window=3, min_periods=1).mean()
            df[f'{stat}_ma5'] = df[stat].rolling(window=5, min_periods=1).mean()
            df[f'{stat}_ma10'] = df[stat].rolling(window=10, min_periods=1).mean()
            df[f'{stat}_std5'] = df[stat].rolling(window=5, min_periods=1).std().fillna(0)
            
            if f'{stat}_ma10' in df.columns:
                df[f'{stat}_trend'] = df[f'{stat}_ma3'] - df[f'{stat}_ma10']
            else:
                df[f'{stat}_trend'] = 0
                
            # Impact Score (Difesa/Pace)
            if 'opp_def_rating' in df.columns:
                df['defense_factor'] = (df['opp_def_rating'] - 112) / 10.0
                df['pace_factor'] = (df['opp_pace'] - 99) / 5.0
                df[f'{stat}_proj'] = df[f'{stat}_ma5'] * (1 + (df['defense_factor'] * 0.05) + (df['pace_factor'] * 0.05))
            else:
                df[f'{stat}_proj'] = df[f'{stat}_ma5']
        
        df['days_rest'] = df['date'].diff().dt.days.fillna(3) 
        df['is_back_to_back'] = (df['days_rest'] <= 1).astype(int)
        
        if 'fga' in df.columns:
            df['fg_pct'] = np.where(df['fga'] > 0, df['fgm'] / df['fga'], 0)
            df['usage_proxy'] = df['minutes'] * (df['fga'] + df.get('fta',0) * 0.44 + df.get('assists',0) * 0.33)
        else:
            df['fg_pct'] = 0
            df['usage_proxy'] = 0
            
        return df.fillna(0)