import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy import stats as sp_stats
from typing import List, Dict, Optional
import time
import random
import os  

from data_collector import NBADataCollector
from ml_predictor import MLPredictor
import config

class MLValueBetFinder:
    """
    Sistema Value Bet PRO (Versione Selective):
    - XGBoost ML
    - Filtro Garbage Time (Spread)
    - Money Management (Kelly Criterion)
    - Buffer Sicurezza Fisso 10%
    - Force Retrain Giornaliero
    """
    
    def __init__(self, nba_api_key: str, odds_api_key: str):
        self.data_collector = NBADataCollector(nba_api_key)
        self.ml_predictor = MLPredictor()
        self.odds_api_key = odds_api_key
        self.game_spreads = {} # Cache per gli spread delle partite
        
        # --- MODIFICA: FORZA IL RI-ADDESTRAMENTO GIORNALIERO ---
        # Definiamo il percorso del file pickle
        model_path = 'nba_ml_models.pkl'
        
        # Se il file esiste, lo cancelliamo per obbligare il sistema a ri-allenarsi
        # con i dati freschi della notte scorsa.
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                print("🧹 Cancellati vecchi modelli: Forzo ri-addestramento giornaliero.")
            except OSError:
                pass

        # Proviamo a caricare i modelli (fallirà se abbiamo appena cancellato il file)
        if not self.ml_predictor.load_models():
            print("🚀 Avvio training modelli con i dati aggiornati a IERI NOTTE (verrà eseguito durante l'analisi)...")
    
    def get_game_ids(self) -> List[str]:
        """Ottiene ID partite e SALVA GLI SPREAD per filtro Garbage Time"""
        url = f"{config.ODDS_BASE_URL}/basketball_nba/odds"
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us,eu',
            'markets': 'h2h,spreads', # Richiediamo anche gli spread
            'oddsFormat': 'decimal'
        }
        try:
            print(f"📡 Recupero lista partite e spread NBA...")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            game_ids = []
            now_utc = datetime.now(timezone.utc)
            cutoff_time = now_utc + timedelta(hours=24)
            
            self.game_spreads = {} # Reset
            
            for game in data:
                commence_time_str = game['commence_time']
                try:
                    commence_time = datetime.fromisoformat(commence_time_str.replace('Z', '+00:00'))
                    # Consideriamo solo partite nelle prossime 24h
                    if now_utc <= commence_time <= cutoff_time:
                        game_id = game['id']
                        game_ids.append(game_id)
                        
                        # --- LOGICA ESTRAZIONE SPREAD ---
                        best_spread = 0.0
                        count = 0
                        for bookie in game.get('bookmakers', []):
                            for market in bookie.get('markets', []):
                                if market['key'] == 'spreads':
                                    for outcome in market['outcomes']:
                                        # Prendiamo il valore assoluto (es. -15.5 diventa 15.5)
                                        if 'point' in outcome:
                                            best_spread += abs(float(outcome['point']))
                                            count += 1
                            if count > 0: break 
                        
                        if count > 0:
                            self.game_spreads[game_id] = best_spread / count
                        else:
                            self.game_spreads[game_id] = 0.0
                            
                except ValueError:
                    continue

            print(f"   ✓ Trovate {len(game_ids)} partite. Spread analizzati.")
            return game_ids
            
        except Exception as e:
            print(f"⚠️ Errore recupero lista partite: {e}")
            return []

    def get_props_for_game(self, game_id: str) -> List[Dict]:
        """Ottiene props per singola partita e passa il game_id"""
        url = f"{config.ODDS_BASE_URL}/basketball_nba/events/{game_id}/odds"
        params = {
            'apiKey': self.odds_api_key,
            'regions': 'us,eu',
            'markets': 'player_points,player_rebounds,player_assists', 
            'oddsFormat': 'decimal'
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 422: return []
            response.raise_for_status()
            data = response.json()
            # Passiamo game_id per mantenere il riferimento
            return self._parse_odds([data], game_id)
        except Exception:
            return []

    def get_live_odds(self) -> List[Dict]:
        all_props = []
        game_ids = self.get_game_ids()
        
        if not game_ids: return []
            
        print(f"🔄 Inizio scaricamento Player Props per {len(game_ids)} partite...")
        
        for i, game_id in enumerate(game_ids):
            props = self.get_props_for_game(game_id)
            if props:
                all_props.extend(props)
                print(f"   ✓ Partita {i+1}/{len(game_ids)}: Trovate {len(props)} quote.")
            else:
                print(f"   x Partita {i+1}/{len(game_ids)}: Nessuna quota giocatori.")
            time.sleep(1.0) 
            
        return all_props
    
    def _parse_odds(self, data: List[Dict], game_id: str = None) -> List[Dict]:
        props = []
        market_map = {'player_points': 'points', 'player_rebounds': 'rebounds', 'player_assists': 'assists'}
        
        for event in data:
            home_team = event.get('home_team')
            away_team = event.get('away_team')
            for bookmaker in event.get('bookmakers', []):
                bookie_name = bookmaker.get('title')
                for market in bookmaker.get('markets', []):
                    stat_type = market_map.get(market.get('key'))
                    if not stat_type: continue
                    for outcome in market.get('outcomes', []):
                        bet_type = outcome.get('name') 
                        if bet_type not in ['Over', 'Under']: continue
                        player_name = outcome.get('description')
                        line = outcome.get('point')
                        odds = outcome.get('price')
                        if player_name and line and odds:
                            odds_val = float(odds)
                            if not (1.7 <= odds_val <= 2.10): continue
                            props.append({
                                'game_id': game_id, # Importante per il filtro spread
                                'player': player_name,
                                'stat': stat_type,
                                'line': float(line),
                                'odds': odds_val,
                                'bookmaker': bookie_name,
                                'matchup': f"{away_team} @ {home_team}",
                                'bet_type': bet_type
                            })
        return props

    def _calculate_kelly_stake(self, bankroll: float, odds: float, true_prob: float) -> float:
        """Calcola la puntata ideale (Kelly Frazionato 1/4)"""
        if true_prob <= 0 or odds <= 1: return 0.0
        
        decimal_odds = odds - 1
        # Formula di Kelly: f* = (bp - q) / b
        kelly_pct = (decimal_odds * true_prob - (1 - true_prob)) / decimal_odds
        
        # Kelly 1/4 per sicurezza (riduce volatilità)
        safe_kelly = kelly_pct / 4
        
        if safe_kelly <= 0: return 0.0
        
        # Cap massimo al 5% del bankroll per singola bet
        if safe_kelly > 0.05: safe_kelly = 0.05
        
        stake = bankroll * safe_kelly
        return round(stake, 2)
    
    def analyze_with_ml(self, bet_info: Dict) -> Optional[Dict]:
        """Analizza con XGBoost, Filtro Spread e Kelly"""
        player_name = bet_info['player']
        stat_type = bet_info['stat']
        line = bet_info['line']
        odds = bet_info['odds']
        bet_type = bet_info['bet_type']
        game_id = bet_info.get('game_id') 
        
        # --- FILTRO GARBAGE TIME (SPREAD) ---
        spread = self.game_spreads.get(game_id, 0.0)
        # Se lo spread previsto è > 13.5 punti, alto rischio blowout
        if spread > 13.5 and bet_type == 'Over':
            return None # Scartiamo Over in partite a senso unico
        
        player_data = self.data_collector.get_player_game_logs_extended(player_name, n_games=85)
        
        if player_data.empty or len(player_data) < 10: return None
        
        ml_result = self.ml_predictor.predict(player_data, stat_type, player_name)
        if not ml_result: return None
        
        prediction = ml_result['prediction']
        
        # --- FILTRO BUFFER DI SICUREZZA (FISSO 10%) ---
        buffer_pct = 0.10 
            
        diff_pct = (prediction - line) / line
        
        if bet_type == 'Over':
            if diff_pct < buffer_pct: return None
        elif bet_type == 'Under':
            if diff_pct > -buffer_pct: return None

        # --- FILTRO COERENZA MODELLI (RF vs XGB) ---
        rf_pred = ml_result['rf_pred']
        gb_pred = ml_result['gb_pred']
        if abs(rf_pred - gb_pred) > (prediction * 0.25): 
            return None 

        # Calcolo Probabilità e EV
        player_std = player_data[stat_type].std()
        if pd.isna(player_std) or player_std < 1.0: player_std = 1.0
        
        model_uncertainty = ml_result['uncertainty']
        final_uncertainty = (model_uncertainty * 0.7) + (player_std * 0.3)
        if final_uncertainty <= 0: final_uncertainty = 1.0
        
        z_score = (prediction - line) / final_uncertainty
        prob_over = sp_stats.norm.cdf(z_score)   
        prob_under = 1 - prob_over               
        
        true_prob = prob_over if bet_type == 'Over' else prob_under

        if bet_type == 'Under' and prediction > line: return None 
        if bet_type == 'Over' and prediction < line: return None

        implied_prob = 1 / odds
        edge = true_prob - implied_prob
        ev = (true_prob * (odds - 1)) - (1 - true_prob)

        # Calcolo Stake consigliata su 1000€
        stake_1k = self._calculate_kelly_stake(1000, odds, true_prob)

        return {
            **bet_info,
            'ml_prediction': round(prediction, 1),
            'prob_real': round(true_prob * 100, 1), 
            'edge': round(edge * 100, 1),
            'ev': round(ev, 3),
            'confidence_score': ml_result['confidence'],
            'diff_pct': round(diff_pct * 100, 1),
            'spread_game': round(spread, 1),
            'rec_stake_1k': stake_1k
        }
    
    def run_ml_analysis(self) -> pd.DataFrame:
        print("\n" + "="*80)
        print("🤖 NBA PRO BET FINDER (XGBoost + Kelly + Selective 10%)")
        print("="*80 + "\n")
        
        odds_data = self.get_live_odds()
        if not odds_data:
            print("\n❌ Nessuna quota trovata.")
            return pd.DataFrame()
        
        print(f"\n🧠 Avvio Analisi su {len(odds_data)} quote...")
        print("   (Filtri attivi: Conf > 0.7, Buffer > 10%, No Blowouts)")
        
        results = []
        random.shuffle(odds_data) 
        
        for i, bet in enumerate(odds_data):
            print(f"   🔎 Analisi ({i+1}/{len(odds_data)}): {bet['player']}...", end="\r")
            try:
                analysis = self.analyze_with_ml(bet)
                if analysis:
                    if analysis['ev'] > 0 and analysis['confidence_score'] > 0.75:
                        results.append(analysis)
            except Exception: pass
            
        print(f"\n\n   ✅ Analisi completata.")
        
        if not results:
            print("\n⚠️ Nessuna scommessa trovata con i criteri restrittivi.")
            return pd.DataFrame()
            
        df = pd.DataFrame(results)
        df = df.sort_values('ev', ascending=False)
        df = df.drop_duplicates(subset=['player', 'stat'], keep='first')
        
        self._print_ml_report(df)
        
        filename = f"bets_nba_PRO_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"\n📊 Report Excel salvato: {filename}")
        
        return df
    
    def _print_ml_report(self, df: pd.DataFrame):
        print(f"\n{'='*80}")
        print(f"💰 TOP BETS CONSIGLIATE (Kelly Stake su bankroll 1000€)")
        print(f"{'='*80}")
        for i, row in df.head(10).iterrows():
            type_icon = "⬆️" if row['bet_type'] == 'Over' else "⬇️"
            spread_info = f"(Spread: {row['spread_game']})" if row['spread_game'] > 0 else ""
            
            print(f"\n💎 {row['player']} ({row['matchup']}) {spread_info}")
            print(f"   {type_icon} {row['bet_type'].upper()} {row['stat'].upper()} {row['line']} @ {row['odds']}")
            print(f"   🤖 XGBoost: {row['ml_prediction']} | Prob: {row['prob_real']}%")
            print(f"   💵 PUNTATA CONSIGLIATA: {row['rec_stake_1k']}€")