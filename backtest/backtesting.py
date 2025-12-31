import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from scipy import stats as sp_stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# Importazioni moduli locali
from data_collector import NBADataCollector
from ml_predictor import MLPredictor
import config

warnings.filterwarnings('ignore')

class BacktestingEngine:
    """
    Sistema di backtesting robusto con estrazione manuale delle features
    per evitare conflitti di slicing nel MLPredictor.
    """
    
    def __init__(self):
        self.predictor = MLPredictor()
        
    def simulate_bet(self, bet_type: str, prediction: float, actual: float, 
                    line: float, odds: float, stake: float = 100) -> Dict:
        """Simula l'esito della scommessa"""
        if bet_type == 'OVER':
            win = actual > line
        else: # UNDER
            win = actual < line
            
        profit = stake * (odds - 1) if win else -stake
        
        return {
            'result': 'WIN' if win else 'LOSS',
            'profit': profit,
            'stake': stake,
            'prediction': prediction,
            'actual': actual,
            'line': line,
            'odds': odds,
            'type': bet_type
        }

    def _get_features_for_row(self, row: pd.Series, stat_type: str) -> np.ndarray:
        """
        Estrae le features da una singola riga per la predizione.
        Deve corrispondere esattamente alle colonne usate in MLPredictor.
        """
        feature_cols = [
            f'{stat_type}_ma3', f'{stat_type}_ma5', f'{stat_type}_ma10',
            f'{stat_type}_std5', f'{stat_type}_trend',
            'minutes_ma5', 'minutes_ma10', 'is_home', 'is_back_to_back', 'days_rest',
            'fg_pct', 'usage_proxy'
        ]
        
        # Gestione valori mancanti
        values = []
        for col in feature_cols:
            val = row.get(col, 0)
            if pd.isna(val): val = 0
            values.append(val)
            
        return np.array(values).reshape(1, -1)
    
    def backtest_player(self, player_name: str, player_data: pd.DataFrame,
                       stat_type: str, edge_threshold: float = 0.05,
                       min_train_games: int = 25, verbose: bool = False) -> Dict:
        """
        Walk-Forward Backtesting Robusto
        """
        # Verifica dati minimi
        if player_data.empty or len(player_data) < (min_train_games + 5):
            return {'error': 'Dati insufficienti'}
        
        bets = []
        total_games = len(player_data)
        
        # Indice di partenza: andiamo dal passato (high index) al presente (low index)
        start_index = total_games - min_train_games - 1
        
        debug_counter = 0
        
        for i in range(start_index, -1, -1):
            # Test su partita 'i', Train su tutte le successive (che sono storiche)
            test_game = player_data.iloc[i]
            train_data = player_data.iloc[i+1:].copy()
            
            # --- TRAINING ---
            # Riadestra periodicamente (ogni 5 match) per simulare l'evoluzione della stagione
            if i == start_index or i % 5 == 0:
                self.predictor.models = {} # Reset modelli per evitare bias
                metrics = self.predictor.train_model(train_data, stat_type)
                if 'error' in metrics: continue

            # Verifica esistenza modello
            model_key = f"{stat_type}"
            if model_key not in self.predictor.models: continue
                
            # --- PREDIZIONE MANUALE ---
            # Bypassiamo self.predictor.predict per evitare problemi di slicing sulle features
            try:
                models = self.predictor.models[model_key]
                scaler = models['scaler']
                
                # Estrai features dalla riga corrente
                X_raw = self._get_features_for_row(test_game, stat_type)
                X_scaled = scaler.transform(X_raw)
                
                rf_pred = models['rf'].predict(X_scaled)[0]
                gb_pred = models['gb'].predict(X_scaled)[0]
                prediction = (rf_pred + gb_pred) / 2
                
            except Exception as e:
                continue
            
            # --- SIMULAZIONE BOOKMAKER ---
            # Linea basata sulle ultime 10 partite (media semplice)
            # Il nostro modello vince perché usa trend, varianza e fattori esterni (casa/trasferta)
            simulated_line = round(train_data[stat_type].head(10).mean())
            if simulated_line < 0.5: simulated_line = 0.5
            
            # Deviazione standard storica per calcolo probabilità
            player_std = train_data[stat_type].std()
            if np.isnan(player_std) or player_std == 0: player_std = 1.0
            
            # --- CALCOLO EDGE ---
            z_score = (prediction - simulated_line) / player_std
            prob_over = sp_stats.norm.cdf(z_score)
            prob_under = 1 - prob_over
            
            # Implied prob per quota standard 1.90 (-110 american)
            implied_prob = 0.526
            
            edge = 0
            bet_type = None
            
            if (prob_over - implied_prob) >= edge_threshold:
                bet_type = 'OVER'
                edge = prob_over - implied_prob
            elif (prob_under - implied_prob) >= edge_threshold:
                bet_type = 'UNDER'
                edge = prob_under - implied_prob
            
            # Debug (mostra solo i primi 3 match per non intasare la console)
            if verbose and debug_counter < 3 and bet_type:
                print(f"   [DEBUG] {stat_type.upper()} | Pred: {prediction:.1f} | Line: {simulated_line} | "
                      f"ProbOv: {prob_over:.2f} | Edge: {edge:.3f}")
                debug_counter += 1
                
            if bet_type:
                res = self.simulate_bet(bet_type, prediction, test_game[stat_type], simulated_line, 1.90)
                res.update({
                    'player': player_name,
                    'stat': stat_type,
                    'date': test_game['date'],
                    'edge': round(edge, 3)
                })
                bets.append(res)
        
        if not bets:
            return {'error': 'No bets found'}
            
        df_bets = pd.DataFrame(bets)
        wins = len(df_bets[df_bets['result'] == 'WIN'])
        
        return {
            'player': player_name,
            'stat': stat_type,
            'total_bets': len(df_bets),
            'wins': wins,
            'win_rate': round((wins/len(df_bets))*100, 2),
            'roi': round((df_bets['profit'].sum() / df_bets['stake'].sum()) * 100, 2),
            'bets_detail': df_bets
        }

    def run_full_backtest(self, players_data: Dict[str, pd.DataFrame],
                         stat_types: List[str] = ['points'],
                         edge_thresholds: List[float] = [0.05]) -> pd.DataFrame:
        
        print("\n" + "="*80)
        print("📊 BACKTESTING COMPLETO - ANALISI ROBUSTA")
        print("="*80 + "\n")
        
        all_results = []
        
        # Usiamo solo la prima soglia per evitare duplicazioni nel report
        threshold = edge_thresholds[0]
        
        for player, data in players_data.items():
            for stat in stat_types:
                print(f"👉 Analisi: {player} | {stat} (Thr: {threshold})")
                
                res = self.backtest_player(player, data, stat, threshold, verbose=True)
                
                if 'error' not in res:
                    res['edge_threshold'] = threshold
                    all_results.append(res)
                    print(f"   ✅ Trovate {res['total_bets']} scommesse. ROI: {res['roi']}% (Win Rate: {res['win_rate']}%)")
                else:
                    print(f"   ⚠️ {res.get('error')}")
                print("-" * 40)
        
        if not all_results:
            return pd.DataFrame(), []
            
        summary = []
        for r in all_results:
            summary.append({
                'player': r['player'],
                'stat': r['stat'],
                'bets': r['total_bets'],
                'win_rate': r['win_rate'],
                'roi': r['roi']
            })
            
        return pd.DataFrame(summary), all_results

    def plot_results(self, summary_df: pd.DataFrame):
        """Genera grafici dei risultati"""
        if summary_df.empty: return
        try:
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Grafico ROI
            sns.barplot(data=summary_df, x='player', y='roi', hue='stat', ax=axes[0])
            axes[0].set_title('ROI per Giocatore e Statistica')
            axes[0].axhline(0, color='red', linestyle='--')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Grafico Win Rate
            sns.scatterplot(data=summary_df, x='bets', y='win_rate', hue='stat', size='roi', sizes=(20, 200), ax=axes[1])
            axes[1].set_title('Win Rate vs Volume (Size = ROI)')
            axes[1].axhline(52.6, color='red', linestyle='--', label='Break-Even (1.90)')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(filename)
            print(f"\n📊 Grafico salvato: {filename}")
        except Exception as e:
            print(f"Impossibile generare grafici: {e}")

class NBABacktestingSystem:
    def __init__(self, nba_api_key: str):
        self.collector = NBADataCollector(api_key=nba_api_key)
        self.engine = BacktestingEngine()
        
    def run(self, players: List[str], stats: List[str]):
        print("🔄 Recupero dati storici...")
        data_map = {}
        for p in players:
            # 82 partite per avere una stagione intera di storico
            df = self.collector.get_player_game_logs_extended(p, n_games=82)
            if not df.empty and len(df) > 30:
                data_map[p] = df
                print(f"   ✓ {p}: {len(df)} games")
            else:
                print(f"   ⚠️ {p}: Dati insufficienti (recuperati {len(df)})")
        
        if not data_map: return
        
        # Esegui backtest con soglia 0.02 (2% edge)
        summary, _ = self.engine.run_full_backtest(
            data_map, 
            stat_types=stats,
            edge_thresholds=[0.02] 
        )
        
        if not summary.empty:
            print("\n🏆 RISULTATI FINALI:")
            print(summary.to_string(index=False))
            
            total_roi = summary['roi'].mean()
            print(f"\n📈 ROI MEDIO SISTEMA: {total_roi:.2f}%")
            
            # Genera grafici
            self.engine.plot_results(summary)
            
            # Salva Excel
            summary.to_excel(f"backtest_summary_{datetime.now().strftime('%Y%m%d')}.xlsx", index=False)
        else:
            print("\n❌ Nessuna scommessa trovata.")

if __name__ == "__main__":
    # CONFIGURAZIONE TEST
    PLAYERS = [
        "LeBron James", 
        "Luka Doncic", 
        "Nikola Jokic", 
        "Jayson Tatum", 
        "Giannis Antetokounmpo",
        "Shai Gilgeous-Alexander",
        "Anthony Edwards"
    ]
    
    # Inizializza sistema
    system = NBABacktestingSystem(nba_api_key=config.NBA_API_KEY)
    
    # Esegui su Points, Rebounds, Assists
    system.run(players=PLAYERS, stats=['points', 'rebounds', 'assists'])