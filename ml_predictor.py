import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, Tuple, Optional

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # <--- NUOVO MOTORE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

class MLPredictor:
    """
    Modello Machine Learning per predizioni NBA.
    Ottimizzato: XGBoost + RandomForest e features difensive.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def _get_feature_cols(self, target_stat: str) -> list:
        """Restituisce la lista delle colonne usate come features"""
        return [
            f'{target_stat}_ma3', f'{target_stat}_ma5', f'{target_stat}_ma10',
            f'{target_stat}_std5', f'{target_stat}_trend',
            f'{target_stat}_proj', # Feature pesata (Difesa/Pace)
            'minutes', 'is_home', 'is_back_to_back', 'days_rest',
            'fg_pct', 'usage_proxy',
            'opp_def_rating', # FONDAMENTALE
            'opp_pace'        # FONDAMENTALE
        ]

    def prepare_training_data(self, df: pd.DataFrame, 
                            target_stat: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepara i dati allineando features (t-1) con target (t)"""
        if df.empty or len(df) < 10:
            return None, None
        
        feature_cols = self._get_feature_cols(target_stat)
        
        # TIME SHIFT: Usiamo le stat della partita PRECEDENTE per predire l'ATTUALE
        df_shifted = df.copy()
        for col in feature_cols:
            if col in df.columns:
                df_shifted[col] = df[col].shift(1)
        
        # Assicuriamoci che tutte le colonne esistano
        available_cols = [c for c in feature_cols if c in df_shifted.columns]
        if len(available_cols) != len(feature_cols):
            return None, None

        df_clean = df_shifted.dropna(subset=available_cols + [target_stat])
        
        # Filtro minuti solo per il training
        df_clean = df_clean[df_clean['minutes'] >= 12]
        
        if len(df_clean) < 10:
            return None, None
            
        try:
            X = df_clean[available_cols].values
            y = df_clean[target_stat].values
            return X, y
        except KeyError:
            return None, None
    
    def train_model(self, player_data: pd.DataFrame, stat_type: str, player_name: str) -> Dict:
        """Allena un modello SPECIFICO usando XGBoost e Random Forest"""
        X, y = self.prepare_training_data(player_data, stat_type)
        
        if X is None or len(X) < 10:
            return {'error': 'Dati insufficienti'}
        
        # Split temporale
        test_size = 0.2
        split_idx = int(len(X) * test_size)
        
        X_test = X[:split_idx]
        y_test = y[:split_idx]
        X_train = X[split_idx:]
        y_train = y[split_idx:]
        
        scaler = StandardScaler()
        if len(X_train) < 2: return {'error': 'Dati training insufficienti'}
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Random Forest (Stabilità)
        rf_model = RandomForestRegressor(
            n_estimators=200,      
            max_depth=5,           
            min_samples_leaf=4,    
            max_features='sqrt',   
            random_state=42, 
            n_jobs=-1
        )
        
        # 2. XGBoost (Precisione e Velocità)
        gb_model = XGBRegressor(
            n_estimators=200, 
            max_depth=3, 
            learning_rate=0.03,    
            subsample=0.7, 
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Valutazione Ensemble
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        r2 = r2_score(y_test, ensemble_pred)
        
        model_key = f"{player_name}_{stat_type}"
        self.models[model_key] = {'rf': rf_model, 'gb': gb_model, 'scaler': scaler}
        self.feature_importance[model_key] = rf_model.feature_importances_
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n_samples': len(X)}
    
    def predict(self, player_data: pd.DataFrame, stat_type: str, player_name: str) -> Optional[Dict]:
        """Predizione per la prossima partita"""
        
        model_key = f"{player_name}_{stat_type}"
        
        if model_key not in self.models:
            metrics = self.train_model(player_data, stat_type, player_name)
            if 'error' in metrics:
                return None
        
        if player_data.empty: return None
        
        latest_game = player_data.iloc[0]
        feature_cols = self._get_feature_cols(stat_type)
        features = []
        
        for col in feature_cols:
            val = latest_game.get(col, 0)
            if pd.isna(val): val = 0
            features.append(val)
            
        X_latest = np.array([features])
        
        scaler = self.models[model_key]['scaler']
        try:
            X_scaled = scaler.transform(X_latest)
        except Exception:
            return None
        
        rf_pred = self.models[model_key]['rf'].predict(X_scaled)[0]
        gb_pred = self.models[model_key]['gb'].predict(X_scaled)[0]
        
        prediction = (rf_pred + gb_pred) / 2
        uncertainty = np.std([rf_pred, gb_pred])
        
        # Confidenza basata sul CV
        if prediction > 0:
            cv = uncertainty / prediction
        else:
            cv = 1.0
            
        confidence = np.exp(-5 * cv)
        if confidence > 0.95: confidence = 0.95
        
        return {
            'prediction': round(prediction, 2),
            'uncertainty': round(uncertainty, 2),
            'confidence': round(confidence, 2),
            'rf_pred': round(rf_pred, 2),
            'gb_pred': round(gb_pred, 2)
        }
    
    def save_models(self, filepath: str = 'nba_ml_models.pkl'):
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({'models': self.models, 'scalers': self.scalers, 'feature_importance': self.feature_importance}, f)
            print(f"✓ {len(self.models)} modelli specifici salvati in {filepath}")
        except Exception as e:
            print(f"⚠️ Errore salvataggio modelli: {e}")
    
    def load_models(self, filepath: str = 'nba_ml_models.pkl'):
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.models = data['models']
                self.scalers = data['scalers']
                self.feature_importance = data.get('feature_importance', {})
            print(f"✓ {len(self.models)} modelli caricati da {filepath}")
            return True
        except Exception:
            return False