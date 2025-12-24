import traceback
import config
import requests
from bet_finder import MLValueBetFinder

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║              🤖 NBA ML-POWERED VALUE BET FINDER 🎯                        ║
║                                                                          ║
║        Random Forest + Gradient Boosting per predizioni accurate         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Inizializza sistema ML usando le chiavi dal config
    finder = MLValueBetFinder(
        nba_api_key=config.NBA_API_KEY,
        odds_api_key=config.ODDS_API_KEY
    )
    
    # Esegui analisi
    try:
        print("🚀 Avvio analisi ML...")
        results = finder.run_ml_analysis()
        
        if not results.empty:
            print("\n" + "="*80)
            print(f"✅ ANALISI ML COMPLETATA!")
            print(f"   {len(results)} value bets identificati con Machine Learning")
            print("="*80)
            
            # Salva modelli per riutilizzo futuro
            finder.ml_predictor.save_models()
        
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()