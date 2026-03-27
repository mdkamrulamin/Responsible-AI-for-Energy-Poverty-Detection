from src.detect_anomalies import run_detection
from src.config import ARTIFACTS_SAVE_PATH, LSTM_SAVED_MODEL_PATH, THRESHOLD_PERCENTILE

def main():
   run_detection(
       artifacts_dir=ARTIFACTS_SAVE_PATH,
       model_path=LSTM_SAVED_MODEL_PATH,
       threshold_percentile=THRESHOLD_PERCENTILE
   )
    
    

if __name__ == "__main__":
    main()