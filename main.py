from src.prepare_sequences import prepare_lstm_inputs

def main():
    X_train, X_val, X_test, scaler = prepare_lstm_inputs()
    
    

if __name__ == "__main__":
    main()