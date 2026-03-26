from src.prepare_sequences import prepare_lstm_inputs, save_artifacts

def main():
    X_train, X_val, X_test, scaler = prepare_lstm_inputs()
    
    save_artifacts(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        scaler=scaler
    )
    
    

if __name__ == "__main__":
    main()