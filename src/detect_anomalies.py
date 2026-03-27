import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from src.train_autoencoder import run_training, load_artifacts
from src.prepare_sequences import load_master_dataset, split_dataset_by_time
from src.config import *

def load_trained_model(
    model_path: str = LSTM_SAVED_MODEL_PATH,
    input_dir: str = ARTIFACTS_SAVE_PATH,
    model_output_dir: str = MODEL_SAVE_PATH,
    plot_output_dir: str = PLOTS_SAVE_PATH
) -> Model:
    #First check if trained model exists
    #If not then train the model first and save it
    
    if os.path.exists(model_path):
        print(f"\n:: LOADING EXISTING MODEL FROM: {model_path} ::")
        model = load_model(model_path)
        return model
    
    print("\n:: TRAINED MODEL NOT FOUND ::")
    print("Running training to create and save the model")
    
    model, history = run_training(
        input_dir=input_dir,
        model_output_dir=model_output_dir,
        plot_output_dir=plot_output_dir,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    print(f"\n:: MODEL TRAINED AND SAVED TO: {model_path} ::")
    return model

def reconstruct_sequences(model, X: np.ndarray) -> np.ndarray:
    #Reconstruct the sequences using trained autoencoder
    return model.predict(X, verbose=0)

def compute_reconstruction_error(X_true: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
    #Compute the MSE reconstruction error per sequence
    errors = np.mean(np.square(X_true - X_pred), axis=(1,2))
    return errors

def get_anomaly_threshold(train_errors: np.ndarray, percentile: float = THRESHOLD_PERCENTILE) -> float:
    #Compute anomaly threshold from training reconstruction errors
    threshold = np.percentile(train_errors, percentile)
    print(f"\n:: THRESHOLD SUMMARY ::")
    print(f"Percentile used: {percentile}")
    print(f"Threshold value: {threshold:.8f}")
    return threshold

def build_error_results_dataframe(
    split_df: pd.DataFrame,
    errors: np.ndarray,
    threshold: float,
    sequence_length: int = SEQUENCE_LENGTH
) -> pd.DataFrame:
    #Align one reconstruction error to the last timestamp of each sequence
    
    #For a rolling window of length T, the first error belongs to row T-1
    aligned_df = split_df.iloc[sequence_length -1:].copy().reset_index(drop=True)
    
    if len(aligned_df) != len(errors):
        raise ValueError(f"Length mismatch: aligned_df={len(aligned_df)} vs errors={len(errors)}")
    
    aligned_df["reconstruction_error"] = errors
    aligned_df["anomaly_flag"] = (aligned_df["reconstruction_error"] > threshold).astype(int)
    
    return aligned_df

def save_anomaly_results(
    train_results: pd.DataFrame,
    val_results: pd.DataFrame,
    test_results: pd.DataFrame,
    threshold: float,
    percentile: float,
    output_dir: str = ANOMALY_SAVE_PATH
) -> None:
    #Saves the anomaly result as CSV and also saves threshold metadata
    
    os.makedirs(output_dir, exist_ok=True)
    train_results.to_csv(os.path.join(output_dir, "train_anomaly_results.csv"), index=False)
    val_results.to_csv(os.path.join(output_dir, "val_anomaly_results.csv"), index=False)
    test_results.to_csv(os.path.join(output_dir, "test_anomaly_results.csv"), index=False)
    
    metadata = {
        "threshold_percentile": percentile,
        "threshold_value": float(threshold)
    }
    
    with open(os.path.join(output_dir, "threshold_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSaved anomaly result files to: {output_dir}")
    

def plot_reconstruction_error_distribution(
    train_errors: np.ndarray,
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    threshold: float,
    output_dir: str = PLOTS_SAVE_PATH
) -> str:
    #Plot reconstruction error histograms for train, val and test
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "reconstruction_error_distribution.png")
    
    plt.figure(figsize=(10, 6))
    plt.hist(train_errors, bins=60, alpha=0.5, label="Train Errors")
    plt.hist(val_errors, bins=60, alpha=0.5, label="Validation Errors" )
    plt.hist(test_errors, bins=60, alpha=0.5, label="Test Errors" )
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.6f}")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Reconstruction error distribution plot saved to: {plot_path}")
    return plot_path


def print_anomaly_summary(name: str, results_df: pd.DataFrame) -> None:
    #Print summary statistics for anomaly results
    anomaly_count = results_df["anomaly_flag"].sum()
    total_count = len(results_df)
    anomaly_rate = (anomaly_count / total_count ) * 100 if total_count > 0 else 0
    
    print(f"\n:: {name.upper()} ANOMALY SUMMARY ::")
    print("Total sequences:", total_count)
    print("Anomalies detected:", anomaly_count)
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    print("Reconstruction error stats:")
    print(results_df["reconstruction_error"].describe())
    
def run_detection(
    artifacts_dir: str = ARTIFACTS_SAVE_PATH,
    model_path: str = LSTM_SAVED_MODEL_PATH,
    master_dataset_path: str = MASTER_DATASET_PATH,
    threshold_percentile: float = THRESHOLD_PERCENTILE,
    sequence_length: int = SEQUENCE_LENGTH
):
    #Load sequences and model
    X_train, X_val, X_test, scaler = load_artifacts(input_dir=artifacts_dir)
    model = load_trained_model(model_path=model_path)
    
    #Reconstruct sequences
    X_train_pred = reconstruct_sequences(model, X_train)
    X_val_pred = reconstruct_sequences(model, X_val)
    X_test_pred = reconstruct_sequences(model, X_test)
    
    #Computer reconstruction errors
    train_errors = compute_reconstruction_error(X_train, X_train_pred)
    val_errors = compute_reconstruction_error(X_val, X_val_pred)
    test_errors = compute_reconstruction_error(X_test, X_test_pred)
    
    #Threshold from training only
    threshold = get_anomaly_threshold(train_errors, percentile=threshold_percentile)
    
    #Reload master dataset to recover timestamps and weather context
    master_df = load_master_dataset(master_dataset_path)
    train_df, val_df, test_df = split_dataset_by_time(master_df)
    
    #Build aligned result dataframes
    train_results = build_error_results_dataframe(
        train_df, train_errors, threshold, sequence_length=sequence_length
    )
    val_results = build_error_results_dataframe(
        val_df, val_errors, threshold, sequence_length=sequence_length
    )
    test_results = build_error_results_dataframe(
        test_df, test_errors, threshold, sequence_length=sequence_length
    )
    
    #Print summaries
    print_anomaly_summary("train", train_results)
    print_anomaly_summary("validation", val_results)
    print_anomaly_summary("test", test_results)
    
    #Save CSVs, metadata and plot
    save_anomaly_results(
        train_results=train_results,
        val_results=val_results,
        test_results=test_results,
        threshold=threshold,
        percentile=threshold_percentile,
        output_dir=ANOMALY_SAVE_PATH
    )
    
    plot_reconstruction_error_distribution(
        train_errors=train_errors,
        val_errors=val_errors,
        test_errors=test_errors,
        threshold=threshold,
        output_dir=PLOTS_SAVE_PATH
    )
    
    return train_results, val_results, test_results, threshold
    