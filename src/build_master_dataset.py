import pandas as pd
from src.load_data import (
    load_raw_electricity_demand_data,
    load_raw_ontario_proxy_temp_data
)

from src.preprocess import (
    preprocess_electricity_demand_data,
    preprocess_weather_data
)

from src.config import MASTER_DATASET_PATH

def build_master_dataset() -> pd.DataFrame:
    demand_df = load_raw_electricity_demand_data()
    weather_df = load_raw_ontario_proxy_temp_data()

    demand_df = preprocess_electricity_demand_data(demand_df)
    weather_df = preprocess_weather_data(weather_df)
    
    master_df = pd.merge(
        demand_df,
        weather_df,
        on="date_only",
        how="left"
    )
    
    #Sort and index master database
    master_df = master_df.sort_values("datetime").reset_index(drop=True)
    
    # Basic checks
    print("\nMaster Dataset Head:")
    print(master_df.head())
    print("\nMaster Dataset Shape:")
    print(master_df.shape)
    print("\nMissing values in master dataset:")
    print(master_df.isnull().sum())
    
    return master_df

def save_master_dataset(master_df: pd.DataFrame, path: str = MASTER_DATASET_PATH) -> None:
    master_df.to_csv(path, index=False)
    print(f"\nMaster dataset saved to: {path}")
    
    
    
    

