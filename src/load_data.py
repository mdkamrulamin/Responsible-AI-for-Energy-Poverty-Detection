import pandas as pd

def load_raw_electricity_demand_data():
    dataset = pd.read_csv('data/ontario_electricity_demand.csv')
    print("Electricity Demand Data")
    print("Head: \n",  dataset.head()) 
    print("Columns: \n", dataset.columns) 
    print("Shape: \n", dataset.shape)
    print("Missing val: \n", dataset.isnull().sum())
    print("Data types: \n", dataset.dtypes)
    return dataset

def load_raw_ontario_proxy_temp_data():
    dataset = pd.read_csv('data/ontario_weather_proxy_2019_2023.csv')
    print("\nOntario Weather Proxy Data")
    print("Head: \n",  dataset.head()) 
    print("Columns: \n", dataset.columns) 
    print("Shape: \n", dataset.shape)
    print("Missing val: \n", dataset.isnull().sum())
    print("Data types: \n", dataset.dtypes)
    return dataset