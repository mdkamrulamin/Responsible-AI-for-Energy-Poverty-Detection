from src.build_master_dataset import build_master_dataset, save_master_dataset

def main():
    master_df = build_master_dataset()
    save_master_dataset(master_df)
    

if __name__ == "__main__":
    main()