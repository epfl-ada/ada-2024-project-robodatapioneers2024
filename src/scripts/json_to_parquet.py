import pandas as pd
import os

# Note: run this script from the root directory of the project
def convert_json_to_parquet(json_file_path: str, parquet_file_path: str, chunksize: int = 1_000_000):
    for i, chunk in enumerate(pd.read_json(json_file_path, lines=True, chunksize=chunksize)):
        print(f"Processing chunk {i}")
        if i == 0:
            chunk.to_parquet(parquet_file_path)
        else:
            chunk.to_parquet(parquet_file_path,
                             engine="fastparquet", append=True)
            
if __name__ == "__main__":
    json_file_path = os.path.join("./data", "yt_metadata_en.json")
    parquet_file_path = os.path.join("./data", "yt_metadata_en.parquet")
    
    print("Converting JSON to Parquet...")
    convert_json_to_parquet(json_file_path, parquet_file_path)
    print("Conversion completed!")