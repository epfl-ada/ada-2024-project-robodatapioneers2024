import os
import pyarrow.parquet as pq
import pandas as pd 

# Note: run this script from the root directory of the project

def convert_json_to_parquet(json_file_path: str, parquet_file_path: str, chunksize: int = 1_000_000):
    # json_file_path = "../data/yt_metadata_en.jsonl"
    # chunksize = 1_000_000
    # parquet_file_path = "../data/yt_metadata_en.parquet"

    for i, chunk in enumerate(pd.read_json(json_file_path, lines=True, chunksize=chunksize)):
        print(f"Processing chunk {i}")
        if i == 0:
            chunk.to_parquet(parquet_file_path)
        else:
            chunk.to_parquet(parquet_file_path,
                             engine="fastparquet", append=True)

def get_sports_catogory_videos(parquet_file_path: str) -> pd.DataFrame:
    """
    Get sports category videos from the youtube metadata.
    """
    pq_metadata = pq.ParquetFile(parquet_file_path)
    sport_df = pd.DataFrame()

    for batch in pq_metadata.iter_batches(batch_size=1000000):
        temp_df = batch.to_pandas()
        temp_df = temp_df[temp_df['categories'] == "Sports"]
        sport_df = pd.concat([sport_df, temp_df])
        
    sport_df.to_parquet(os.path.join("./data", "yt_metadata_en_sport.parquet"), engine='fastparquet')
    
    return sport_df

if __name__ == "__main__":
    json_file_path = os.path.join("./data", "yt_metadata_en.jsonl")
    parquet_file_path = os.path.join("./data", "yt_metadata_en.parquet")
    convert_json_to_parquet(json_file_path, parquet_file_path)

    sport_df = get_sports_catogory_videos(parquet_file_path)
    print(sport_df.head())
    print(sport_df.shape)