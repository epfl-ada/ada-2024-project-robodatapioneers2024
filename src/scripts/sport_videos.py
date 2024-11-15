import os
import pyarrow.parquet as pq
import pandas as pd 


# Note: run this script from the root directory of the project

def get_sports_catogory_videos(parquet_file_path: str, new_file_path: str, batch_size: int = 1_000_000) -> pd.DataFrame:
    """
    Get sports category videos from the youtube metadata.
    """
    pq_metadata = pq.ParquetFile(parquet_file_path)
    sport_df = pd.DataFrame()

    for batch in pq_metadata.iter_batches(batch_size=batch_size):
        temp_df = batch.to_pandas()
        temp_df = temp_df[temp_df['categories'] == "Sports"]
        sport_df = pd.concat([sport_df, temp_df])

    sport_df.to_parquet(
        new_file_path, engine="fastparquet"
    )

    return sport_df

if __name__ == "__main__":
    parquet_file_path = os.path.join("./data", "yt_metadata_en.parquet")
    sport_file_path = os.path.join("./data", "yt_metadata_en_sport.parquet")

    sport_df = get_sports_catogory_videos(parquet_file_path, sport_file_path)
    print(sport_df.head())
    print(sport_df.shape)