from src.models.sentiment_analysis import SentimentAnalysisModel

DATA_PATH = "/data"

if __name__ == "__main__":
    # Load sports category videos dataset (with descriptions and tags)
    pq_sports = pq.ParquetFile(os.path.join(DATA_PATH, "yt_metadata_en_sport.parquet"))
    df_vd_sports = pq_sports.read().to_pandas()

    # Change upload_date to datetime
    df_vd_sports["upload_date"] = pd.to_datetime(
        df_vd_sports["upload_date"], format="%Y-%m-%d %H:%M:%S"
    )

    sentiment_model = SentimentAnalysisModel()

    df_vd_sports = df_vd_sports[~df_vd_sports_copy["title"].str.contains("fishing", case=False)]

    df_vd_sports["sentiment"] = df_vd_sports["title"].apply(
        lambda x: sentiment_model.get_sentiment_label(x)
    )

    # df_vd_sports["description_sentiment"] = df_vd_sports["description"].apply(
    #     lambda x: sentiment_model.get_sentiment_label(x)
    # )

    # Save the dataset
    df_vd_sports.to_parquet(
        os.path.join(DATA_PATH, "yt_metadata_en_sport_sentiment.parquet"),
        engine="pyarrow",
    )
