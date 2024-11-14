import pandas as pd
import re
import pyarrow.parquet as pq

def get_related_videos_with_keywords(
    df: pd.DataFrame,
    keywords: list[str],
    check_tags: bool = True,
    check_title: bool = True,
    check_description: bool = True,
) -> pd.DataFrame:
    """
    Get videos related to a specific event by filtering the videos with tags or titles or descriptions containing the event keywords.
    """
    keywords = [keyword.lower() for keyword in keywords]
    df = df.copy()

    assert (
        check_tags or check_title or check_description
    ), "At least one of the check_tags, check_title, or check_description should be True."

    if check_title:
        df["title_lower"] = df["title"].str.lower()
        df["is_related"] = df["title_lower"].apply(
            lambda x: any(keyword in x for keyword in keywords)
        )

    if check_tags:
        df["tags_lower"] = df["tags"].str.lower()
        df["tags_lower"] = df["tags_lower"].str.split(",")
        df["is_related"] = df["is_related"] | df["tags_lower"].apply(
            lambda x: any(keyword in x for keyword in keywords)
        )

    if check_description:
        df["description_lower"] = df["description"].str.lower()
        df["description_lower"].apply(
            lambda x: any(keyword in x for keyword in keywords)
        )
        df["is_related"] = df["is_related"] | df["description_lower"].apply(
            lambda x: any(keyword in x for keyword in keywords)
        )

    return df[df["is_related"]]


def keyword_searcher(df, keywords):
    return df[
        df.apply(lambda row: any([bool(re.search(keyword, row['tags'], re.IGNORECASE)) or bool(
            re.search(keyword, row['title'], re.IGNORECASE)) for keyword in keywords]), axis=1)
    ].copy()


def convert_json_to_parquet():
    json_file_path = "../data/yt_metadata_en.jsonl"
    chunksize = 1_000_000
    parquet_file_path = "../data/yt_metadata_en.parquet"

    for i, chunk in enumerate(pd.read_json(json_file_path, lines=True, chunksize=chunksize)):
        print(f"Processing chunk {i}")
        if i == 0:
            chunk.to_parquet(parquet_file_path)
        else:
            chunk.to_parquet(parquet_file_path, engine="fastparquet", append=True)