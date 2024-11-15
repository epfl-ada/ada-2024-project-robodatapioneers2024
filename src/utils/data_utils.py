import pandas as pd


def get_related_videos_with_keywords(
    df: pd.DataFrame, 
    keywords: list[str],
    check_tags: bool = True,
    check_title: bool = True,
    check_description: bool = True
) -> pd.DataFrame:
    """
    Get videos related to a specific event by filtering the videos with tags or titles or descriptions containing the event keywords.
    """
    keywords = [keyword.lower() for keyword in keywords]
    df = df.copy()
    
    assert check_tags or check_title or check_description, "At least one of the check_tags, check_title, or check_description should be True."
    
    if check_title:
        df["title_lower"] = df["title"].str.lower()
        df["is_related"] = df["title_lower"].apply(lambda x: any(keyword in x for keyword in keywords))
        
    if check_tags:
        df["tags_lower"] = df["tags"].str.lower()
        df["tags_lower"] = df["tags_lower"].str.split(",")
        df["is_related"] = df["is_related"] | df["tags_lower"].apply(lambda x: any(keyword in x for keyword in keywords))
        
    if check_description:
        df["description_lower"] = df["description"].str.lower()
        df["description_lower"].apply(lambda x: any(keyword in x for keyword in keywords))
        df["is_related"] = df["is_related"] | df["description_lower"].apply(lambda x: any(keyword in x for keyword in keywords))
                           
    return df[df["is_related"]]
