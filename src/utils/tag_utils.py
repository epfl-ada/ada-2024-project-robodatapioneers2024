import pandas as pd
import re

def get_processed_tags(df: pd.DataFrame) -> list[str]:
    df['tags'] = df['tags'].str.lower()
    df['tags'] = df['tags'].str.replace("world cup", "world_cup")
    df['tags'] = df['tags'].str.replace("table tennis", "table_tennis")
    df['tags'] = df['tags'].str.replace("ping pong", "table_tennis")
    
    all_tags = []
    for tags in df['tags']:
        tag_list = tags.split(",")
        for sentence in tag_list:
            sentence = re.sub(r'[^\w\s]', '', sentence)
            all_tags.extend(sentence.split(' '))
            
    return all_tags