import pandas as pd
import re

def keyword_searcher(df, keywords):
    pattern = '|'.join([r'\b' + re.escape(keyword) + r'\b' for keyword in keywords])

    return df[
        df.apply(lambda row: bool(re.search(pattern, row['tags'], re.IGNORECASE)) or bool(re.search(pattern, row['title'], re.IGNORECASE)), axis=1)
    ].copy()