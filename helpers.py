import pandas as pd
import re

def keyword_searcher(df, keywords):
    return df[
        df.apply(lambda row: any([bool(re.search(keyword, row['tags'], re.IGNORECASE)) or bool(re.search(keyword, row['title'], re.IGNORECASE)) for keyword in keywords]), axis=1)
    ].copy()