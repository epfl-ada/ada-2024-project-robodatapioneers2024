import pandas as pd
import re

def keyword_searcher(df, keywords, columns):
    pattern = '|'.join([r'\b' + re.escape(keyword) + r'\b' for keyword in keywords])

    return df[
        df.apply(lambda row: any(bool(re.search(pattern, row[col], re.IGNORECASE)) for col in columns), axis=1)
    ].copy()