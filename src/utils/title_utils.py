import re

stop_words = set(stopwords.words('english'))
stop_words.discard('how')
stop_words.discard('against')

def get_processed_title(title: str):
    # Remove punctuation and non-alphanumeric characters
    cleaned_text = re.sub(r'[^\w\s]', '', title)
    
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.replace("world cup", "world_cup")
    cleaned_text = cleaned_text.replace("table tennis", "table_tennis")
    cleaned_text = cleaned_text.replace("ping pong", "table_tennis")
    
    cleaned_text.replace("world cup", "")
    cleaned_text = [word for word in cleaned_text.split() if word not in stop_words]
    
    return cleaned_text