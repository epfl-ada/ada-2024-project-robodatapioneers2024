from transformers import pipeline
import torch
import pandas as pd
import numpy as np
from datasets import Dataset


BATCH_SIZE = 1024
MAX_LENGTH = 512  # Maximum length for the model


def analyze_sentiment_in_batches(sentiment_task, df, column_names, batch_size=BATCH_SIZE):
    dataset = Dataset.from_pandas(df)
    sentiments = []

    def process_batch(batch):
        batch_text = []
        for title, desc in zip(batch[column_names[0]], batch[column_names[1]]):
            combined_text = f"{title}\n\n{desc}"
            batch_text.append(combined_text)

        results = sentiment_task(
            batch_text, verbose=True, batch_size=512, truncation=True, max_length=MAX_LENGTH
        )

        batch['sentiment'] = [item['label'] for item in results]
        batch['sentiment_score'] = [item['score'] for item in results]

        return batch

    results = dataset.map(process_batch, batched=True, batch_size=batch_size)

    df['sentiment'] = np.concatenate([result['sentiment'] for result in results])
    df['sentiment_score'] = np.concatenate([result['sentiment_score'] for result in results])

    return df

def main():
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    sentiment_task = pipeline("sentiment-analysis", model=model_path,
                              tokenizer=model_path, device=device)

    print(device)

    print("loading data...")
    filtered_df_sport_category = pd.read_parquet(
        'filtered_sport_category_metadata.parquet')

    print("Classifying...")
    filtered_df_sport_category = analyze_sentiment_in_batches(
        sentiment_task, filtered_df_sport_category, ["title", 'description'], 4096)

    print("Save...")
    filtered_df_sport_category = filtered_df_sport_category.drop(
        columns=['description'])
    filtered_df_sport_category.to_parquet(
        "filtered_sport_category_with_sentiment_and_without_description_column_metadata.parquet")


if __name__ == "__main__":
    main()
