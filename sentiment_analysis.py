from transformers import pipeline
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from typing import List, Dict

BATCH_SIZE = 32768
MAX_LENGTH = 512


def create_sentiment_pipeline(model_path: str) -> pipeline:
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline("sentiment-analysis",
                    model=model_path,
                    tokenizer=model_path,
                    device=device,
                    batch_size=BATCH_SIZE)


def process_texts(sentiment_task: pipeline, texts: List[str]) -> List[Dict]:
    """Process a batch of texts in parallel chunks"""
    return sentiment_task(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        verbose=True,
        batch_size=512,
    )


def analyze_sentiment_in_batches(sentiment_task: pipeline, df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    # Pre-combine texts to avoid doing it in each batch
    df['combined_text'] = df[column_names[0]] + "\n\n" + df[column_names[1]]

    # Convert to datasets format for efficient batch processing
    dataset = Dataset.from_pandas(df)

    def process_batch(batch):
        # Process all three types of text in parallel using generator expressions
        results = process_texts(sentiment_task, batch['combined_text'])
        results_title = process_texts(sentiment_task, batch[column_names[0]])
        results_desc = process_texts(sentiment_task, batch[column_names[1]])

        # Efficiently assign results
        batch.update({
            'sentiment': [r['label'] for r in results],
            'sentiment_score': [r['score'] for r in results],
            'sentiment_title': [r['label'] for r in results_title],
            'sentiment_score_title': [r['score'] for r in results_title],
            'sentiment_description': [r['label'] for r in results_desc],
            'sentiment_score_description': [r['score'] for r in results_desc]
        })
        return batch

    # Process in batches
    results = dataset.map(
        process_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Processing batches"
    )

    # Convert results back to DataFrame efficiently
    for col in ['sentiment', 'sentiment_score', 'sentiment_title',
                'sentiment_score_title', 'sentiment_description',
                'sentiment_score_description']:
        df[col] = results[col]

    # Clean up temporary column
    df.drop('combined_text', axis=1, inplace=True)

    return df


def main():
    model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    sentiment_task = create_sentiment_pipeline(model_path)

    print(f"Using device: {sentiment_task.device}")
    print("Loading data...")

    filtered_df_sport_category = pd.read_parquet(
        'filtered_sport_category_metadata.parquet')

    print("Classifying...")
    filtered_df_sport_category = analyze_sentiment_in_batches(
        sentiment_task,
        filtered_df_sport_category,
        ["title", 'description']
    )

    print("Saving...")
    filtered_df_sport_category.to_parquet(
        "filtered_sport_category_with_sentiment_column_metadata.parquet"
    )
    print("Finsihed saving")


if __name__ == "__main__":
    main()
