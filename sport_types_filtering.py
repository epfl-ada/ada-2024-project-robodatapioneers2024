from transformers import pipeline
import torch
import pandas as pd
import numpy as np
import polars as pl
import json
import time

BATCH_SIZE = 1024


def zero_shot_classification_in_batches(df, classification_task, batch_size=BATCH_SIZE):
    total_rows = len(df)
    classifications = {label: [] for label in sport_labels}

    for i in range(0, total_rows, batch_size):
        end_index = min(i + batch_size, total_rows)
        # Extract the batch
        batch = df[i:end_index]

        texts = [json.dumps({"tags": row["tags"], "title": row["title"], "description": row["description"]})
                 for row in batch.to_dicts()]

        results = classification_task(
            texts, sport_labels, multi_label=True, truncation=True, batch_size=256, verbose=True)

        for j, classification in enumerate(results):
            for z, label in enumerate(sport_labels):
                classifications[label].append(classification["scores"][z])

        print(f"Processed {min(i + batch_size, total_rows)
                           }/{total_rows} rows...")

    for key in classifications:
        df = df.with_columns(
            pl.Series(f"{key}_classification", classifications[key]))

    return df


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"

    model_path = "facebook/bart-large-mnli"
    classification_task = pipeline("zero-shot-classification", model=model_path,
                                   tokenizer=model_path, device=device)

    sport_labels = ["football", "basketball", "tennis", "golf", "rugby", "cricket",
                    "hockey", "baseball", "volleyball", "american football", "olympics"]

    print("Loading data...")
    filtered_df_sport_category = pl.read_parquet(
        'filtered_sport_category_metadata.parquet')

    print("Classifying....")
    filtered_df_sport_category_new = zero_shot_classification_in_batches(
        filtered_df_sport_category, classification_task, batch_size=BATCH_SIZE)

    print("Saving...")
    filtered_df_sport_category_new = filtered_df_sport_category_new.drop(
        "description")
    filtered_df_sport_category_new.write_parquet(
        'filtered_sport_category_without_description_metadata_with_classifications.parquet')
