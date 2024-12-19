from datasets import Dataset
from transformers import pipeline
import torch
import polars as pl
import json
import time

BATCH_SIZE = 1024  # Set batch size for processing


def zero_shot_classification_with_dataset(df, classification_task, labels, batch_size=1024):
    # Convert Polars DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df.to_pandas())

    # Define a function to preprocess each row (to handle the batch)
    def preprocess_and_classify(batch):
        texts = []
        for i in range(len(batch['tags'])):
            texts.append(json.dumps({
                "tags": batch["tags"][i],
                "title": batch["title"][i],
                "description": batch["description"][i]
            }))

        # Perform zero-shot classification on the batch
        results = classification_task(
            texts,
            candidate_labels=labels,
            multi_label=True,
            truncation=True,
            verbose=True,
            batch_size=512
        )

        # Extract scores for each label
        batch_results = {label: [] for label in labels}
        for result in results:
            for label, score in zip(result["labels"], result["scores"]):
                batch_results[label].append(score)

        return batch_results

    # Apply preprocessing and classification with native batching
    results = dataset.map(preprocess_and_classify,
                          batched=True, batch_size=batch_size)

    print(results)

    # Add classification scores back to DataFrame
    all_results = {label: [] for label in labels}

    for result in results:
        for label in labels:
            all_results[label].append(result["scores"].get(label, 0))

    # Convert the results to Polars and join with the original DataFrame
    for label in labels:
        df = df.with_columns(
            pl.Series(f"{label}_classification", all_results[label])
        )

    return df


def main():
    device = "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    # model_path = "facebook/bar t-large-mnli"
    # model_path = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    # maybe this model is faster, this needs tokenizer and sentencepiece
    model_path = "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-binary"
    classification_task = pipeline(
        "zero-shot-classification", model=model_path, tokenizer=model_path, device=device)

    sport_labels = ["football", "basketball", "tennis", "golf", "rugby", "cricket",
                    "hockey", "baseball", "volleyball", "american football", "olympics"]

    major_events_labels = ["Olympics Rio de Janeiro 2016", "Olympics London 2012", "Olympics Beijing 2008", "Olympics Pyeongchang 2018",
                           "Olympics Sochi 2014", "Olympics Vancouver 2010", "World cup Russia 2018", "World cup Brazil 2014", "World cup South Africa 2010"]

    print("Loading data...")
    filtered_df_sport_category = pl.read_parquet(
        'filtered_sport_category_metadata.parquet')

    print("Classifying...")
    filtered_df_sport_category_new = zero_shot_classification_with_dataset(
        filtered_df_sport_category, classification_task, major_events_labels, 8192
    )

    print("Saving...")
    filtered_df_sport_category_new = filtered_df_sport_category_new.drop(
        "description")

    # Events
    filtered_df_sport_category_new.write_parquet(
        'filtered_sport_category_without_description_metadata_with_classifications_events.parquet'
    )


if __name__ == "__main__":
    main()
