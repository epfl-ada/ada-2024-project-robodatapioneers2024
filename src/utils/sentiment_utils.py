import matplotlib.pyplot as plt
import numpy as np
import pymannkendall as mk


def plot_sentiment_percentage_distribution_over_years(
    df,
    title,
    sentiment_column="title_sentiment",
):
    """This function plots the sentiment percentage distribution over years."""
    df = df.copy()
    df["upload_date_year"] = df["upload_date"].dt.year

    sentiment_counts = (
        df.groupby(["upload_date_year", sentiment_column])
        .size()
        .reset_index(name="count")
    )
    
    total_counts_per_year = sentiment_counts.groupby("upload_date_year")[
        "count"
    ].transform("sum")
    
    sentiment_counts["percentage"] = (
        sentiment_counts["count"] / total_counts_per_year * 100
    )

    # Pop "neutral" sentiment column since it is dominant
    sentiment_counts = sentiment_counts[sentiment_counts[sentiment_column] != "neutral"]

    pivot_df = sentiment_counts.pivot(
        index="upload_date_year", columns=sentiment_column, values="percentage"
    ).fillna(0)

    pivot_df.plot(kind="bar", stacked=True, figsize=(15, 5), cmap="viridis")
    plt.xlabel("Year")
    plt.ylabel("Percentage")
    plt.title(title)
    plt.legend(title="Sentiment")
    plt.xticks(rotation=45)
    plt.show()


def mann_kendall_trend_test(df):
    """
    This function performs the Mann-Kendall negative trend test for title sentiment.
    """
    df = df.copy()
    df["upload_date_year"] = df["upload_date"].dt.year

    neg_sentiment = df[df["title_sentiment"] == "negative"]
    yearly_negative = (
        neg_sentiment.groupby("upload_date_year").size()
        / df.groupby("upload_date_year").size()
    ).reset_index(name="negative_percentage")

    # Apply moving average for smoothing
    window_size = 3
    yearly_negative["negative_percentage"] = (
        yearly_negative["negative_percentage"].rolling(window=window_size).mean()
    )

    # Perform Mann-Kendall trend test
    mk_result = mk.original_test(yearly_negative["negative_percentage"].dropna())
    print(f"Mann-Kendall Test Result: {mk_result}")

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        yearly_negative["upload_date_year"],
        yearly_negative["negative_percentage"],
        label="Raw Percentage",
        marker="o",
    )
    ax.plot(
        yearly_negative["upload_date_year"],
        yearly_negative["smoothed_percentage"],
        label=f"{window_size}-Year Moving Average",
        linestyle="--",
    )
    plt.title("Trend of Negative Sentiment in Sports Videos Over Time")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Negative Sentiment")
    plt.legend()
    plt.grid()
    plt.show()
