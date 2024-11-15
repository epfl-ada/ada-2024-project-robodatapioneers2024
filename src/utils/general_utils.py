from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_wordcloud(texts: list[str], title: str):
    word_counts = Counter(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
    
    
