from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np


class SentimentAnalysisModel:
    def __init__(self, model_name: str = "vader"):
        self.model_name = model_name
        if self.model_name == "vader":
            self.model = SentimentIntensityAnalyzer()
        else:
            raise ValueError("Model not supported")

    def get_sentiment_label(self, text: str):
        if self.model_name == "vader":
            result = self.model.polarity_scores(text)
            label = np.argmax([result["neg"], result["neu"], result["pos"]])
            if label == 0:
                return "negative"
            elif label == 1:
                return "neutral"
            else:
                return "positive"
