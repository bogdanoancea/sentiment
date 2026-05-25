"""
Sentiment Analysis Pipeline

This script performs three sentiment-analysis tasks:

1. Transformer-based three-class sentiment classification
   using cardiffnlp/twitter-roberta-base-sentiment.

2. Continuous sentiment scoring using VADER.

3. Computation of a blended sentiment index:
   sentiment polarity multiplied by transformer confidence.

Input:
    sentiment.csv

Required column:
    notes

Outputs:
    sentiment_results_3class_fulltext.csv
    vader_sentiment_continuous.csv
    blended_sentiment_index.csv
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("sentiment.csv")

TRANSFORMER_OUTPUT_FILE = Path("sentiment_results_3class_fulltext.csv")
VADER_OUTPUT_FILE = Path("vader_sentiment_continuous.csv")
BLENDED_OUTPUT_FILE = Path("blended_sentiment_index.csv")

TEXT_COLUMN = "notes"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
}

LABEL_TO_POLARITY = {
    "Negative": -1,
    "Neutral": 0,
    "Positive": 1,
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_dataset(file_path: Path, text_column: str) -> pd.DataFrame:
    """
    Load the input CSV file and validate the text column.

    Parameters
    ----------
    file_path : Path
        Path to the input CSV file.
    text_column : str
        Name of the column containing text to classify.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with the text column converted to string.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found in the input file. "
            f"Available columns are: {list(df.columns)}"
        )

    df[text_column] = df[text_column].fillna("").astype(str)

    return df


def create_transformer_pipeline(model_name: str):
    """
    Create a Hugging Face sentiment-analysis pipeline.

    Note
    ----
    The pipeline is configured with truncation=False in order to process
    the full text whenever possible.

    If the transformer model fails because some texts are longer than the
    model's maximum input length, change the pipeline configuration to:

        truncation=True,
        max_length=512

    This will truncate long texts to the first 512 tokens, which is the
    standard maximum length for many RoBERTa-based models.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=False,
        return_all_scores=False,
        device=device,
    )

def classify_transformer_sentiment(
    text: str,
    sentiment_pipeline,
) -> Tuple[str, float]:
    """
    Classify text sentiment using the transformer model.

    Parameters
    ----------
    text : str
        Input text.
    sentiment_pipeline : transformers.Pipeline
        Hugging Face sentiment-analysis pipeline.

    Returns
    -------
    tuple[str, float]
        Sentiment label and confidence score.
    """
    if not text.strip():
        return "Empty", 0.0

    try:
        result = sentiment_pipeline(text)[0]
        label = LABEL_MAP.get(result["label"], result["label"])
        confidence = float(result["score"])

        return label, confidence

    except Exception as error:
        print(f"Warning: sentiment classification failed: {error}")
        return "Error", 0.0


def add_transformer_sentiment(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Add transformer-based sentiment labels and confidence scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str
        Name of the text column.

    Returns
    -------
    pd.DataFrame
        Dataframe with sentiment label and confidence columns.
    """
    sentiment_pipeline = create_transformer_pipeline(MODEL_NAME)

    results = df[text_column].apply(
        lambda text: classify_transformer_sentiment(text, sentiment_pipeline)
    )

    df[["Sentiment_Label", "Sentiment_Confidence"]] = pd.DataFrame(
        results.tolist(),
        index=df.index,
    )

    return df


def compute_vader_compound_score(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """
    Compute VADER compound sentiment score.

    Parameters
    ----------
    text : str
        Input text.
    analyzer : SentimentIntensityAnalyzer
        VADER sentiment analyzer.

    Returns
    -------
    float
        VADER compound score in the interval [-1, 1].
    """
    if not text.strip():
        return 0.0

    return analyzer.polarity_scores(text)["compound"]


def add_vader_sentiment(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Add VADER continuous sentiment scores.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    text_column : str
        Name of the text column.

    Returns
    -------
    pd.DataFrame
        Dataframe with VADER compound sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()

    df["Sentiment_Compound"] = df[text_column].apply(
        lambda text: compute_vader_compound_score(text, analyzer)
    )

    return df


def add_blended_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the blended sentiment index.

    The index is defined as:

        Blended Sentiment Index = Sentiment Polarity × Confidence

    where:
        Positive = +1
        Neutral  =  0
        Negative = -1

    The resulting index ranges from -1 to +1.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing Sentiment_Label and Sentiment_Confidence.

    Returns
    -------
    pd.DataFrame
        Dataframe with the blended sentiment index.
    """
    df["Sentiment_Polarity"] = df["Sentiment_Label"].map(LABEL_TO_POLARITY).fillna(0)

    df["Blended_Sentiment_Index"] = (
        df["Sentiment_Polarity"] * df["Sentiment_Confidence"]
    )

    return df


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the full sentiment-analysis workflow.
    """
    print("Loading dataset...")
    df = load_dataset(INPUT_FILE, TEXT_COLUMN)

    print("Running transformer-based sentiment classification...")
    df = add_transformer_sentiment(df, TEXT_COLUMN)
    df.to_csv(TRANSFORMER_OUTPUT_FILE, index=False)
    print(f"Transformer results saved to: {TRANSFORMER_OUTPUT_FILE}")

    print("Computing VADER continuous sentiment scores...")
    df = add_vader_sentiment(df, TEXT_COLUMN)
    df.to_csv(VADER_OUTPUT_FILE, index=False)
    print(f"VADER results saved to: {VADER_OUTPUT_FILE}")

    print("Computing blended sentiment index...")
    df = add_blended_sentiment_index(df)
    df.to_csv(BLENDED_OUTPUT_FILE, index=False)
    print(f"Blended sentiment index saved to: {BLENDED_OUTPUT_FILE}")

    print("Sentiment-analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()