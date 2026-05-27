"""Train and save a Naive Bayes spam email classifier."""

from pathlib import Path
import pickle
import warnings

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spam_emails.csv"
MODEL_PATH = BASE_DIR / "model" / "spam_classifier.pkl"


def train_and_save_model() -> None:
    """Train Naive Bayes spam classifier and save it using pickle."""
    # Load dataset with expected columns.
    df = pd.read_csv(DATA_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must include 'text' and 'label' columns.")
    if len(df) < 100:
        warnings.warn(
            "Dataset is small; model quality may be limited. Consider adding more labeled emails.",
            stacklevel=2,
        )

    # Create text features using word frequencies, then fit Naive Bayes.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        [
            ("vectorizer", CountVectorizer(stop_words="english", max_features=5000)),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(X_train, y_train)

    # Evaluate and persist model + metrics for UI display.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "accuracy": float(accuracy),
                "confusion_matrix": cm,
            },
            model_file,
        )

    print(f"Model trained and saved to: {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print("Confusion matrix [ham, spam]:")
    print(cm)


if __name__ == "__main__":
    train_and_save_model()
