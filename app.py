"""Streamlit web app for spam email classification."""

from pathlib import Path
import pickle

import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "model" / "spam_classifier.pkl"


@st.cache_resource
def load_model_artifact() -> dict:
    """Load model artifact from pickle file."""
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def main() -> None:
    """Render the Streamlit UI and run predictions."""
    st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")

    st.markdown(
        """
        <style>
            .result-box {
                border-radius: 12px;
                padding: 16px;
                margin-top: 12px;
                font-size: 18px;
                font-weight: 600;
            }
            .spam {
                background-color: rgba(255, 76, 76, 0.15);
                color: #ff4c4c;
                border: 1px solid rgba(255, 76, 76, 0.4);
            }
            .ham {
                background-color: rgba(46, 204, 113, 0.15);
                color: #2ecc71;
                border: 1px solid rgba(46, 204, 113, 0.4);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📧 Machine Learning Spam Email Classifier")
    st.caption("Built with Scikit-learn Naive Bayes and Streamlit")

    # Load model with clear error messaging.
    try:
        artifact = load_model_artifact()
        model = artifact["model"]
    except FileNotFoundError:
        st.error("Trained model not found. Run `python train_model.py` first.")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load model: {exc}")
        return

    email_text = st.text_area(
        "Enter email text",
        height=180,
        placeholder="Paste email content here, then click Predict.",
    )

    if st.button("Predict", type="primary"):
        if not email_text.strip():
            st.warning("Please enter email text before predicting.")
            return

        try:
            prediction = model.predict([email_text])[0]
            probability = model.predict_proba([email_text])[0]
            spam_probability = float(probability[list(model.classes_).index("spam")])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Prediction failed: {exc}")
            return

        is_spam = prediction == "spam"
        result_class = "spam" if is_spam else "ham"
        result_text = "🚨 Spam" if is_spam else "✅ Not Spam"
        st.markdown(f'<div class="result-box {result_class}">{result_text}</div>', unsafe_allow_html=True)

        # Optional enhancement: confidence + validation metrics.
        st.progress(min(max(spam_probability, 0.0), 1.0), text=f"Spam probability: {spam_probability:.2%}")
        if "accuracy" in artifact:
            st.caption(f"Model validation accuracy: {artifact['accuracy']:.2%}")
        if "confusion_matrix" in artifact:
            labels = artifact.get("labels", list(model.classes_))
            st.write(f"Confusion matrix {labels}:", artifact["confusion_matrix"])


if __name__ == "__main__":
    main()
