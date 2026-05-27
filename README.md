# Spam-Email-Classifier

A Machine Learning-based Email Spam Classifier web app built with **Python**, **Scikit-learn**, and **Streamlit**.

## Features

- Naive Bayes spam classifier
- Word-frequency text features via `CountVectorizer`
- Streamlit UI with email text input and Predict button
- Styled Spam / Not Spam output
- Prediction probability score
- Validation accuracy and confusion matrix display
- Pickle-based model save/load workflow
- Error handling for missing model and invalid input

## Project Structure

```text
Spam-Email-Classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── spam_emails.csv
└── model/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train Model

```bash
python train_model.py
```

This creates `model/spam_classifier.pkl`.

## Run Streamlit App

```bash
streamlit run app.py
```

Enter email text in the text area and click **Predict** to classify as Spam or Not Spam.
