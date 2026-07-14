
import joblib
import numpy as np
import pandas as pd
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import collections





def load_dataset():
    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")

    fake_df["label"] = 0   
    true_df["label"] = 1   

    
    df = pd.concat([fake_df, true_df], ignore_index=True)

    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["text"] = (
        df["title"].fillna("") + " " +
        df["text"].fillna("")
    )

    return df[["text", "label"]]

#Feature Engineering


def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s!?.,]', '', text)
    return text.strip()

def extract_features(text):
    """Extract hand-crafted features for explainability."""
    t = text.lower()
    words = t.split()
    
    # Emotional / sensational words
    sensational = ['shocking', 'breaking', 'urgent', 'exposed', 'secret', 'hidden',
                   'truth', 'bombshell', 'leaked', 'proof', 'hoax', 'fake', 'lie',
                   'conspiracy', 'banned', 'censored', 'suppressed', 'must share',
                   'they dont want', 'wake up', 'share before deleted']
    
    # Credibility signals
    credible = ['according to', 'study', 'research', 'published', 'university',
                'journal', 'percent', 'report', 'official', 'announced', 'data',
                'statistics', 'findings', 'evidence', 'peer-reviewed', 'confirmed']
    
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation = text.count('!')
    question = text.count('?')
    sensational_score = sum(1 for w in sensational if w in t)
    credibility_score = sum(1 for w in credible if w in t)
    word_count = len(words)
    
    return {
        "caps_ratio": round(caps_ratio * 100, 1),
        "exclamation_count": exclamation,
        "question_count": question,
        "sensational_score": sensational_score,
        "credibility_score": credibility_score,
        "word_count": word_count,
    }



#Train & Save


def train():
    print("Loading dataset")
    df = load_dataset()
    df['text'] = df['text'].apply(preprocess)

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training TFidf Log Regression pipeline")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            stop_words='english',
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/fake_news_model.joblib")
    print("\nModel saved → model/fake_news_model.joblib")

if __name__ == "__main__":
    train()
