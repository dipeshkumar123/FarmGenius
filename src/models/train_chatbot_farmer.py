import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def parse_corpus(filepath):
    queries = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        in_table = False
        for line in lines:
            line = line.strip()
            if line.startswith("| ID | Language"):
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                continue
            if in_table and line.startswith("|"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 6:
                    lang = parts[2]
                    query = parts[3]
                    category = parts[5]
                    
                    if not query or category == "Category":
                        continue
                    
                    queries.append(query)
                    labels.append(category)
                    
                    # Augment Hinglish for Hindi
                    if lang.lower() == 'hindi':
                        # Basic Hinglish augmentation: inserting english crop names if applicable
                        if len(parts) >= 7:
                            crop_en = parts[6]
                            if crop_en and crop_en != 'NA' and crop_en != 'Any':
                                # Replace first word roughly or just append to make it Hinglish-like
                                hinglish = query.replace(" ki ", f" {crop_en} ki ")
                                if hinglish != query:
                                    queries.append(hinglish)
                                    labels.append(category)
            
            if in_table and not line:
                in_table = False
                
    return queries, labels

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Note: intentionally NOT removing stopwords because farmers' stop words carry meaning
    return text

def train_model():
    corpus_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../FARMER_CORPUS.md'))
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        return
        
    queries, labels = parse_corpus(corpus_path)
    if not queries:
        print("No queries parsed!")
        return
        
    # Preprocess
    queries = [preprocess(q) for q in queries]
    
    X_train, X_test, y_train, y_test = train_test_split(queries, labels, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        ('clf', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))
    
    model_dir = os.path.join(os.path.dirname(__file__), '../../models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'chatbot_farmer_v1.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
