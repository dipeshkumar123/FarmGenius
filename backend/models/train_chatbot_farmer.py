import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

corpus_file = r'd:\Projects\FarmGenius\FARMER_CORPUS.md'
output_model_file = r'd:\Projects\FarmGenius\backend\models\chatbot_farmer_v1.pkl'

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def augment_hinglish(query):
    # Simple heuristic dictionary for hinglish augmentation
    replacements = {
        'gehu': 'wheat',
        'kapas': 'cotton',
        'dhan': 'paddy',
        'aloo': 'potato',
        'patti': 'leaves',
        'patte': 'leaves',
        'tamatam': 'tomato',
        'chane': 'chickpea',
        'baigan': 'brinjal'
    }
    augmented = query.lower()
    for k, v in replacements.items():
        augmented = augmented.replace(k, v)
    return augmented

queries = []
categories = []

with open(corpus_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('|') and not line.startswith('| ID') and not line.startswith('|---'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 8:
                lang = parts[2]
                query = parts[3]
                category = parts[5]
                if query and category and category != 'Category' and category != 'General':
                    queries.append(query)
                    categories.append(category)
                    
                    if lang.lower() == 'hindi':
                        queries.append(augment_hinglish(query))
                        categories.append(category)

X = [preprocess(q) for q in queries]
y = categories

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
    ('clf', MultinomialNB())
])

pipeline.fit(X, y)

y_pred = pipeline.predict(X)
print("Classification Report:")
print(classification_report(y, y_pred))

with open(output_model_file, 'wb') as f:
    pickle.dump(pipeline, f)

print(f"Model saved to {output_model_file}")
