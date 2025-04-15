import string
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_dataset(categories, limit_per_category=300):
    """
    Load and slice the 20 Newsgroups dataset
    """
    data = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    
    # Limit number of samples per category
    filtered_data = {'data': [], 'target': [], 'target_names': data.target_names}
    count = {i: 0 for i in range(len(categories))}
    
    for text, label in zip(data.data, data.target):
        if count[label] < limit_per_category:
            filtered_data['data'].append(text)
            filtered_data['target'].append(label)
            count[label] += 1
        if all(c >= limit_per_category for c in count.values()):
            break

    return filtered_data

def clean_text(text):
    """
    Basic preprocessing: lowercase, remove punctuation, stopwords
    """
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_corpus(corpus):
    return [clean_text(doc) for doc in corpus]

def vectorize_corpus(cleaned_corpus, max_features=3000):
    """
    Vectorize using TF-IDF
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(cleaned_corpus)
    return X, vectorizer
