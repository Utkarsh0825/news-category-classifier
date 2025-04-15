import joblib
from utils.preprocessing import preprocess_corpus

def load_models():
    svm = joblib.load('models/svm_model.pkl')
    logreg = joblib.load('models/logreg_model.pkl')
    nb = joblib.load('models/nb_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    with open('data/categories_used.txt') as f:
        categories = [line.strip() for line in f.readlines()]
    return svm, logreg, nb, vectorizer, categories

def predict_news(text_list):
    svm, logreg, nb, vectorizer, categories = load_models()
    cleaned = preprocess_corpus(text_list)
    X_new = vectorizer.transform(cleaned)

    predictions = []
    for i, text in enumerate(text_list):
        preds = {
            'text': text,
            'svm': categories[svm.predict(X_new[i])[0]],
            'logreg': categories[logreg.predict(X_new[i])[0]],
            'nb': categories[nb.predict(X_new[i])[0]],
        }
        predictions.append(preds)
    return predictions
