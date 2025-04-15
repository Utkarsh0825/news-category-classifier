import streamlit as st
import sys, os

# Add root directory to path so utils/ can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.predict_from_text import predict_news

# Streamlit page config
st.set_page_config(page_title="News Category Classifier", layout="centered")

# UI Title
st.title("📰 News Category Classifier")
st.markdown("Classify news articles into topics using machine learning models.\n")

# Text input
st.write("Enter a short news snippet below:")
user_input = st.text_area("✏️ Your News Snippet", height=150)

# Dropdown model choice
model_choice = st.selectbox("🧠 Choose Model", ["SVM", "Logistic Regression", "Naive Bayes"])

# Map dropdown label to internal model keys
model_key_map = {
    "SVM": "svm",
    "Logistic Regression": "logreg",
    "Naive Bayes": "nb"
}

# Predict Button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a news snippet to classify.")
    else:
        try:
            result = predict_news([user_input])[0]
            label = result[model_key_map[model_choice]]
            st.success(f"📢 **Predicted Category:** `{label}`")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --- Optional: Batch File Upload
st.markdown("---")
st.subheader("📂 Optional: Upload a .txt file with multiple news lines")
uploaded_file = st.file_uploader("Upload a TXT file (1 news snippet per line)", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    st.write(f"✅ Loaded {len(lines)} lines.")

    if st.button("📊 Predict All"):
        try:
            predictions = predict_news(lines)
            for i, pred in enumerate(predictions):
                st.write(f"**{i+1}.** {pred['text'][:80]}... → 🎯 `{pred[model_key_map[model_choice]]}`")
        except Exception as e:
            st.error(f"❌ Batch prediction failed: {e}")
