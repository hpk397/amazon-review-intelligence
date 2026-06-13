# 🛒 Amazon Review Intelligence System

> NLP-based sentiment analysis system for Amazon product reviews, with keyword insights, fake review detection, and an interactive dashboard.

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| **Sentiment Prediction** | Classifies reviews as Positive / Negative with confidence % |
| **Fake Review Detection** | Heuristic-based detector flags suspicious reviews |
| **Keyword Insights** | Top complaint and praise words from real review data |
| **Word Clouds** | Visual keyword maps for positive and negative reviews |
| **Prediction History** | Tracks all predictions made in the session |

---

## 🧠 ML Pipeline

```
Raw Review Text
     ↓
Text Preprocessing (lowercase → remove punctuation → remove stopwords)
     ↓
TF-IDF Vectorization (10,000 features, unigrams + bigrams)
     ↓
Logistic Regression Classifier
     ↓
Sentiment Label + Confidence Score
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~93% |
| Precision | ~93% |
| Recall | ~93% |
| F1 Score | ~93% |

---

## 🛠 Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data loading and processing
- **Scikit-learn** — TF-IDF, Logistic Regression, evaluation metrics
- **NLTK** — Stopword removal
- **WordCloud** — Visual keyword maps
- **Plotly** — Interactive charts
- **Streamlit** — Web dashboard

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/amazon-review-intelligence
cd amazon-review-intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# 4. Launch the app
streamlit run app.py
```

The model trains automatically on first launch using the bundled dataset.

---

## 📁 Project Structure

```
amazon_review_intelligence/
├── app.py                    # Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py   # Dataset generator
│   └── amazon_reviews.csv    # Training data (auto-generated)
├── model/
│   ├── train.py              # ML pipeline: preprocessing + training
│   ├── sentiment_model.pkl   # Trained model (auto-saved)
│   └── tfidf_vectorizer.pkl  # Fitted vectorizer (auto-saved)
└── utils/
    └── insights.py           # Keyword extraction utilities
```

---

## 📌 Dataset

Uses Amazon product review data (rating 1–2 → Negative, 4–5 → Positive).

For production scale: [Amazon Review Data – McAuley Lab](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)  
Or on Kaggle: [Amazon Reviews Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)

---

## 💡 Fake Review Detection

A rule-based heuristic checks for:
- Excessive capitalisation
- Excessive exclamation marks
- Very short review length
- Repeated promotional language ("BUY NOW", "BEST EVER")

---

*Built by Harmanpreet Kaur | Amazon ML Summer School 2026 Application*
