# 🛒 Amazon Review Intelligence System

> An end-to-end NLP and Machine Learning system that analyses Amazon product reviews — predicting sentiment, detecting fake reviews, extracting keyword insights, and presenting everything through an interactive Streamlit dashboard.

**Built by Harmanpreet Kaur | CSE | Amazon ML Summer School 2026 Application**

---

## 📸 Screenshots

![Dashboard Overview](assets/s1.png)
![Sentiment Prediction](assets/s2.png)
![Dataset Insights](assets/s3.png)
![Word Clouds](assets/s4.png)

---

## 🎯 What It Does

| Feature | Description |
|---|---|
| **Sentiment Prediction** | Classifies any review as Positive / Negative with a confidence % |
| **Fake Review Detection** | Rule-based heuristic flags suspicious or bot-generated reviews |
| **Keyword Insights** | Extracts top complaint and praise words from 173,000 real reviews |
| **Word Clouds** | Visual keyword maps separately for positive and negative reviews |
| **Interactive Dashboard** | Full Streamlit web app with charts, history, and real-time prediction |
| **Prediction History** | Tracks and displays all predictions made in the session |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 92.85% |
| **Precision** | 93.75% |
| **Recall** | 98.23% |
| **F1 Score** | 95.94% |
| **Dataset Size** | 173,000 real Amazon reviews |
| **Model** | Logistic Regression with TF-IDF (10,000 features, bigrams) |

---

## 🧠 ML Pipeline

```
Raw Amazon Review Text
        ↓
Text Preprocessing
(lowercase → remove punctuation → remove stopwords → tokenization)
        ↓
TF-IDF Vectorization
(10,000 features, unigrams + bigrams)
        ↓
Logistic Regression Classifier
        ↓
Sentiment Label (Positive / Negative) + Confidence Score
        ↓
Fake Review Heuristic Check
(caps ratio, exclamation count, promotional language detection)
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Pandas / NumPy** | Data loading, cleaning, manipulation |
| **Scikit-learn** | TF-IDF vectorizer, Logistic Regression, evaluation metrics |
| **NLTK** | Stopword removal and text preprocessing |
| **WordCloud** | Visual keyword maps |
| **Plotly** | Interactive charts and confusion matrix |
| **Streamlit** | Web dashboard for real-time predictions |
| **Matplotlib** | Supporting visualisations |

---

## 📁 Project Structure

```
amazon_review_intelligence/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
├── assets/                   # Screenshots
│   ├── s1.png
│   ├── s2.png
│   ├── s3.png
│   └── s4.png
├── data/
│   └── generate_dataset.py   # Fallback dataset generator
├── model/
│   ├── train.py              # Full ML pipeline
│   ├── sentiment_model.pkl   # Trained model (auto-saved)
│   └── tfidf_vectorizer.pkl  # Fitted vectorizer (auto-saved)
└── utils/
    └── insights.py           # Keyword extraction utilities
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/hpk397/amazon-review-intelligence
cd amazon-review-intelligence

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Download NLTK data
python3 -c "import nltk; nltk.download('stopwords')"

# 4. Add dataset (download from Kaggle link below)
# Place as: data/amazon_reviews.json

# 5. Launch the app
python3 -m streamlit run app.py
```

The model trains automatically on first launch (~1-2 minutes for 173k reviews).

---

## 📌 Dataset

- **Source:** Amazon Cell Phones & Accessories Reviews
- **Size:** 173,000 reviews
- **Labels:** Rating 1–2 → Negative | Rating 4–5 → Positive | Rating 3 → Excluded
- **Download:** [Kaggle – Amazon Reviews (Abdallah Wagih)](https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews)

---

## 💡 Fake Review Detection

A rule-based heuristic scores each review for authenticity by checking:

- Excessive capitalisation (caps ratio > 40%)
- Excessive exclamation marks (3 or more)
- Very short review length (under 8 words)
- Repeated promotional language ("BUY NOW", "BEST EVER", "MUST HAVE")
- High word repetition

Returns a fake probability score from 0–100%.

---

## 📈 Key Learnings

- Text preprocessing significantly impacts model performance
- TF-IDF with bigrams captures phrase-level sentiment better than unigrams alone
- Logistic Regression outperforms more complex models on this task due to the linear separability of sentiment in TF-IDF space
- Class imbalance (148k positive vs 24k negative) affects recall on negative class

---

*Built by Harmanpreet Kaur | B.Tech CSE | Amazon ML Summer School 2026 Application*