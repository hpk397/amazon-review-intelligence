"""
Insight utilities: keyword extraction, word frequency analysis.
"""

import re
from collections import Counter
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

EXTRA_STOP = {
    "product", "item", "thing", "one", "use", "used", "using", "get",
    "got", "like", "would", "really", "also", "even", "much", "well",
    "good", "great", "very", "quite", "just", "still", "bought",
    "purchase", "order", "ordered", "received", "delivery", "amazon",
}
ALL_STOP = STOP_WORDS | EXTRA_STOP


def extract_keywords(text: str, top_n: int = 10) -> list[tuple[str, int]]:
    """Return top N meaningful words from text."""
    words = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    words = [w for w in words if w not in ALL_STOP and len(w) > 3]
    return Counter(words).most_common(top_n)


def get_sentiment_keywords(reviews: list[str], top_n: int = 15) -> list[tuple[str, int]]:
    """Aggregate top keywords across a list of reviews."""
    all_words = []
    for review in reviews:
        words = re.sub(r"[^a-z\s]", " ", review.lower()).split()
        all_words.extend([w for w in words if w not in ALL_STOP and len(w) > 3])
    return Counter(all_words).most_common(top_n)
