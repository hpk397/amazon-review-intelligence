"""
Generate a realistic Amazon reviews dataset for training.
In production, replace with: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
"""

import pandas as pd
import random

random.seed(42)

POSITIVE_REVIEWS = [
    "This product is absolutely amazing! Works perfectly.", "Great quality, fast shipping, very happy!",
    "Exceeded my expectations. Would definitely buy again.", "Fantastic product, exactly as described.",
    "Super easy to use and works like a charm.", "Best purchase I've made this year!",
    "Really good quality for the price. Highly recommend.", "Arrived on time, packaging was great.",
    "Love this product! Using it every single day now.", "Outstanding quality and very durable.",
    "Five stars! This is exactly what I needed.", "Works great, setup was easy, no issues at all.",
    "Very satisfied with this purchase. Good value.", "Solid product, well built, highly recommend.",
    "Product looks exactly like in the photos. Love it!", "Great item, very fast delivery, excellent!",
    "Impressed with the build quality, feels premium.", "Bought this for my family, everyone loves it.",
    "Worth every penny. Top notch quality.", "Perfect size, perfect quality, very happy.",
    "This works so well, super glad I bought it.", "Beautifully made and works as expected.",
    "Amazing value for money. Totally worth it.", "Highly recommend this to anyone looking for quality.",
    "Good product, arrived well packaged and on time.", "Great functionality and very easy to set up.",
    "Excellent customer service and a brilliant product.", "So happy with this purchase. Will buy again!",
    "Fantastic! Does exactly what it says it will do.", "Very good product. Exceeded my expectations.",
    "Love the design and it works flawlessly.", "Brilliant product at a reasonable price.",
    "This is my second purchase and it's just as good.", "Quick delivery and a very well made product.",
    "Happy with the purchase. Good quality overall.", "Works better than I expected, great buy.",
    "Elegant design and very functional. Impressed!", "Durable and well-made. Worth every rupee.",
    "Loved the product. Family is very happy with it.", "Performs exactly as described. Five stars!"
]

NEGATIVE_REVIEWS = [
    "Terrible product. Stopped working after 2 days.", "Very poor quality. Not worth the money.",
    "Completely disappointed. Nothing like in the photos.", "Broke down in a week. Total waste of money.",
    "Poor build quality, feels very cheap.", "Does not work as advertised. Very misleading.",
    "Had to return this, it was defective out of the box.", "Very disappointed. Expected much better quality.",
    "Would not recommend. Poor value for money.", "Worst purchase ever. Complete junk.",
    "Battery life is terrible, barely lasts an hour.", "Stopped working after the first use. Useless.",
    "The material feels flimsy and low quality.", "Packaging was damaged, product was broken.",
    "Instructions were unclear and product didn't work.", "Very cheap feeling. Not durable at all.",
    "Returned this immediately. Didn't match description.", "Customer service was unhelpful and rude.",
    "Do not buy this. It is a complete waste of money.", "Extremely fragile. Broke within the first week.",
    "Product is nothing like in the pictures. Refund requested.", "Very slow delivery, product was damaged.",
    "Overpriced for the quality received. Disappointed.", "Doesn't fit as described. Poor sizing.",
    "Felt cheaply made. Expected better for the price.", "Absolute rubbish. Would give zero stars.",
    "This is not what was advertised. Very misleading.", "Ordered this twice, both times defective.",
    "Waste of money. Not functional at all.", "Very bad smell when unboxed. Not acceptable.",
    "Totally useless product. Cannot recommend.", "Doesn't do what it claims. False advertising.",
    "Broke on the first day of use. Very poor.", "No quality control on this product at all.",
    "Battery doesn't charge properly. Major defect.", "Looks good but doesn't work. Avoid this.",
    "Not durable at all. Broke very quickly.", "Very poorly made. Would not buy again.",
    "Product has a manufacturing defect. Very annoying.", "Absolutely horrible. Return process was also bad.",
]

FAKE_PATTERNS = [
    "BEST PRODUCT EVER BUY NOW!!!!! AMAZING!!!!",
    "WOW THIS IS THE GREATEST PURCHASE I EVER MADE BUY BUY BUY",
    "INCREDIBLE!!!! YOU MUST BUY THIS!!!! FIVE STARS NO DOUBT!!!!",
    "I LOVE THIS SO MUCH EVERYONE SHOULD BUY THIS TODAY!!!!!",
    "5 stars no doubt best seller must have product of the year",
]

rows = []
for _ in range(2000):
    rows.append({"reviewText": random.choice(POSITIVE_REVIEWS), "overall": random.choice([4, 5])})
for _ in range(2000):
    rows.append({"reviewText": random.choice(NEGATIVE_REVIEWS), "overall": random.choice([1, 2])})
for pat in FAKE_PATTERNS * 20:
    rows.append({"reviewText": pat, "overall": 5})

random.shuffle(rows)
df = pd.DataFrame(rows)
df.to_csv("/home/claude/amazon_review_intelligence/data/amazon_reviews.csv", index=False)
print(f"Dataset created: {len(df)} reviews")
