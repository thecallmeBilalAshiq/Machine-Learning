import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required resources if not already available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Sample dataset: Simulated customer reviews
reviews = [
    "I absolutely LOOOVE this product!!! It's sooo good and powerful. 💯💯💯",
    "Not satisfied at all... delivery was late, and packaging was broken!",
    "Awesome experience. Very user-friendly and responsive.",
    "Worst product ever. Waste of money!",
    "Sooo many bugs in the app, it's frustrating :(",
    "Great customer support! Resolved my issue within minutes.",
    "The interface looks outdated but still functions well.",
    "Couldn’t believe how FAST the response was! 😍",
    "Disappointed. I had high hopes but it failed to perform.",
    "Five stars!!! ⭐⭐⭐⭐⭐ Highly recommended!"
]

# Preprocessing steps for each review
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

preprocessed_reviews = []

for review in reviews:
    # Lowercase
    review = review.lower()
    
    # Remove punctuation/symbols/emojis
    review = re.sub(r'[^\w\s]', '', review)
    
    # Tokenize
    words = review.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    words = [stemmer.stem(word) for word in words]
    
    # Store the result
    preprocessed_reviews.append(words)

# Print the preprocessed output
for i, processed in enumerate(preprocessed_reviews):
    print(f"Review {i+1} Processed:", processed)
