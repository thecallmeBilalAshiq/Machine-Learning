import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (only needs to be done once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Sample text
text = "I LOOOVE AI!!! It's sooo powerful :)"

# Step 1: Lowercase
text = text.lower()

# Step 2: Remove punctuation
text = re.sub(r'[^\w\s]', '', text)

# Step 3: Tokenize (split into words)
words = text.split()

# Step 4: Remove stopwords
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Step 5: Stemming
stemmer = PorterStemmer()
words = [stemmer.stem(word) for word in words]

# Output the result
print(words)