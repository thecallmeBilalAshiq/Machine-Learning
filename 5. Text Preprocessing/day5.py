import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

text = "I LOOOVE AI!!! It's sooo powerful :)"

# Lowercase
text = text.lower()

# Remove punctuation
text = re.sub(r'[^\w\s]', '', text)

# Tokenize
words = text.split()

# Remove stopwords
words = [w for w in words if w not in stopwords.words('english')]

# Stemming
stemmer = PorterStemmer()
words = [stemmer.stem(w) for w in words]

print(words)
# Output: ['loov', 'ai', 'power']
