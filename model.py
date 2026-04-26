import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Select needed columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Input and output
x = data['message']
y = data['label']

# Convert text to numbers
cv = CountVectorizer()
x = cv.fit_transform(x)

# Train model
model = MultinomialNB()
model.fit(x, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("Model trained successfully")