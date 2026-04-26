import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Get user input
message = input("Enter message: ")

# Convert input
data = cv.transform([message])

# Predict
prediction = model.predict(data)

# Show result
if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")