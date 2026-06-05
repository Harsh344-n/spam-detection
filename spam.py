from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    "Congratulations you won a prize",
    "Claim your free money now",
    "Meeting at 5 PM tomorrow",
    "Let's study together",
    "You have won a lottery",
    "Project submission is tomorrow"
]

labels = [
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Spam",
    "Not Spam"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(messages)

model = MultinomialNB()

model.fit(X, labels)

test_message = ["meeting at 10 am tomorrow"]

test_data = vectorizer.transform(test_message)

prediction = model.predict(test_data)

print("Prediction:", prediction[0])