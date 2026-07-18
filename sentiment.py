import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# -----------------------------
# Dataset
# -----------------------------
data = {
    'Review': [

        # ---------------- Positive ----------------
        'This movie is amazing',
        'I loved this film',
        'Fantastic acting and story',
        'Best movie ever',
        'Wonderful experience',
        'Excellent screenplay',
        'Awesome movie',

        # Positive Movie Names
        'Harry Potter',
        'Jumanji',
        'Pirates of the Caribbean',
        'Avengers',

        # Positive Sentences
        'Harry Potter is fantastic',
        'Jumanji is awesome',
        'Pirates of the Caribbean is exciting',
        'Avengers is amazing',

        # ---------------- Negative ----------------
        'Worst movie ever',
        'I hated this film',
        'Terrible acting',
        'Very boring movie',
        'Waste of time',
        'Bad screenplay',

        # Negative Movie Names
        'ABC',
        'DEF',
        '123',

        # Negative Sentences
        'ABC movie is bad',
        'DEF is terrible',
        '123 movie is worst',

        # ---------------- Neutral ----------------
        'Movie was okay',
        'Average film',
        'Nothing special',
        'It was fine',
        'Normal experience',

        # Neutral Movie Names
        'Movie xyz',
        'movie',
        'movies',

        # Neutral Sentences
        'Movie xyz is average',
        'movie is okay',
        'movies are normal'
    ],

    'Sentiment': [

        # Positive
        'Positive',
        'Positive',
        'Positive',
        'Positive',
        'Positive',
        'Positive',
        'Positive',

        # Positive Movie Names
        'Positive',
        'Positive',
        'Positive',
        'Positive',

        # Positive Sentences
        'Positive',
        'Positive',
        'Positive',
        'Positive',

        # Negative
        'Negative',
        'Negative',
        'Negative',
        'Negative',
        'Negative',
        'Negative',

        # Negative Movie Names
        'Negative',
        'Negative',
        'Negative',

        # Negative Sentences
        'Negative',
        'Negative',
        'Negative',

        # Neutral
        'Neutral',
        'Neutral',
        'Neutral',
        'Neutral',
        'Neutral',

        # Neutral Movie Names
        'Neutral',
        'Neutral',
        'Neutral',

        # Neutral Sentences
        'Neutral',
        'Neutral',
        'Neutral'
    ]
}

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame(data)

# -----------------------------
# Split Dataset
# -----------------------------
X = df['Review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Convert Text to Numbers
# -----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english'
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()

model.fit(X_train_vectors, y_train)

# -----------------------------
# Accuracy
# -----------------------------
y_pred = model.predict(X_test_vectors)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# User Prediction
# -----------------------------
while True:

    user_review = input("\nEnter movie name/review (type 'exit' to stop): ")

    if user_review.lower() == 'exit':
        print("Program Ended")
        break

    # Convert input
    user_vector = vectorizer.transform([user_review])

    # Predict
    prediction = model.predict(user_vector)

    print("Predicted Sentiment:", prediction[0])
