from django.shortcuts import render
import joblib
import os
import string
from nltk.corpus import stopwords

# ==============================
# 1️⃣ Load model and vectorizer safely
# ==============================

# Get the absolute base directory (points to: .../spamdetection)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and vectorizer using absolute paths
model_path = os.path.join(BASE_DIR, 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# ==============================
# 2️⃣ Optional: Simple text preprocessing
# (You can modify it based on how you cleaned text during training)
# ==============================
def transform_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)


# ==============================
# 3️⃣ Main view function
# ==============================
def home(request):
    if request.method == 'POST':
        message = request.POST.get('message')

        # Handle empty input
        if not message:
            return render(request, 'index.html', {'error': 'Please enter a message before checking.'})

        # Preprocess and predict
        transformed_msg = transform_text(message)
        vector_input = vectorizer.transform([transformed_msg]).toarray()
        prediction = model.predict(vector_input)[0]
        result = "Spam" if prediction == 1 else "Not Spam"

        return render(request, 'index.html', {'result': result, 'message': message})

    return render(request, 'index.html')