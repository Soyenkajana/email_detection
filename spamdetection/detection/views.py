from django.shortcuts import render
import joblib

# Load the saved model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def home(request):
    if request.method == 'POST':
        message = request.POST.get('message')

        # Safety check for empty input
        if not message:
            return render(request, 'index.html', {'error': 'Please enter a message before checking.'})

        # Transform and predict
        data = vectorizer.transform([message]).toarray()
        prediction = model.predict(data)[0]
        result = "Spam" if prediction == 1 else "Not Spam"

        return render(request, 'index.html', {'result': result, 'message': message})
    
    return render(request, 'index.html')


