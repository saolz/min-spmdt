import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your dataset and prepare the model (replace with your data and model)
data = pd.read_csv("md.csv")  # Replace with your dataset path
X = data['Message']  # Replace 'email_column' with the actual column name
y = data['Category']  # Assuming 'label' is the column containing spam/ham labels

# Convert labels to 1 for spam and 0 for ham (if needed)
y = (y == 'spam').astype(int)

# Use TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train a Support Vector Machine (SVM) classifier
model = SVC(C=1.0, kernel='linear', gamma='scale')
model.fit(X_tfidf, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_input = request.form['email']
        X_user = tfidf_vectorizer.transform([user_input])
        prediction = model.predict(X_user)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

