from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import RS

df = RS.load_data('Coursera.csv')
RS.clean_data(df)
print("loading data")

app = Flask(__name__)
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_interests = data["interests"]
    recommended_courses = RS.recommend_courses(user_interests, df)
    print(recommended_courses)
    return recommended_courses.to_json()


# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
