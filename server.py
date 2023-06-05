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

    # Combine similar keys in each object into a new object
    combined_recommendations = {}
    for index, row in recommended_courses.iterrows():
        course_id = row['Course ID']
        if course_id not in combined_recommendations:
            combined_recommendations[course_id] = {
                'Course ID': course_id,
                'Course Name': row['Course Name'],
                'University': row['University'],
                'Difficulty Level': [],
                'Rating': [],
                'URL': [],
                'Description': []
            }
        combined_recommendations[course_id]['Difficulty Level'].append(row['Difficulty Level'])
        combined_recommendations[course_id]['Rating'].append(row['Rating'])
        combined_recommendations[course_id]['URL'].append(row['URL'])
        combined_recommendations[course_id]['Description'].append(row['Description'])

    # Create a list of the combined recommendations
    final_recommendations = list(combined_recommendations.values())

    print(final_recommendations)
    return jsonify(final_recommendations)


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
