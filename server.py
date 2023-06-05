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
        course_name = row['Course Name']
        if course_name not in combined_recommendations:
            combined_recommendations[course_name] = {
                'Course Name': course_name,
                'Course Rating': [],
                'Difficulty Level': [],
                'Course URL': [],
                'Course Description': [],
                'University': row['University']
            }
        combined_recommendations[course_name]['Course Rating'].append(row['Course Rating'])
        combined_recommendations[course_name]['Difficulty Level'].append(row['Difficulty Level'])
        combined_recommendations[course_name]['Course URL'].append(row['Course URL'])
        combined_recommendations[course_name]['Course Description'].append(row['Course Description'])

    # Create a list of the combined recommendations in the desired order
    final_recommendations = []
    for course_name, details in combined_recommendations.items():
        final_recommendations.append({
            'Course Name': course_name,
            'Course Rating': details['Course Rating'],
            'Difficulty Level': details['Difficulty Level'],
            'Course URL': details['Course URL'],
            'Course Description': details['Course Description'],
            'University': details['University']
        })

    print(final_recommendations)
    return jsonify(final_recommendations)


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
