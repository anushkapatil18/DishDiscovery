from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Read the recipe dataset from a CSV file
recipe_data = pd.read_csv("dataset1.csv")

# Preprocess recipe ingredients: convert to lowercase and remove unwanted characters
recipe_data["IngredientsProcessed"] = recipe_data["IngredientsRaw"].fillna("").astype(str).str.lower().apply(
    lambda x: re.sub(r'[^a-z\s]', '', x))

# Apply TF-IDF vectorization to convert text into numerical features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(recipe_data["IngredientsProcessed"])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommend():
    ingredients = request.form.get('ingredients')

    if ingredients is None:
        return "Invalid request. Please provide ingredients."

    ingredients = ingredients.split(',')

    # Preprocess user ingredients: convert to lowercase and remove unwanted characters
    user_ingredients_processed = [re.sub(r'[^a-z\s]', '', ingredient.lower()) for ingredient in ingredients]

   # Calculate the cosine similarity between the user's ingredient vector and all recipes
    user_vector = vectorizer.transform([" ".join(user_ingredients_processed)])
    cosine_similarities = linear_kernel(user_vector, tfidf_matrix)[0]

    # Calculate the combined similarity scores
    combined_scores =  cosine_similarities

    # Get indices of recipes sorted by sum of combined similarity scores
    similar_recipe_indices = combined_scores.argsort()[::-1]

    # Recommend recipes based on combined similarity
    num_recommendations =  5  # Specify the number of recommendations needed

    recommendations = recipe_data.loc[similar_recipe_indices.tolist()][:num_recommendations]

    recommendations_df = pd.DataFrame(recommendations)

    # Prepare the data to be sent to the template
    recipe_names = []
    ingredients_needed = []
    total_time = []
    avg_rating = []

    for _, recipe in recommendations_df.iterrows():
        recipe_names.append(recipe['Title'])
        ingredients_needed.append(recipe["IngredientsRaw"].replace('#item', ' '))
        total_time.append(recipe["TotalTime"])
        if pd.isnull(recipe["AvgRating"]):
            avg_rating.append("Not rated")
        else:
            avg_rating.append(recipe["AvgRating"])

    return render_template('recommend.html', recipe_names=recipe_names, ingredients=ingredients_needed,total_time=total_time, avg_rating=avg_rating)


if __name__ == '__main__':
    app.run(debug=True)
