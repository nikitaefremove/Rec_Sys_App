# Recommendation System App

## Description

This app is a recommendation system built using FastAPI and CatBoost. 
It provides post recommendations to users based on their personal information and post data, using content-based recommendation approach.
The application interfaces with a PostgreSQL database to fetch features for users and posts 
and then uses a trained CatBoost model to predict the top 5 recommendations for a user.
For training model, and creating features,
I analyzed tables containing users, posts, and the history of users who viewed or liked the posts. 
Next, I generated features for all users using popular and mean values by grouping them into clusters 
formed using K-Means and One-Hot Encoding (OHE).
For the posts table, I created features using the TF-IDF approach and then grouped them using Principal Component Analysis (PCA).
Finally, I wrote a function that merges the dataset with user and post features to predict the top 5 recommended posts for a given user.

## Features

1. **Loading the CatBoost Model**: The model is trained externally and saved. The app can dynamically determine the model's path based on environment variables.
2. **Batch Loading from PostgreSQL**: Efficiently loads large amounts of data from the PostgreSQL database using chunked data retrieval.
3. **Endpoints**:
   - **/post/recommendations/**: Returns the top 5 recommended posts for a given user.

## Installation and Setup

### Dependencies:
```
fastapi==0.75.1
pandas==1.4.2
sqlalchemy==1.4.35
requests==2.27.1
catboost==1.2
numpy==1.25.2
pydantic==1.9.1
scikit_learn==1.3.0
psycopg2-binary==2.9.7
uvicorn==0.16.0
category-encoders==2.5.0
loguru==0.6.0
implicit==0.7.0
lightfm==1.17
datetime==5.2
nltk==3.8.1 
```

### Running the app:

Navigate to the directory containing the app and run:
```
uvicorn <filename>:app --reload
```
Replace <filename> with the name of the file that contains the FastAPI application.

### Usage

To get the top 5 recommended posts for a user:
```
GET /post/recommendations/?id=<USER_ID>
```
Replace <USER_ID> with the desired user's ID.



