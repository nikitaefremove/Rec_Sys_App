from typing import List
from fastapi import FastAPI, HTTPException
import pandas as pd
from sqlalchemy import create_engine
import os
from catboost import CatBoostClassifier
from datetime import datetime
from pydantic import BaseModel


class PostGet(BaseModel):
    post_id: int
    text: str
    topic: str

    class Config:
        from_attributes = True


app = FastAPI()


# Load catboost_model
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # Checking localhost or LMS service
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/Users/nikitaefremov/Documents/DATA_SCIENCE/SML_ML/REC_SYS/catboost_model")
    model = CatBoostClassifier().load_model('catboost_model', format='cbm')
    return model


model = load_models()


### Load features from database

df1 = pd.read_csv('nikita_efremov_user_features_df.csv')
df2 = pd.read_csv('nikita_efremov_post_features_df.csv')


### Load post_text_df dataframe

post_text_df = pd.read_csv('post_text_df.csv')


### Function for prediction

def prediction_top_5_posts(user_features_df, post_features_df, user_id, model):
    ## Save the place for features is important for model
    places_for_features_columns = ['user_id', 'post_id', 'gender', 'age', 'country', 'city',
                                   'exp_group', 'os', 'source', 'count_actions', 'category_of_age',
                                   'cluster_feature', 'month', 'day', 'second', 'weekday', 'is_weekend',
                                   'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
                                   'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
                                   'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15',
                                   'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20',
                                   'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25',
                                   'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30',
                                   'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35',
                                   'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40',
                                   'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45',
                                   'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50',
                                   'part_of_day', 'topic']

    # Create copy of dataframes and find the data of this user
    this_user_data = user_features_df.copy().loc[user_features_df['user_id'] == user_id]
    all_post_features_df = post_features_df.copy()

    # Merge dataframes on key column
    this_user_data['key'] = 1
    all_post_features_df['key'] = 1
    result = this_user_data.merge(all_post_features_df, on='key').drop('key', axis=1)
    result = result[places_for_features_columns].set_index(['user_id', 'post_id'])
    result['prediction'] = model.predict_proba(result)[:, 1]
    top_5_posts = result.sort_values('prediction', ascending=False).head(5).index.get_level_values('post_id').tolist()
    return top_5_posts


### Endpoints


### Get information about post by post_id
@app.get("/post/{id}", response_model=PostGet)
def get_post_id(id: int) -> PostGet:
    post_record = post_text_df[post_text_df['post_id'] == id]

    if post_record.empty:
        raise HTTPException(404, "ID not found")

    post_dict = post_record.iloc[0].to_dict()
    print(post_dict)  # Add this print statement
    return post_dict


### Get 5 recommendation of post to user
@app.get("/post/recommendations/{id}", response_model=List[PostGet])
def recommended_posts(id: int, limit: int=5) -> List[PostGet]:
    top_5_posts_ids = prediction_top_5_posts(df1, df2, id, model)

    # Filter top 5 posts from post_texts_df DataFrame
    posts = post_text_df[post_text_df['post_id'].isin(top_5_posts_ids)]

    if len(posts) != 5:
        raise HTTPException(404, "Some recommended posts not found")

    return posts.to_dict('records')

