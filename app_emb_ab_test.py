from typing import List
from fastapi import FastAPI
import pandas as pd
from sqlalchemy import create_engine
import os
from catboost import CatBoostClassifier
from datetime import datetime
from schema import PostGet

app = FastAPI()


# Load catboost_model
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # Checking localhost or LMS service
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path(
        "/Users/nikitaefremov/Documents/DATA_SCIENCE/SML_ML/Rec_Sys_App/Rec_Sys_App/catboost_model")
    model = CatBoostClassifier().load_model(model_path, format='cbm')
    return model


model = load_models()


### Load features from database

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features():
    query1 = 'SELECT * FROM nikita_efremov_user_features_df'
    query2 = 'SELECT * FROM nikita_efremov_post_features_df_emb'  # features with embeddings from Bert
    query3 = 'SELECT * FROM nikita_efremov_post_features_df_emb'  # features with TFIDF
    return batch_load_sql(query1), batch_load_sql(query2), batch_load_sql(query3)


df1, df2, df3 = load_features()


### Load post_text_df dataframe

def load_post_text_df() -> pd.DataFrame:
    query = 'SELECT * FROM public.post_text_df'
    return batch_load_sql(query)


def load_post_texts(post_ids: List[int]) -> List[dict]:
    global post_texts_df
    if post_texts_df is None:
        raise ValueError("First call load_post_texts_df().")

    records_df = post_texts_df[post_texts_df['post_id'].isin(post_ids)]
    return records_df.to_dict("records")


post_text_df = load_post_text_df()


### Function for prediction

def prediction_top_5_posts(user_features_df, post_features_df, user_id, model):
    ## Save the place for features is important for model
    places_for_features_columns = ['user_id', 'post_id', 'gender', 'age', 'country', 'city',
                                   'exp_group', 'os', 'source', 'count_actions', 'category_of_age',
                                   'cluster_feature', 'month', 'day', 'second', 'weekday', 'is_weekend',
                                   'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
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


# Function for getting top recommended posts with TFIDF features (model_control)
def recommended_posts_train(
        id: int,
        time: datetime = datetime.now(),
        limit: int = 10) -> List[PostGet]:
    top_5_posts_ids = prediction_top_5_posts(df1, df3, id, model)

    # Filter top 5 posts from post_texts_df DataFrame
    posts = post_text_df[post_text_df['post_id'].isin(top_5_posts_ids)]

    # Convert posts to list of dictionaries and ensure they match PostGet model
    posts_list = []
    for _, row in posts.iterrows():
        post_dict = row.to_dict()
        post_dict["id"] = post_dict.pop("post_id")
        if "Unnamed: 0" in post_dict:
            del post_dict["Unnamed: 0"]
        posts_list.append(PostGet(**post_dict))

    return posts_list


# Function for getting top recommended posts with Bert features (model_test)
def recommended_posts_test(
        id: int,
        time: datetime = datetime.now(),
        limit: int = 10) -> List[PostGet]:
    top_5_posts_ids = prediction_top_5_posts(df1, df2, id, model)

    # Filter top 5 posts from post_texts_df DataFrame
    posts = post_text_df[post_text_df['post_id'].isin(top_5_posts_ids)]

    # Convert posts to list of dictionaries and ensure they match PostGet model
    posts_list = []
    for _, row in posts.iterrows():
        post_dict = row.to_dict()
        post_dict["id"] = post_dict.pop("post_id")
        if "Unnamed: 0" in post_dict:
            del post_dict["Unnamed: 0"]
        posts_list.append(PostGet(**post_dict))

    return posts_list


### Endpoints

### Get 5 recommendation of post to user
@app.get("/post/recommendations/", response_model=List[PostGet])
def rec_post(id: int) -> List[PostGet]:
    return recommended_posts_train(id)


print(rec_post(id=205))
