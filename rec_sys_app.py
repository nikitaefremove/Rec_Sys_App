from typing import List
from fastapi import Depends, FastAPI, HTTPException
import pandas as pd
from sqlalchemy import create_engine
import os
from catboost import CatBoostClassifier
import datetime
from sqlalchemy.orm import Session
from sqlalchemy.sql import desc
from sqlalchemy.sql.functions import count
from database import SessionLocal
from schema import UserGet, PostGet, FeedGet
from table_user import User
from table_post import Post
from table_feed import Feed
from schema import PostGet

app = FastAPI()


### Load catboost_model
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # Checking localhost or LMS service
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("/Users/nikitaefremov/Documents/DATA_SCIENCE/SML_ML/REC_SYS/catboost_model")
    model = CatBoostClassifier().load_model(model_path, format='cbm')
    return model


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


def load_features() -> pd.DataFrame:
    query = 'SELECT * FROM nikita_efremov_feature'
    return batch_load_sql(query)


### Endpoint
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:

