from typing import List
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import desc
from sqlalchemy.sql.functions import count
from database import SessionLocal
from schema import UserGet, PostGet, FeedGet
from table_user import User
from table_post import Post
from table_feed import Feed

app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db


@app.get("/user/{id}", response_model=UserGet)
def get_user_id(id: int, db: Session = Depends(get_db)) -> UserGet:
    query_user_id = db.query(User).filter(User.id == id).one_or_none()
    if not query_user_id:
        raise HTTPException(404, "ID not found")
    return query_user_id


@app.get("/post/{id}", response_model=PostGet)
def get_post_id(id: int, db: Session = Depends(get_db)) -> PostGet:
    query_post_id = db.query(Post).filter(Post.id == id).one_or_none()
    if not query_post_id:
        raise HTTPException(404, "ID not found")
    return query_post_id


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = 10,  db: Session = Depends(get_db)) -> List[FeedGet]:
    query_user_feed = db.query(Feed) \
                        .filter(Feed.user_id == id) \
                        .order_by(desc(Feed.time)) \
                        .limit(limit) \
                        .all()
    return query_user_feed


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = 10,  db: Session = Depends(get_db)) -> List[FeedGet]:
    query_post_feed = db.query(Feed) \
                        .filter(Feed.post_id == id) \
                        .order_by(desc(Feed.time)) \
                        .limit(limit) \
                        .all()
    return query_post_feed


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_recommended_feed(id: int, limit: int = 10, db: Session = Depends(get_db)) -> List[PostGet]:
    query_rec_feed = db.query(Post) \
                        .select_from(Feed) \
                        .filter(Feed.action == "like") \
                        .join(Post) \
                        .group_by(Post.id) \
                        .order_by(desc(count(Post.id))) \
                        .limit(limit) \
                        .all()
    return query_rec_feed

