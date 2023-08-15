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