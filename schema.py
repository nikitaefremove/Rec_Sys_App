import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserGet(BaseModel):
    id : int
    age : int
    city : str
    country : str
    exp_group : int
    gender : int
    os : str
    source : str
    class Config:
        from_attributes = True


class PostGet(BaseModel):
    id : int
    text : str
    topic : str
    class Config:
        from_attributes = True


class FeedGet(BaseModel):
    action : str
    post_id : int
    time : datetime.datetime
    user_id : int
    user: UserGet
    post: PostGet
    class Config:
        from_attributes = True
