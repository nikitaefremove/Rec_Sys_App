from database import Base, SessionLocal, engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import desc


class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    session = SessionLocal()
    answer = []
    for post in session.query(Post)\
        .filter(Post.topic == 'business')\
        .order_by(desc(Post.id))\
        .limit(10).all():
        answer.append(post.id)
    print(answer)
