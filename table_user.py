from database import Base, SessionLocal, engine
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import desc
from sqlalchemy.sql.functions import count


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    os = Column(String)
    source = Column(String)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    session = SessionLocal()

    query = session.query(User.country, User.os, count().label('count')) \
        .filter(User.exp_group == 3) \
        .group_by(User.country, User.os) \
        .order_by(desc(count())) \
        .having(count() > 100)

    results = [(i.country, i.os, i.count) for i in query.all()]

    print(results)
