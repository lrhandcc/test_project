from models.user import User


def create_user(db, user):
    new_user = User(
        username=user.username,
        age=user.age,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


def get_users(db):
    return db.query(User).all()


def get_user(db, user_id):
    return (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )


def update_user(db, user_id, user):
    db_user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if db_user is None:
        return None

    db_user.username = user.username
    db_user.age = user.age

    db.commit()
    db.refresh(db_user)

    return db_user


def delete_user(db, user_id):
    db_user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if db_user is None:
        return False

    db.delete(db_user)
    db.commit()

    return True