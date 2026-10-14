from fastapi import APIRouter, Depends, HTTPException
from schemas.user import UserCreate, UserResponse, UserCreateResponse, MessageResponse
from services.user_service import create_user, get_users, get_user, update_user, delete_user
from sqlalchemy.orm import Session

from database import get_db


router = APIRouter()


@router.post("/user", response_model=UserCreateResponse)
def create_user_api(
    user: UserCreate,
    db: Session = Depends(get_db)
):

    new_user = create_user(db, user)

    return {
        "message": "创建成功",
        "user": new_user
    }


@router.get("/users", response_model=list[UserResponse])
def get_users_api(
    db: Session = Depends(get_db)
):

    return get_users(db)


@router.get("/user/{user_id}", response_model=UserResponse)
def get_user_api(
    user_id: int,
    db: Session = Depends(get_db)
):

    user = get_user(db, user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="用户不存在"
        )

    return user

@router.put("/user/{user_id}", response_model=UserCreateResponse)
def update_user_api(
    user_id: int,
    user: UserCreate,
    db: Session = Depends(get_db)
):

    result = update_user(db, user_id, user)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="用户不存在"
        )

    return {
        "message": "修改成功",
        "user": result
    }


@router.delete("/user/{user_id}", response_model=MessageResponse)
def delete_user_api(
    user_id: int,
    db: Session = Depends(get_db)
):

    result = delete_user(db, user_id)

    if result is False:
        raise HTTPException(
            status_code=404,
            detail="用户不存在"
        )

    return {
        "message": "删除成功"
    }