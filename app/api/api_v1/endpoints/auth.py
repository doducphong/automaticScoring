from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.config import settings
from app.core.database import get_db
from app.models.models import Teacher
from app.schemas.token import Token
from app.schemas.teacher import TeacherCreate, TeacherResponse

router = APIRouter()


@router.post("/register", response_model=TeacherResponse)
def register_teacher(*, db: Session = Depends(get_db), teacher_in: TeacherCreate) -> Any:
    """
    Register a new teacher account.
    """
    # Check if email is already taken
    teacher = db.query(Teacher).filter(Teacher.email == teacher_in.email).first()
    if teacher:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create new teacher
    hashed_password = get_password_hash(teacher_in.password)
    db_obj = Teacher(
        email=teacher_in.email,
        name=teacher_in.name,
        password=hashed_password,
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj


@router.post("/login", response_model=Token)
def login_access_token(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    # Authenticate teacher
    teacher = db.query(Teacher).filter(Teacher.email == form_data.username).first()
    if not teacher or not verify_password(form_data.password, teacher.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            subject=str(teacher.id), expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }
