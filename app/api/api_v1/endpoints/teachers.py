from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Teacher
from app.schemas.teacher import TeacherResponse, TeacherUpdate
from app.api.api_v1.deps import get_current_teacher

router = APIRouter()


@router.get("/", response_model=List[TeacherResponse])
def read_teachers(
    db: Session = Depends(get_db),
    skip: int = 0, 
    limit: int = 100,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Retrieve all teachers.
    """
    teachers = db.query(Teacher).offset(skip).limit(limit).all()
    return teachers


@router.get("/me", response_model=TeacherResponse)
def read_current_teacher(
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Get current teacher information.
    """
    return current_teacher


@router.put("/me", response_model=TeacherResponse)
def update_teacher(
    *,
    db: Session = Depends(get_db),
    teacher_in: TeacherUpdate,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Update teacher information.
    """
    from app.core.security import get_password_hash
    
    # Update only provided fields
    data_to_update = teacher_in.dict(exclude_unset=True)
    
    # Hash password if provided
    if "password" in data_to_update:
        data_to_update["password"] = get_password_hash(data_to_update["password"])
    
    # Check if email is already taken
    if "email" in data_to_update and data_to_update["email"] != current_teacher.email:
        db_teacher = db.query(Teacher).filter(Teacher.email == data_to_update["email"]).first()
        if db_teacher:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
    
    # Update teacher
    for field, value in data_to_update.items():
        setattr(current_teacher, field, value)
    
    db.add(current_teacher)
    db.commit()
    db.refresh(current_teacher)
    
    return current_teacher
