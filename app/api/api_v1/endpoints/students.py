from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.models import Student, Teacher, Exam
from app.schemas.student import StudentResponse, StudentCreate, StudentUpdate
from app.api.api_v1.deps import get_current_teacher

router = APIRouter()


@router.get("/", response_model=List[StudentResponse])
def read_students(
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Retrieve students who have exams graded by current teacher.
    """
    student_ids = (
        db.query(Exam.student_id)
        .filter(Exam.teacher_id == current_teacher.id)
        .distinct()
        .all()
    )
    student_ids = [sid[0] for sid in student_ids]  # unwrap tuples

    # Truy vấn student theo các ID đó
    students = (
        db.query(Student)
        .filter(Student.id.in_(student_ids))
        .offset(skip)
        .limit(limit)
        .all()
    )

    return students


@router.post("/", response_model=StudentResponse)
def create_student(
    *,
    db: Session = Depends(get_db),
    student_in: StudentCreate,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Create new student.
    """
    # Check if student code is already used
    student = db.query(Student).filter(Student.student_code == student_in.student_code).first()
    if student:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student code already registered",
        )
    
    # Create new student
    student = Student(**student_in.dict())
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@router.get("/{student_id}", response_model=StudentResponse)
def read_student(
    *,
    db: Session = Depends(get_db),
    student_id: int,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Get student by ID.
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )
    return student


@router.put("/{student_id}", response_model=StudentResponse)
def update_student(
    *,
    db: Session = Depends(get_db),
    student_id: int,
    student_in: StudentUpdate,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Update a student.
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )
    
    # Check if student code is already used by another student
    if student_in.student_code and student_in.student_code != student.student_code:
        existing_student = db.query(Student).filter(Student.student_code == student_in.student_code).first()
        if existing_student and existing_student.id != student_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student code already registered",
            )
    
    # Update student
    update_data = student_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(student, field, value)
    
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@router.delete("/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(
    *,
    db: Session = Depends(get_db),
    student_id: int,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> None:
    """
    Delete a student and all related exams.
    """
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )

    # Xóa tất cả các exam liên quan tới student này
    exams = db.query(Exam).filter(Exam.student_id == student_id).all()
    for exam in exams:
        if exam.submission_url:
            try:
                parsed = urlparse(exam.submission_url)
                path = parsed.path
                parts = path.split("/")
                upload_idx = parts.index("upload")
                public_id_with_ext = "/".join(parts[upload_idx + 2:])
                public_id = os.path.splitext(public_id_with_ext)[0]
                delete_from_cloudinary(public_id)
            except Exception as e:
                print(f"Warning: Failed to delete Cloudinary file for exam ID {exam.id}: {e}")
        
        db.delete(exam)

    # Sau đó xóa student
    db.delete(student)
    db.commit()
