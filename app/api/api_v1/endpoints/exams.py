from typing import Any, List, Dict
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import ValidationError

from app.core.database import get_db
from app.core.config import settings
from app.models.models import Exam, Student, Teacher
from app.schemas.exam import ExamResponse, ExamWithRelations, ExamCreate, ExamUpdate
from app.api.api_v1.deps import get_current_teacher
from app.services.scorer.evaluate_docx_submission import evaluate_submission, compute_final_score
from app.services.scorer.evaluate_xlsx_submission import evaluate_excel_result, load_excel_result
from app.utils.cloudinary_utils import upload_file_to_cloudinary, delete_from_cloudinary
from sqlalchemy.orm import joinedload
from app.api.api_v1 import deps
import traceback
router = APIRouter()


@router.get("/", response_model=List[ExamWithRelations])
def read_exams(
    db: Session = Depends(deps.get_db),
    current_teacher: Teacher = Depends(get_current_teacher),
    ):
    exams = db.query(Exam).options(
        joinedload(Exam.student),
        joinedload(Exam.teacher)
    ).filter(Exam.teacher_id == current_teacher.id).all()

    return [
        ExamWithRelations(
            **ExamResponse.from_orm(exam).dict(),
            student_name=exam.student.name if exam.student else "",
            student_code=exam.student.student_code if exam.student else "",
            teacher_name=exam.teacher.name if exam.teacher else None,
        )
        for exam in exams
    ]


@router.post("/grade-docx", response_model=ExamResponse)
async def grade_docx_exam(
    *,
    db: Session = Depends(get_db),
    sample_file: UploadFile = File(...),
    submission_file: UploadFile = File(...),
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Grade a docx file submission against a sample file.
    This endpoint will:
    1. Extract student info from submission
    2. Find or create student record
    3. Grade the document and create exam record
    """
    # Validate file formats
    if not sample_file.filename.endswith(".docx") or not submission_file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both files must be in .docx format",
        )
    
    # Create unique filenames for uploaded files
    sample_path = Path(settings.UPLOAD_DIR) / f"{uuid4()}-sample.docx"
    submission_path = Path(settings.UPLOAD_DIR) / f"{uuid4()}-submission.docx"
    
    exam_code = None
    try:
        # Save uploaded files
        with open(sample_path, "wb") as buffer:
            shutil.copyfileobj(sample_file.file, buffer)
        
        with open(submission_path, "wb") as buffer:
            shutil.copyfileobj(submission_file.file, buffer)
        
        # Run evaluation
        grading_result = evaluate_submission(str(sample_path), str(submission_path))
        submission_filename = Path(submission_file.filename).stem
        # Extract student info (attempt 1: from file content)
        info = grading_result.get("information_studen")
        student_code, student_name = None, None
        birth_date = datetime.now().date()
        place_of_birth = None

        if info and len(info) >= 2:
            student_code = info[0]
            student_name = info[1]
            if len(info) > 2:
                try:
                    birth_date = datetime.strptime(info[2], "%d/%m/%Y").date()
                except Exception:
                    birth_date = datetime.now().date()
            if len(info) > 3:
                place_of_birth = info[3]
            if len(info) > 4:
                exam_code = info[4]
        else:
            # Attempt 2: extract from submission filename
            clean_name = submission_filename.replace(" ", "_")
            student_name = clean_name
            student_code = clean_name
            exam_code = Path(sample_file.filename).stem

        if not student_name or not student_code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot extract student information from submission or filename.",
            )
        
        # Find or create student
        student = db.query(Student).filter(Student.student_code == student_code).first()
        if not student:
            student = Student(
                student_code=student_code,
                name=student_name,
                birth_date=birth_date,
                place_of_birth=place_of_birth,
            )
            db.add(student)
            db.commit()
            db.refresh(student)
        else:
            # Check if DOCX exam already exists
            existing_exam = db.query(Exam).filter(
                Exam.student_id == student.id,
                Exam.file_format == "docx"
            ).first()
            if existing_exam:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Student has already submitted a DOCX exam."
                )
        
        # Score
        score_data = compute_final_score(grading_result)

        # Upload submission file to Cloudinary
        with open(submission_path, "rb") as f:
            submission_url = upload_file_to_cloudinary(f, filename=submission_filename)

        # Xoá file sau khi upload
        if os.path.exists(sample_path): os.unlink(sample_path)
        if os.path.exists(submission_path): os.unlink(submission_path)


        # Save exam
        exam = Exam(
            name=submission_filename,
            file_format="docx",
            score=score_data["final_score"],
            report=grading_result,
            exam_code=exam_code,
            student_id=student.id,
            teacher_id=current_teacher.id,
            submission_url=submission_url,
        )
        
        db.add(exam)
        db.commit()
        db.refresh(exam)
        
        return exam
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Clean up files in case of error
        if os.path.exists(sample_path):
            os.unlink(sample_path)
        if os.path.exists(submission_path):
            os.unlink(submission_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}",
        )


@router.get("/{exam_id}", response_model=ExamWithRelations)
def read_exam(
    *,
    db: Session = Depends(get_db),
    exam_id: int,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Get exam by ID with related student and teacher information.
    """
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exam not found",
        )
    
    exam_dict = ExamResponse.model_validate(exam).model_dump()
    if exam.student:
        exam_dict["student.name"] = exam.student.name
        exam_dict["student.student_code"] = exam.student.student_code
    if exam.teacher:
        exam_dict["teacher.name"] = exam.teacher.name
    
    return exam_dict


@router.put("/{exam_id}", response_model=ExamResponse)
def update_exam(
    *,
    db: Session = Depends(get_db),
    exam_id: int,
    exam_in: ExamUpdate,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> Any:
    """
    Update an exam.
    """
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exam not found",
        )
    
    # Update exam
    update_data = exam_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(exam, field, value)
    
    db.add(exam)
    db.commit()
    db.refresh(exam)
    return exam


@router.post("/grade-xlsx", response_model=ExamResponse)
async def grade_xlsx_exam(
    *,
    db: Session = Depends(get_db),
    sample_file: UploadFile = File(...),
    submission_file: UploadFile = File(...),
    current_teacher: Teacher = Depends(get_current_teacher),
) -> ExamResponse:
    # Validate file formats
    if not sample_file.filename.endswith(".xlsx") or not submission_file.filename.endswith(".xlsx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both files must be in .xlsx format",
        )

    sample_path = Path(settings.UPLOAD_DIR) / f"{uuid4()}-sample.xlsx"
    submission_path = Path(settings.UPLOAD_DIR) / f"{uuid4()}-submission.xlsx"
    exam_code = None

    try:
        # Save uploaded files
        with open(sample_path, "wb") as buffer:
            shutil.copyfileobj(sample_file.file, buffer)
        with open(submission_path, "wb") as buffer:
            shutil.copyfileobj(submission_file.file, buffer)


        # Grade the exam
        model_result = load_excel_result(str(sample_path))
        student_result = load_excel_result(str(submission_path))
        grading_result = evaluate_excel_result(model_result, student_result)

        submission_filename = Path(submission_file.filename).stem
        # Lấy thông tin sinh viên từ grading_result
        info = grading_result.get("info_student")
        if info:
            student_code = info.get("student_code")
            student_name = info.get("name")
            exam_code = info.get("exam_code")
        else:
            # Nếu không có info, dùng tên file làm mã và tên sinh viên
            student_code = submission_filename
            student_name = submission_filename
            exam_code = Path(sample_file.filename).stem

        # Kiểm tra student
        student = db.query(Student).filter(Student.student_code == student_code).first()
        if student:
            # Kiểm tra đã nộp bài chưa
            existing_exam = db.query(Exam).filter(
                Exam.student_id == student.id,
                Exam.file_format == "xlsx"
            ).first()
            if existing_exam:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Student has already submitted an xlsx exam."
                )
        else:
            student = Student(
                student_code=student_code,
                name=student_name,
                birth_date=datetime.now().date(),
            )
            db.add(student)
            db.commit()
            db.refresh(student)

        # Upload submission file to Cloudinary
        with open(submission_path, "rb") as f:
            submission_url = upload_file_to_cloudinary(f, filename=submission_filename)

        # Xoá file sau khi upload
        if os.path.exists(sample_path): os.unlink(sample_path)
        if os.path.exists(submission_path): os.unlink(submission_path)

        # Create exam record
        exam = Exam(
            name=submission_filename,
            file_format="xlsx",
            score=grading_result.get("score", 0.0),
            report=grading_result,
            student_id=student.id,
            teacher_id=current_teacher.id,
            exam_code=exam_code,
            submission_url=submission_url,
        )

        db.add(exam)
        db.commit()
        db.refresh(exam)

        return exam

    except Exception as e:
        if os.path.exists(sample_path):
            os.unlink(sample_path)
        if os.path.exists(submission_path):
            os.unlink(submission_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}",
        )



@router.delete("/{exam_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_exam(
    *,
    db: Session = Depends(get_db),
    exam_id: int,
    current_teacher: Teacher = Depends(get_current_teacher),
) -> None:
    """
    Delete an exam.
    """
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Exam not found",
        )

    # Xoá file khỏi Cloudinary nếu submission_url tồn tại
    if exam.submission_url:
        try:
            parsed = urlparse(exam.submission_url)
            path = parsed.path  # /<...>/upload/vXXX/folder/filename.docx
            parts = path.split("/")
            # Tìm vị trí phần sau "upload/"
            upload_idx = parts.index("upload")
            public_id_with_ext = "/".join(parts[upload_idx + 2:])
            public_id = os.path.splitext(public_id_with_ext)[0]  # Bỏ đuôi .docx

            delete_from_cloudinary(public_id)
        except Exception as e:
            print(f"Warning: Failed to delete from Cloudinary: {e}")
    
    db.delete(exam)
    db.commit()
    
    return None

@router.get("/{exam_id}/report")
def get_exam_report(
    *,
    db: Session = Depends(get_db),
    exam_id: int,
    current_teacher: Teacher = Depends(get_current_teacher),
):
    exam = db.query(Exam).filter(
        Exam.id == exam_id,
        Exam.teacher_id == current_teacher.id
    ).first()

    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    return {
        "report": exam.report,
        "score": exam.score,
    }
