# app/utils/cloudinary_utils.py
import cloudinary
import cloudinary.uploader
from app.core.config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)

def upload_file_to_cloudinary(file_data, filename: str, folder: str = "exam-submissions"):
    result = cloudinary.uploader.upload(
        file_data,
        resource_type="auto",
        folder=folder,
        public_id=filename,
        use_filename=True,
        unique_filename=False,
        overwrite=True,
    )
    return result.get("secure_url")

def delete_from_cloudinary(public_id: str) -> None:
    cloudinary.uploader.destroy(public_id, resource_type="auto")