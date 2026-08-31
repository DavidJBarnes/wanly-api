import asyncio
import os
from uuid import UUID

from fastapi import UploadFile

from app.config import settings
from app.s3 import upload_bytes


