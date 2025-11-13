from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class User(BaseModel):
    userId: str = Field(..., description="Unique user id")
    name: str
    email: Optional[str] = None
    role: str = Field(..., description="teacher|student")

class Session(BaseModel):
    sessionId: str
    teacherId: str
    teacherName: Optional[str] = None
    teacherLat: float
    teacherLon: float
    startsAt: datetime
    expiresAt: datetime
    status: str = Field("active", description="active|stopped|expired")

class LocationPing(BaseModel):
    sessionId: str
    userId: str
    lat: float
    lon: float
    clientTimestamp: datetime
    serverTimestamp: datetime
    distanceMeters: Optional[float] = None
    ipAddress: Optional[str] = None

class Attendance(BaseModel):
    attendanceId: str
    sessionId: str
    userId: str
    status: str = Field("uploaded", description="uploaded|pending|verified|overridden_present|overridden_absent")
    photoUrl: Optional[str] = None
    studentLat: Optional[float] = None
    studentLon: Optional[float] = None
    teacherLat: Optional[float] = None
    teacherLon: Optional[float] = None
    distanceMeters: Optional[float] = None
    clientTimestamp: Optional[datetime] = None
    serverTimestamp: datetime
    ipAddress: Optional[str] = None
    verificationConfidence: Optional[float] = None
