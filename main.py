import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from jose import JWTError, jwt

import socketio

from database import db, create_document, get_documents
from schemas import Session as SessionModel, LocationPing, Attendance as AttendanceModel

# ----------------------
# Config & Globals
# ----------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
SELFIE_DISTANCE_THRESHOLD_METERS = float(os.getenv("SELFIE_DISTANCE_THRESHOLD_METERS", "5.0"))
LAST_LOCATION_MAX_AGE_SECONDS = int(os.getenv("LAST_LOCATION_MAX_AGE_SECONDS", "30"))
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "attendance-selfies/")
RETENTION_DAYS = int(os.getenv("PHOTO_RETENTION_DAYS", "30"))

# Socket.IO
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
sio_app = socketio.ASGIApp(sio, socketio_path="/socket.io")

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Socket.IO under /ws
app.mount("/ws", sio_app)

# ----------------------
# Utility functions
# ----------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, atan2, sqrt
    R = 6371000.0
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ----------------------
# Auth
# ----------------------
class AuthedUser(BaseModel):
    userId: str
    name: Optional[str] = None
    role: str  # 'teacher' | 'student'


def decode_jwt(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e


def get_current_user(request: Request) -> AuthedUser:
    auth = request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    data = decode_jwt(token)
    user_id = data.get("userId") or data.get("sub")
    role = data.get("role")
    name = data.get("name")
    if not user_id or not role:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return AuthedUser(userId=user_id, role=role, name=name)


# Mock login for demo/testing
class MockLoginBody(BaseModel):
    userId: str
    role: str
    name: Optional[str] = None
    expMinutes: int = 120


@app.post("/api/auth/mock-login")
def mock_login(body: MockLoginBody):
    exp = now_utc() + timedelta(minutes=body.expMinutes)
    payload = {"sub": body.userId, "userId": body.userId, "role": body.role, "name": body.name, "exp": int(exp.timestamp())}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token, "expiresAt": to_iso(exp)}


# ----------------------
# Session APIs
# ----------------------
class StartSessionBody(BaseModel):
    teacherId: str
    lat: float
    lon: float
    expiryMinutes: int = 15
    teacherName: Optional[str] = None


@app.post("/api/session/start")
def start_session(body: StartSessionBody, user: AuthedUser = Depends(get_current_user)):
    if user.role != "teacher" or user.userId != body.teacherId:
        raise HTTPException(status_code=403, detail="Only the teacher can start session")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    starts_at = now_utc()
    expires_at = starts_at + timedelta(minutes=body.expiryMinutes)

    session_doc = SessionModel(
        sessionId=session_id,
        teacherId=body.teacherId,
        teacherName=body.teacherName or user.name,
        teacherLat=body.lat,
        teacherLon=body.lon,
        startsAt=starts_at,
        expiresAt=expires_at,
        status="active",
    )
    create_document("session", session_doc)

    return {
        "sessionId": session_id,
        "startsAt": to_iso(starts_at),
        "expiresAt": to_iso(expires_at),
        "teacherLocation": {"lat": body.lat, "lon": body.lon},
    }


class LocationBody(BaseModel):
    userId: str
    lat: float
    lon: float
    clientTimestamp: str


@app.post("/api/session/{sessionId}/location")
def post_location(sessionId: str, body: LocationBody, request: Request, user: AuthedUser = Depends(get_current_user)):
    if user.userId != body.userId:
        raise HTTPException(status_code=403, detail="userId mismatch")

    # Find session
    session_docs = get_documents("session", {"sessionId": sessionId})
    if not session_docs:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = session_docs[0]

    if sess.get("status") != "active" or now_utc() > sess.get("expiresAt"):
        raise HTTPException(status_code=400, detail="Session inactive or expired")

    teacher_lat = float(sess["teacherLat"])
    teacher_lon = float(sess["teacherLon"])
    distance = haversine(teacher_lat, teacher_lon, body.lat, body.lon)

    server_ts = now_utc()
    ip_addr = request.client.host if request.client else None

    ping_doc = LocationPing(
        sessionId=sessionId,
        userId=body.userId,
        lat=body.lat,
        lon=body.lon,
        clientTimestamp=datetime.fromisoformat(body.clientTimestamp.replace("Z", "+00:00")),
        serverTimestamp=server_ts,
        distanceMeters=distance,
        ipAddress=ip_addr,
    )
    create_document("locationping", ping_doc)

    allowed = distance <= SELFIE_DISTANCE_THRESHOLD_METERS

    return {
        "allowed": allowed,
        "distanceMeters": round(distance, 2),
        "maxAgeSeconds": LAST_LOCATION_MAX_AGE_SECONDS,
        "serverTimestamp": to_iso(server_ts),
    }


# ----------------------
# Storage (S3 or local)
# ----------------------

def store_image(file: UploadFile, sessionId: str, userId: str) -> str:
    filename = f"{sessionId}/{userId}/{uuid.uuid4().hex}.jpg"
    if S3_BUCKET:
        import boto3
        s3 = boto3.client("s3")
        body = file.file.read()
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=S3_PREFIX + filename,
            Body=body,
            ContentType=file.content_type or "image/jpeg",
            ServerSideEncryption="AES256",
        )
        base_url = os.getenv("S3_BASE_URL")
        if base_url:
            return f"{base_url}/{S3_PREFIX}{filename}"
        else:
            return f"s3://{S3_BUCKET}/{S3_PREFIX}{filename}"
    else:
        # Local fallback
        local_dir = os.path.join("uploads", sessionId, userId)
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, f"{uuid.uuid4().hex}.jpg")
        with open(path, "wb") as f:
            f.write(file.file.read())
        return f"/uploads/{sessionId}/{userId}/" + os.path.basename(path)


# ----------------------
# Selfie upload
# ----------------------

@app.post("/api/session/{sessionId}/selfie")
def upload_selfie(sessionId: str, request: Request, file: UploadFile = File(...), user: AuthedUser = Depends(get_current_user)):
    # Validate session
    session_docs = get_documents("session", {"sessionId": sessionId})
    if not session_docs:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = session_docs[0]

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Get last location for this user in this session
    pings = db["locationping"].find({"sessionId": sessionId, "userId": user.userId}).sort("serverTimestamp", -1).limit(1)
    last_ping = next(iter(pings), None)
    if not last_ping:
        raise HTTPException(status_code=400, detail="No recent location")

    # Check age and distance
    last_ts: datetime = last_ping.get("serverTimestamp")
    if (now_utc() - last_ts).total_seconds() > LAST_LOCATION_MAX_AGE_SECONDS:
        raise HTTPException(status_code=400, detail="Location too old")

    distance = float(last_ping.get("distanceMeters", 1e9))
    if distance > SELFIE_DISTANCE_THRESHOLD_METERS:
        raise HTTPException(status_code=400, detail="Too far from teacher")

    # Store image
    photo_url = store_image(file, sessionId, user.userId)

    attendance_id = f"att_{uuid.uuid4().hex[:10]}"
    att_doc = AttendanceModel(
        attendanceId=attendance_id,
        sessionId=sessionId,
        userId=user.userId,
        status="uploaded",
        photoUrl=photo_url,
        studentLat=float(last_ping.get("lat")),
        studentLon=float(last_ping.get("lon")),
        teacherLat=float(sess.get("teacherLat")),
        teacherLon=float(sess.get("teacherLon")),
        distanceMeters=distance,
        clientTimestamp=last_ping.get("clientTimestamp"),
        serverTimestamp=now_utc(),
        ipAddress=(request.client.host if request.client else None),
    )
    create_document("attendance", att_doc)

    # Emit realtime event to teacher room
    try:
        import anyio
        anyio.from_thread.run(
            sio.emit,
            "attendance:uploaded",
            {"sessionId": sessionId, "userId": user.userId, "attendanceId": attendance_id, "photoUrl": photo_url, "distance": round(distance, 2)},
            to=f"session:{sessionId}:teacher",
        )
    except Exception:
        pass

    return {"status": "uploaded", "attendanceId": attendance_id, "photoUrl": photo_url}


# ----------------------
# Teacher view
# ----------------------

@app.get("/api/session/{sessionId}/teacher-view")
def teacher_view(sessionId: str, user: AuthedUser = Depends(get_current_user)):
    if user.role != "teacher":
        raise HTTPException(status_code=403, detail="Forbidden")

    session_docs = get_documents("session", {"sessionId": sessionId})
    if not session_docs:
        raise HTTPException(status_code=404, detail="Session not found")
    sess = session_docs[0]

    # For each distinct user in pings, get last ping and attendance if any
    pipeline = [
        {"$match": {"sessionId": sessionId}},
        {"$sort": {"serverTimestamp": -1}},
        {"$group": {"_id": "$userId", "last": {"$first": "$$ROOT"}}},
    ]
    last_pings = list(db["locationping"].aggregate(pipeline))
    attendance_docs = list(db["attendance"].find({"sessionId": sessionId}))
    att_map = {d["userId"]: d for d in attendance_docs}

    students = []
    for p in last_pings:
        u = p["_id"]
        last = p["last"]
        att = att_map.get(u)
        students.append({
            "userId": u,
            "distance": round(float(last.get("distanceMeters", 0.0)), 2),
            "lastSeen": to_iso(last.get("serverTimestamp")),
            "status": (att.get("status") if att else "pending"),
            "photoUrl": (att.get("photoUrl") if att else None),
        })

    return {
        "session": {
            "sessionId": sess.get("sessionId"),
            "teacherId": sess.get("teacherId"),
            "teacherName": sess.get("teacherName"),
            "startsAt": to_iso(sess.get("startsAt")),
            "expiresAt": to_iso(sess.get("expiresAt")),
        },
        "students": students,
    }


# ----------------------
# Manual override
# ----------------------
class OverrideBody(BaseModel):
    userId: str
    status: str  # overridden_present | overridden_absent


@app.post("/api/session/{sessionId}/override")
def override_attendance(sessionId: str, body: OverrideBody, user: AuthedUser = Depends(get_current_user)):
    if user.role != "teacher":
        raise HTTPException(status_code=403, detail="Forbidden")

    existing = db["attendance"].find_one({"sessionId": sessionId, "userId": body.userId})
    if existing:
        db["attendance"].update_one({"_id": existing["_id"]}, {"$set": {"status": body.status, "updated_at": now_utc()}})
        attendance_id = existing.get("attendanceId")
    else:
        attendance_id = f"att_{uuid.uuid4().hex[:10]}"
        create_document("attendance", AttendanceModel(
            attendanceId=attendance_id,
            sessionId=sessionId,
            userId=body.userId,
            status=body.status,
            serverTimestamp=now_utc(),
        ))

    # Realtime update
    try:
        import anyio
        anyio.from_thread.run(
            sio.emit,
            "attendance:overridden",
            {"sessionId": sessionId, "userId": body.userId, "status": body.status, "attendanceId": attendance_id},
            to=f"session:{sessionId}:teacher",
        )
    except Exception:
        pass

    return {"ok": True}


# ----------------------
# Socket.IO events
# ----------------------

@sio.event
async def connect(sid, environ, auth):
    pass


@sio.event
async def join_teacher(sid, data):
    session_id = data.get("sessionId")
    await sio.enter_room(sid, f"session:{session_id}:teacher")


@sio.event
async def join_student(sid, data):
    session_id = data.get("sessionId")
    await sio.enter_room(sid, f"session:{session_id}:students")


# ----------------------
# Health and database test
# ----------------------
@app.get("/")
def read_root():
    return {"message": "Attendance backend running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ----------------------
# Uvicorn
# ----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
