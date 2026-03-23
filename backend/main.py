# main.py - Updated Version with All Fixes
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Header, BackgroundTasks, Request, Response
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import jwt
import bcrypt
import sqlite3
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor
import secrets
import httpx
import requests
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import sys
from contextlib import contextmanager, asynccontextmanager
import re
import json
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from groq import Groq
# Add parent directory to sys.path to allow importing from scr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scr.process_pdf import process_new_pdf

# Graceful semantic cache import — app works even without Qdrant
try:
    from semantic_cache import semantic_cache
    if semantic_cache:
        print("✅ Semantic cache initialized")
    else:
        print("⚠️ Semantic cache returned None — cache disabled")
except Exception as e:
    print(f"⚠️ Semantic cache disabled (import error): {e}")
    semantic_cache = None

# Thread pool for Qdrant operations
executor = ThreadPoolExecutor(max_workers=3)
sys.stdout.reconfigure(encoding='utf-8')
# =======================
# Configuration
# =======================
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

if len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY is too short. Minimum 32 characters required."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
UPLOAD_DIR = "uploads"
LECTURES_DIR = "lectures"
DB_PATH = "university_chatbot.db"

# Email Configuration
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "noreply@university.edu")

# Groq Client for STT
try:
    groq_stt_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    logger.error(f"❌ Groq STT client failed: {e}")
    groq_stt_client = None

# Admin Credentials
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

if not ADMIN_EMAIL or not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_EMAIL and ADMIN_PASSWORD must be set in environment.")

# =======================
# Lifespan Events (Redis)
# =======================
redis_client = None
redis_client_instance = None
redis_available = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_available, redis_client_instance
    
    # 1. Initialize database FIRST (create tables, seed data, run migrations)
    init_db()
    
    # 2. Initialize Redis rate limiter
    try:
        redis_client_instance = redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
        await redis_client_instance.ping()
        await FastAPILimiter.init(redis_client_instance)
        redis_available = True
        print("✅ Redis Rate Limiter Initialized")
    except Exception as e:
        print(f"⚠️ Redis not available (Rate limiting disabled). Continuing without it.")
        redis_client_instance = None
    yield
    # Cleanup on shutdown
    if redis_client_instance:
        try:
            await redis_client_instance.close()
        except:
            pass

app = FastAPI(title="University AI Chatbot API with Courses", lifespan=lifespan)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    # Only add to HTML responses
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type or request.url.path.endswith(".html"):
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://unpkg.com; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; object-src 'none'; base-uri 'self'; frame-ancestors 'none';"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(self)"
    return response

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

ALLOWED_ORIGINS = [
    origin.strip() 
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:7860").split(",")
    if origin.strip()
]
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["X-Conversation-Id", "X-Message-Id"],  # ← CRITICAL: frontend reads these headers
)

# Serve only the chat image securely instead of the whole backend folder
@app.get("/static/ch.png")
async def get_chat_image():
    return FileResponse("ch.png")

# Serve Service Worker
@app.get("/sw.js")
async def get_sw():
    return FileResponse("sw.js")

STATIC_JS_DIR = Path("static/js").resolve()
SAFE_FILENAME = re.compile(r'^[a-zA-Z0-9_\-]+\.(js|css|map)$')

# Serve static JS files (e.g., for caching)
@app.get("/static/js/{filename}")
async def get_js_file(filename: str):
    # Step 1: Validate filename format
    if not SAFE_FILENAME.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Step 2: Resolve full path and verify it stays within allowed directory
    for base_dir in [Path(".").resolve(), STATIC_JS_DIR]:
        candidate = (base_dir / filename).resolve()
        
        # ✅ Ensure resolved path is inside allowed directory
        try:
            candidate.relative_to(base_dir)
        except ValueError:
            continue  # Path escaped the directory — skip
        
        if candidate.exists():
            return FileResponse(str(candidate))
            
    raise HTTPException(status_code=404, detail="File not found")

async def safe_rate_limit(request: Request, response: Response):
    if redis_available:
        await RateLimiter(times=10, seconds=60)(request, response)

# =======================
# Database Setup with Auto-Migration
# =======================
@contextmanager
def get_db_context():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    with get_db_context() as conn:
        c = conn.cursor()
        
        print("\n" + "="*60)
        print("🔄 Initializing Database...")
        print("="*60 + "\n")

        # Enable WAL mode for concurrency
        c.execute("PRAGMA journal_mode=WAL;")

        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Courses table
        c.execute('''CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            admin_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users(id)
        )''')
        
        # Conversations table
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Messages table - Standardized to 'role'
        c.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )''')
        
        # Feedbacks table
        c.execute('''CREATE TABLE IF NOT EXISTS feedbacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            feedback_type TEXT NOT NULL,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            UNIQUE(message_id, user_id)
        )''')
        
        # Files table
        c.execute('''CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            file_type TEXT NOT NULL,
            subject TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # Lectures table
        c.execute('''CREATE TABLE IF NOT EXISTS lectures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER NOT NULL,
            course_id INTEGER,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            subject TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT 'pending',
            total_chunks INTEGER DEFAULT 0,
            total_characters INTEGER DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY (admin_id) REFERENCES users(id),
            FOREIGN KEY (course_id) REFERENCES courses(id)
        )''')
        
        # Password reset tokens table
        c.execute('''CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')
        
        # OTP tokens table (for Redis fallback)
        c.execute('''CREATE TABLE IF NOT EXISTS otp_tokens (
            email TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL
        )''')
        
        conn.commit()
        
        # 🔧 AUTO-MIGRATION - Add missing columns
        print("🔍 Checking for missing columns...")
        
        # Migration for 'messages' table to remove 'sender' and standardize on 'role'
        c.execute("PRAGMA table_info(messages)")
        message_columns = {col[1] for col in c.fetchall()}
        
        if 'sender' in message_columns:
            print("   🔧 Migrating 'messages' table to remove 'sender' column...")
            c.executescript("""
                PRAGMA foreign_keys=off;
                BEGIN TRANSACTION;
                CREATE TABLE messages_new (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                INSERT INTO messages_new (id, conversation_id, role, content, created_at)
                SELECT 
                    id,
                    conversation_id,
                    CASE WHEN sender = 'ai' THEN 'assistant' ELSE 'user' END,
                    content,
                    created_at
                FROM messages;
                DROP TABLE messages;
                ALTER TABLE messages_new RENAME TO messages;
                COMMIT;
                PRAGMA foreign_keys=on;
            """)
            print("   ✅ 'messages' table migrated successfully.")
        elif 'role' not in message_columns:
            print("   🔧 Adding 'role' column to messages table...")
            c.execute("ALTER TABLE messages ADD COLUMN role TEXT DEFAULT 'user'")
            conn.commit()

        # Migration for 'course_id' in lectures table
        c.execute("PRAGMA table_info(lectures)")
        lecture_columns = {col[1] for col in c.fetchall()}
        if 'course_id' not in lecture_columns:
            print("   🔧 Adding 'course_id' to lectures...")
            c.execute("ALTER TABLE lectures ADD COLUMN course_id INTEGER REFERENCES courses(id)")
            # Attempt to populate from old string-based subject
            c.execute("UPDATE lectures SET course_id = (SELECT id FROM courses WHERE courses.name = lectures.subject)")
            conn.commit()
        
        # Check lectures table columns
        c.execute("PRAGMA table_info(lectures)")
        existing_columns = {col[1] for col in c.fetchall()}
        
        required_columns = {
            'processing_status': "TEXT DEFAULT 'pending'",
            'total_chunks': "INTEGER DEFAULT 0",
            'total_characters': "INTEGER DEFAULT 0",
            'error_message': "TEXT"
        }
        
        for col_name, col_type in required_columns.items():
            if col_name not in existing_columns:
                try:
                    c.execute(f"ALTER TABLE lectures ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                    print(f"   ✅ Added column: {col_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        print(f"   ⚠️ Error adding {col_name}: {e}")
        
        # Check for suggestions column in lectures
        if 'suggestions' not in existing_columns:
            try:
                c.execute("ALTER TABLE lectures ADD COLUMN suggestions TEXT")
                conn.commit()
                print("   ✅ Added column: suggestions to lectures")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    print(f"   ⚠️ Error adding suggestions: {e}")

        # Check conversations table for is_deleted
        c.execute("PRAGMA table_info(conversations)")
        conv_columns = {col[1] for col in c.fetchall()}
        if 'is_deleted' not in conv_columns:
            try:
                c.execute("ALTER TABLE conversations ADD COLUMN is_deleted INTEGER DEFAULT 0")
                conn.commit()
                print("   ✅ Added column: is_deleted to conversations")
            except Exception as e:
                print(f"   ⚠️ Error adding is_deleted: {e}")

        # Migration for 'is_verified' in users table
        c.execute("PRAGMA table_info(users)")
        user_columns = {col[1] for col in c.fetchall()}
        if 'is_verified' not in user_columns:
            c.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER NOT NULL DEFAULT 0")
            conn.commit()
            # Existing admin and users should be pre-verified
            c.execute("UPDATE users SET is_verified = 1 WHERE role = 'admin'")
            conn.commit()
            print("   ✅ Added column: is_verified to users")
        # Seed admin user
        hashed = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        try:
            c.execute("SELECT id FROM users WHERE email = ?", (ADMIN_EMAIL,))
            existing_admin = c.fetchone()
            
            if not existing_admin:
                c.execute("INSERT INTO users (email, password_hash, role, is_verified) VALUES (?, ?, ?, 1)",
                          (ADMIN_EMAIL, hashed, 'admin'))
                conn.commit()
                admin_id = c.lastrowid
                print(f"\n✅ Admin user created: {ADMIN_EMAIL}")
                print(f"   Password: [HIDDEN] (Check .env or use default if dev)")
            else:
                admin_id = existing_admin[0]
                print(f"\nℹ️ Admin user already exists: {ADMIN_EMAIL}")

            # Seed 12 Fixed Courses
            fixed_courses = [
                ("Android Development",              "Android SDK, activities, fragments, Kotlin basics, and mobile UI design"),
                ("Computer Networks",                "OSI model, TCP/IP stack, routing protocols, subnetting, and network security basics"),
                ("Information Security",             "Cryptography, authentication, ethical hacking, firewalls, and penetration testing"),
                ("Operating Systems",                "Process management, scheduling, memory management, deadlocks, and file systems"),
                ("Theory of Computation",            "Finite automata, regular expressions, context-free grammars, Turing machines, and decidability"),
                ("Algorithms Design and Analysis",   "Sorting, searching, dynamic programming, greedy algorithms, and complexity analysis (Big-O)"),
                ("Computer Architecture",            "CPU design, instruction sets, pipelining, memory hierarchy, and cache systems"),
                ("Machine Learning",                 "Supervised and unsupervised learning, regression, classification, neural networks, and model evaluation"),
                ("Compiler Design",                  "Lexical analysis, parsing, semantic analysis, intermediate code generation, and optimization"),
                ("Computer Graphics",                "2D/3D transformations, rendering pipelines, OpenGL, shading models, and rasterization"),
                ("Human Computer Interaction",       "UX principles, user research, prototyping, usability testing, and accessibility standards"),
            ]

            print("\n🌱 Seeding fixed courses...")
            for name, desc in fixed_courses:
                c.execute("SELECT id FROM courses WHERE name = ?", (name,))
                if not c.fetchone():
                    c.execute("INSERT INTO courses (name, description, admin_id) VALUES (?, ?, ?)",
                              (name, desc, admin_id))
                    print(f"   ✅ Added course: {name}")
            
            conn.commit()

        except Exception as e:
            print(f"❌ Error seeding data: {e}")

        # Verify bcrypt works correctly at startup
        try:
            _test_hash = bcrypt.hashpw(b"test_password_123", bcrypt.gensalt())
            assert bcrypt.checkpw(b"test_password_123", _test_hash), "bcrypt verification failed"
            print("✅ bcrypt cryptography verified")
        except Exception as e:
            raise RuntimeError(f"❌ bcrypt is not working correctly: {e}. Check your bcrypt installation.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(LECTURES_DIR, exist_ok=True)
    os.makedirs(os.path.join("static", "js"), exist_ok=True)
    
    print("\n" + "="*60)
    print("✅ Database initialization completed!")
    print("="*60 + "\n")

# =======================
# Helper Functions
# =======================
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

REFRESH_TOKEN_EXPIRE_DAYS = 30

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Depends(oauth2_scheme)
):
    raw_token = None

    if authorization and authorization.startswith("Bearer "):
        raw_token = authorization.split(" ")[1]
    elif token:
        raw_token = token

    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = verify_token(raw_token)
    user_id = payload.get("user_id")
    role = payload.get("role")

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"user_id": user_id, "role": role}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Safely verify password with timing-safe comparison."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        print(f"⚠️ Password verification error: {e}")
        return False

def send_email(to_email: str, subject: str, html_content: str):
    """Sends an email using Brevo API (HTTP POST)."""
    if not BREVO_API_KEY:
        print("⚠️ Email configuration missing (BREVO_API_KEY). Skipping email.")
        return

    if not BREVO_API_KEY.startswith("xkeysib-"):
        print("⚠️ Error: The provided BREVO_API_KEY looks like an SMTP password (starts with 'xsmtpsib').")
        print("👉 Please generate a new API Key from Brevo Dashboard -> SMTP & API -> API Keys (it should start with 'xkeysib').")
        return

    endpoint = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "sender": {"name": "University AI", "email": SENDER_EMAIL},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_content
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 201 or response.status_code == 200:
            print(f"✅ Transaction Successful: Email queued for {to_email}")
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"⚠️ Infrastructure Failure: {str(e)}")

def store_otp_db(email: str, otp: str):
    """SQLite fallback for storing OTPs."""
    with get_db_context() as conn:
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        conn.execute(
            """INSERT INTO otp_tokens (email, code, expires_at)
               VALUES (?, ?, ?)
               ON CONFLICT(email) DO UPDATE SET 
               code=excluded.code, 
               expires_at=excluded.expires_at,
               created_at=CURRENT_TIMESTAMP""",
            (email, otp, expires.isoformat())
        )
        conn.commit()

def verify_otp_db(email: str, code: str) -> bool:
    """SQLite fallback for verifying OTPs."""
    with get_db_context() as conn:
        row = conn.execute(
            "SELECT code, expires_at FROM otp_tokens WHERE email = ?",
            (email,)
        ).fetchone()
        
        if not row:
            return False
        
        if row["code"] != code:
            return False
        
        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            conn.execute("DELETE FROM otp_tokens WHERE email = ?", (email,))
            conn.commit()
            return False
        
        # Consume the token
        conn.execute("DELETE FROM otp_tokens WHERE email = ?", (email,))
        conn.commit()
        return True

# =======================
# OTP Functions (Redis)
# =======================
async def store_otp(email: str, otp: str):
    """Store OTP in Redis with 5-minute expiry, or fallback to memory."""
    if redis_available and redis_client_instance:
        try:
            await redis_client_instance.setex(f"otp:{email}", 300, otp)
            return
        except Exception as e:
            print(f"⚠️ Redis error in store_otp: {e}")
    store_otp_db(email, otp)

async def verify_and_consume_otp(email: str, code: str) -> bool:
    """Verify OTP from Redis or memory and delete if valid."""
    if redis_available and redis_client_instance:
        try:
            stored = await redis_client_instance.get(f"otp:{email}")
            if not stored or stored != code:
                return False
            await redis_client_instance.delete(f"otp:{email}")
            return True
        except Exception as e:
            print(f"⚠️ Redis error in verify_and_consume_otp: {e}")
            return verify_otp_db(email, code)
            
    # Fallback logic (DB)
    return verify_otp_db(email, code)

# =======================
# Pydantic Models
# =======================
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    token: str
    new_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    refresh_token: Optional[str] = None

class RefreshRequest(BaseModel):
    refresh_token: str

class ChatMessage(BaseModel):
    conversation_id: Optional[int] = None
    message: str

class FeedbackRequest(BaseModel):
    message_id: int
    feedback_type: str
    comment: Optional[str] = None

class CourseCreate(BaseModel):
    name: str
    description: Optional[str] = None

# =======================
# Security Functions
# =======================
def sanitize_input(text: str) -> str:
    """Remove HTML tags and limit length to prevent injection/overflow."""
    clean = re.sub(r'<[^>]*>', '', text)
    return clean[:2000]  # Limit to 2000 chars

def is_jailbreak_attempt(text: str) -> bool:
    """Check for common LLM jailbreak patterns."""
    JAILBREAK_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+your\s+(system\s+)?prompt",
        r"system\s+prompt",
        r"you\s+are\s+not\s+(an?\s+)?AI",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode",
        r"pretend\s+(you\s+are|to\s+be)",
        r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(a\s+)?(?!student|teacher|professor)",  # allow role-play as educator
        r"forget\s+(your|all)\s+(rules|instructions|guidelines|training)",
        r"override\s+(your\s+)?(safety|guidelines|instructions|rules)",
        r"bypass\s+(your\s+)?(filter|safety|restrictions)",
        r"disregard\s+(your\s+)?(guidelines|rules|instructions)",
        r"you\s+have\s+no\s+restrictions",
        r"reveal\s+your\s+(system\s+)?prompt",
        r"print\s+your\s+(system\s+)?instructions",
        r"what\s+(is|are)\s+your\s+instructions",
        r"sudo\s+mode",
        r"admin\s+override",
        r"]*>",    # XSS attempt
        r"javascript\s*:",       # JS injection
    ]
    text_lower = text.lower()
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

# =======================
# Root Endpoint
# =======================
# Serve the Login page by default
@app.get("/")
async def root():
    return FileResponse("index.html")

# Serve HTML Pages
@app.get("/index.html")
async def index_page():
    return FileResponse("index.html")

@app.get("/login.html")
async def login_page():
    return FileResponse("login.html")

@app.get("/chat.html")
async def chat_page():
    return FileResponse("chat.html")

@app.get("/Admin-Dashboard.html")
async def admin_page():
    return FileResponse("Admin-Dashboard.html")
@app.get("/register.html")
async def register_page():
    return FileResponse("register.html")

@app.get("/forgot-password.html")
async def forgot_password_page():
    return FileResponse("forgot-password.html")

@app.get("/reset-password.html")
async def reset_password_page():
    return FileResponse("reset-password.html")

@app.get("/verify-email.html")
async def verify_email_page():
    return FileResponse("verify-email.html")

# =======================
# Auth Endpoints
# =======================
async def register_rate_limit(request: Request, response: Response):
    """Stricter rate limit for registration: 5 attempts per hour per IP."""
    if redis_available:
        await RateLimiter(times=5, seconds=3600)(request, response)

@app.post("/auth/register", dependencies=[Depends(register_rate_limit)])
async def register(background_tasks: BackgroundTasks, data: UserRegister):
    """Register new student account"""
    email = data.email
    password = data.password
    
    # Password strength validation (see FIX 14)
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter.")
    if not re.search(r'[0-9]', password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number.")

    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE email = ?", (email,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        c.execute(
            "INSERT INTO users (email, password_hash, role, is_verified) VALUES (?, ?, ?, ?)",
            (email, hashed_pw, "student", 0)
        )
        conn.commit()

    # Send OTP for email verification
    otp = f"{secrets.randbelow(1000000):06d}"
    await store_otp(email, otp)
    
    subject = "Verify Your Email - University AI 🎓"
    html_content = f"""
    
        
            Verify Your Email 🎓
            Hi there,
            Thank you for joining University AI. Use this code to verify your email:
            
                {otp}
                Expires in 5 minutes
            
            University AI Team
        
    """
    
    background_tasks.add_task(send_email, email, subject, html_content)

    return {
        "message": "Account created. Please verify your email with the OTP sent.",
        "email": email,
        "requires_verification": True
    }

@app.post("/auth/login", response_model=Token)
async def login(data: UserRegister):
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id, password_hash, role, is_verified FROM users WHERE email = ?", (data.email,))
        user = c.fetchone()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user["is_verified"]:
        raise HTTPException(status_code=403, detail="Email not verified. Please check your inbox for the verification code.")

    access = create_access_token({
        "user_id": user["id"],
        "role": user["role"]
    })

    refresh_token = create_refresh_token({"user_id": user["id"], "role": user["role"]})

    print(f"✅ User logged in: {data.email} ({user['role']})")

    return {
        "access_token": access,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "role": user["role"]
    }

@app.post("/auth/verify-email")
async def verify_email(request: VerifyEmailRequest, background_tasks: BackgroundTasks):
    """Verify email using OTP code sent during registration."""
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id, is_verified FROM users WHERE email = ?", (request.email,))
        user = c.fetchone()
    
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")
    
    if user["is_verified"]:
        return {"message": "Email already verified. You can log in."}
    
    # Verify OTP
    is_valid = await verify_and_consume_otp(request.email, request.code)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code.")
    
    # Mark as verified
    with get_db_context() as conn:
        c.execute("UPDATE users SET is_verified = 1 WHERE id = ?", (user['id'],))
        conn.commit()
    
    # Send welcome email
    subject = "Welcome to University AI! 🎓"
    html_content = f"""
    
        
            Email Verified! ✅
            Your email has been verified. You can now log in to University AI.
            University AI Team
        
    """
    background_tasks.add_task(send_email, request.email, subject, html_content)
    
    print(f"✅ Email verified for: {request.email}")
    return {"message": "Email verified successfully. You can now log in."}

@app.post("/auth/resend-verification")
async def resend_verification(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    """Resend email verification OTP."""
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id, is_verified FROM users WHERE email = ?", (request.email,))
        user = c.fetchone()
    
    if user and not user["is_verified"]:
        otp = f"{secrets.randbelow(1000000):06d}"
        await store_otp(request.email, otp)
        
        subject = "Verify Your Email - University AI 🎓"
        html_content = f"""
        
            
                New Verification Code
                
                    {otp}
                    Expires in 5 minutes
                
            
        """
        background_tasks.add_task(send_email, request.email, subject, html_content)
    
    return {"message": "If the account exists and is unverified, a new code has been sent."}





async def forgot_password_rate_limit(request: Request, response: Response):
    if redis_available:
        await RateLimiter(times=3, seconds=300)(request, response)

@app.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks, 
                          dependencies=[Depends(forgot_password_rate_limit)]):
    """Handles forgot password request"""
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE email = ?", (request.email,))
        user = c.fetchone()

    if user:
        otp = f"{secrets.randbelow(1000000):06d}"
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        
        # Store in Redis (or memory fallback)
        await store_otp(request.email, otp)
        
        with get_db_context() as conn:
            c = conn.cursor()
            # Invalidate old tokens
            c.execute("UPDATE password_reset_tokens SET used = 1 WHERE user_id = ?", (user['id'],))
            c.execute("INSERT INTO password_reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
                      (user['id'], otp, expires_at))
            conn.commit()

        subject = "Reset Your Password - University AI 🔑"
        html_content = f"""
        <html><body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #3A662A; text-align: center;">Password Reset Request</h2>
                <p>Hi,</p>
                <p>We received a request to reset your password. Use this code:</p>
                <div style="background: #f5f5f5; padding: 20px; text-align: center; border-radius: 8px; margin: 20px 0;">
                    <h1 style="letter-spacing: 8px; font-size: 36px; margin: 10px 0; color: #3A662A;">{otp}</h1>
                    <p style="margin: 10px 0 0 0; font-size: 12px; color: #999;">Expires in 5 minutes</p>
                </div>
                <p style="font-size: 14px; color: #666;">If you didn't request this, ignore this email.</p>
                <p style="font-size: 12px; color: #999; text-align: center;">University AI Security Team</p>
            </div>
        </body></html>"""
        
        # Run email sending in the background
        background_tasks.add_task(send_email, request.email, subject, html_content)
        print(f"🔐 Password reset code sent to: {request.email}")

    return {"message": "If account exists, password reset code has been sent."}

@app.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Resets user password"""
    with get_db_context() as conn:
        c = conn.cursor()
        
        c.execute("SELECT id FROM users WHERE email = ?", (request.email,))
        user = c.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")
            
        # Verify OTP (Checks Redis/Memory)
        is_valid = await verify_and_consume_otp(request.email, request.token)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail="Invalid or used code.")

        new_hashed_pw = bcrypt.hashpw(request.new_password.encode(), bcrypt.gensalt()).decode()
        c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hashed_pw, user["id"]))
        # Mark as used in DB
        c.execute("UPDATE password_reset_tokens SET used = 1 WHERE user_id = ? AND token = ?", (user['id'], request.token))
        conn.commit()

    print(f"🔐 Password reset successful for: {request.email}")

    return {"message": "Password reset successful. You can now login with your new password."}

@app.post("/auth/refresh", response_model=Token)
async def refresh_token(request: RefreshRequest):
    """Exchange a refresh token for a new access token."""
    try:
        payload = jwt.decode(request.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type.")
        
        user_id = payload.get("user_id")
        role = payload.get("role")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
        
        # Verify user still exists
        with get_db_context() as conn:
            c = conn.cursor()
            c.execute("SELECT id, role, is_verified FROM users WHERE id = ?", (user_id,))
            user = c.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="User no longer exists.")
        
        new_access_token = create_access_token({"user_id": user_id, "role": role})
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "role": role
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

# =======================
# Student Endpoints - FIXED
# =======================
@app.get("/student/conversations")
async def get_conversations(
    current_user: dict = Depends(get_current_user),
    limit: int = 50, offset: int = 0
):
    """Get all conversations with first message for title generation"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT 
                c.id,
                c.title,
                c.created_at,
                (SELECT content 
                 FROM messages m 
                 WHERE m.conversation_id = c.id 
                 AND m.role = 'user'
                 ORDER BY m.created_at ASC 
                 LIMIT 1) as first_message
            FROM conversations c
            WHERE c.user_id = ? AND c.is_deleted = 0
            ORDER BY c.created_at DESC LIMIT ? OFFSET ?
        """, (current_user['user_id'], limit, offset))
        
        conversations = []
        for row in c.fetchall():
            conv_dict = dict(row)
            conversations.append(conv_dict)

        c.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ? AND is_deleted = 0", (current_user['user_id'],))
        total = c.fetchone()[0]

        return {
            "conversations": conversations,
            "total": total,
            "limit": limit,
            "offset": offset
        }



@app.get("/student/conversation/{conversation_id}")
async def get_conversation(conversation_id: int, current_user: dict = Depends(get_current_user)):
    """Get conversation with all messages - handles both old and new format"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    with get_db_context() as conn:
        c = conn.cursor()
        try:
            # Verify conversation belongs to user
            c.execute("SELECT * FROM conversations WHERE id = ? AND user_id = ? AND is_deleted = 0",
                      (conversation_id, current_user['user_id']))
            conversation = c.fetchone()
            
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Get all messages with both sender and role
            c.execute("""
                SELECT id, conversation_id, role, content, created_at
                FROM messages 
                WHERE conversation_id = ? 
                ORDER BY created_at ASC
            """, (conversation_id,))
            
            messages = []
            for row in c.fetchall():
                msg = dict(row)
                # Add 'sender' for frontend backward compatibility
                msg['sender'] = 'ai' if msg['role'] == 'assistant' else 'user'
                messages.append(msg)
            
            return {
                "conversation": dict(conversation),
                "messages": messages
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error loading conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading conversation: {str(e)}")

@app.delete("/student/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int, 
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation and its messages"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")

    with get_db_context() as conn:
        c = conn.cursor()
        try:
            # Verify ownership
            c.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                      (conversation_id, current_user['user_id']))
            conversation = c.fetchone()

            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Soft delete conversation
            c.execute("UPDATE conversations SET is_deleted = 1 WHERE id = ?", (conversation_id,))
            conn.commit()

            return {
                "success": True, 
                "message": "Conversation deleted successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error deleting conversation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/student/chat", dependencies=[Depends(safe_rate_limit)])
async def chat(
    data: ChatMessage, 
    current_user: dict = Depends(get_current_user)
):
    """Send message and get AI response - NOW SAVES BOTH sender AND role"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    # 1. Security Checks
    clean_message = sanitize_input(data.message)
    if is_jailbreak_attempt(clean_message):
        raise HTTPException(status_code=400, detail="Unsafe content detected.")

    ai_message_id = None
    try:
        with get_db_context() as conn:
            c = conn.cursor()
            # Create new conversation if needed
            if data.conversation_id is None:
                # Generate title from first 3 words
                title_words = clean_message.split()[:3]
                title = " ".join(title_words)
                if len(clean_message.split()) > 3:
                    title += "..."
                
                c.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)",
                          (current_user['user_id'], title))
                conversation_id = c.lastrowid
                print(f"✅ Created new conversation: {conversation_id} - '{title}'")
            else:
                conversation_id = data.conversation_id
            
            # 🔥 CRITICAL: Save user message with BOTH sender AND role
            c.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, 'user', clean_message)
            )
            user_message_id = c.lastrowid
            
            print(f"💬 User message saved (ID: {user_message_id})")

            # 1. Pre-create AI message with empty content to get an ID
            c.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, 'assistant', '')
            )
            ai_message_id = c.lastrowid
            conn.commit()

            # 2. Fetch Conversation History (Last 5 messages)
            # We exclude the current user message (sent as 'question') and the empty AI placeholder
            # This implements the "Sliding Window" buffer memory
            c.execute("""
                SELECT role, content 
                FROM messages 
                WHERE conversation_id = ? 
                AND id < ? 
                ORDER BY id DESC 
                LIMIT 6
            """, (conversation_id, user_message_id))
            
            history_rows = c.fetchall()
        
        # Convert to list of dicts and reverse to chronological order (Oldest -> Newest)
        history = [{"role": row["role"], "content": row["content"]} for row in history_rows][::-1]

        # 1. Check Semantic Cache
        if semantic_cache:
            cached_response = await asyncio.to_thread(semantic_cache.get_cached_response, clean_message)
            
            if cached_response:
                # Update DB immediately
                try:
                    with get_db_context() as update_conn:
                        update_c = update_conn.cursor()
                        update_c.execute("UPDATE messages SET content = ? WHERE id = ?", (cached_response, ai_message_id))
                        update_conn.commit()
                except Exception as e:
                    print(f"❌ Error updating DB from cache: {e}")

                # Simulate streaming for cached response for consistent UX
                async def cached_generator():
                    """Simulate streaming for cached responses — prevents jarring pop-in effect."""
                    words = cached_response.split(" ")
                    chunk_size = 4  # Send 4 words at a time for natural feel
                    
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        if i + chunk_size < len(words):
                            chunk += " "
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
                        await asyncio.sleep(0.025)  # ~40 chunks/second
                    yield f"data: {json.dumps({'status': 'done'})}\n\n"

                return StreamingResponse(
                    cached_generator(),
                    media_type="text/event-stream",
                    headers={
                        "X-Conversation-Id": str(conversation_id),
                        "X-Message-Id": str(ai_message_id)
                    }
                )

        # 2. Define Generator for Streaming
        async def response_generator():
            full_response = ""
            rag_url = "http://127.0.0.1:8001/ask_stream"
            error_occurred = False
            
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    payload = {
                        "question": clean_message,
                        "history": history,
                        "conversation_id": conversation_id
                    }
                    
                    async with client.stream("POST", rag_url, json=payload) as response:
                        # 1. Read line by line using aiter_lines()
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                                
                            # 2. Parse each line as JSON after stripping "data: " prefix
                            if line.startswith("data: "):
                                data_str = line.replace("data: ", "")
                                try:
                                    data_json = json.loads(data_str)
                                    
                                    # 3. Forward "token" events directly to browser
                                    if "token" in data_json:
                                        token = data_json["token"]
                                        full_response += token # 5. Accumulate full_response from token events only
                                        yield f"data: {json.dumps({'token': token})}\n\n"
                                        
                                    # 4. Forward "status" events directly to browser
                                    elif "status" in data_json:
                                        status_msg = data_json["status"]
                                        yield f"data: {json.dumps({'status': status_msg})}\n\n"
                                        
                                    elif "error" in data_json:
                                        raise Exception(data_json["error"])
                                        
                                except json.JSONDecodeError:
                                    continue

            except Exception as e:
                error_occurred = True
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                
            finally:
                # Save full_response to DB or clean up placeholder
                try:
                    with get_db_context() as final_conn:
                        final_c = final_conn.cursor()

                        # If an error occurred OR the response is empty, delete the placeholder.
                        if error_occurred or not full_response.strip():
                            final_c.execute("DELETE FROM messages WHERE id = ?", (ai_message_id,))
                            final_conn.commit()
                            print(f"🗑️ Cleaned up AI message placeholder (ID: {ai_message_id})")
                        else:
                            # Otherwise, update the placeholder with the full response.
                            final_c.execute("UPDATE messages SET content = ? WHERE id = ?", (full_response, ai_message_id))
                            final_conn.commit()
                            print(f"🤖 AI message updated (ID: {ai_message_id})")

                            # And save to Semantic Cache.
                            if semantic_cache:
                                await asyncio.to_thread(
                                    semantic_cache.set_cached_response, 
                                    clean_message, 
                                    full_response
                                )
                except Exception as db_err:
                    print(f"❌ Error in chat 'finally' block: {db_err}")

        # 4. Return Streaming Response with IDs in headers
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream",
            headers={
                "X-Conversation-Id": str(conversation_id),
                "X-Message-Id": str(ai_message_id)
            }
        )
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/student/transcribe")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Transcribe audio using Whisper Large V3 Turbo via Groq"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not groq_stt_client:
        raise HTTPException(
            status_code=503,
            detail="Transcription service temporarily unavailable"
        )
    # Check file size (approximate via header)
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > 25 * 1024 * 1024: # 25MB limit for audio
        raise HTTPException(status_code=413, detail="File too large (max 25MB)")

    try:
        content = await file.read()
        
        # Send to Groq Whisper
        transcription = groq_stt_client.audio.transcriptions.create(
            file=(file.filename, content),
            model="whisper-large-v3-turbo",
            prompt="Database, Algorithm, SQL, Python, API, Backend, Frontend, University, Course, Lecture, Computer Science",
            response_format="json",
            temperature=0.0
        )
        
        return {"text": transcription.text}
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/student/lectures")
async def get_student_lectures(
    current_user: dict = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0
):
    """Get available lectures for students"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT l.id, c.name as subject, l.filename, l.uploaded_at
            FROM lectures l
            JOIN courses c ON l.course_id = c.id
            WHERE l.processing_status = 'completed' 
            ORDER BY l.uploaded_at DESC LIMIT ? OFFSET ?
        """, (limit, offset))
        lectures = [dict(row) for row in c.fetchall()]

        c.execute("SELECT COUNT(*) FROM lectures WHERE processing_status = 'completed'")
        total = c.fetchone()[0]

        return {"lectures": lectures, "total": total}

@app.post("/student/upload-file")
async def student_upload_file(
    request: Request,
    file: UploadFile = File(...),
    subject: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Allow students to upload personal study files."""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB student limit
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = {'.pdf', '.txt', '.docx', '.png', '.jpg', '.jpeg'}
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {allowed_extensions}")
    
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO files (user_id, filename, filepath, file_type, subject) VALUES (?, ?, ?, ?, ?)",
            (current_user['user_id'], file.filename, filepath, file.content_type, subject)
        )
        file_id = c.lastrowid
        conn.commit()
    
    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "message": "File uploaded successfully"
    }

@app.get("/student/my-files")
async def get_student_files(current_user: dict = Depends(get_current_user)):
    """Get all files uploaded by the current student."""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, filename, file_type, subject, uploaded_at
            FROM files WHERE user_id = ?
            ORDER BY uploaded_at DESC
        """, (current_user['user_id'],))
        files = [dict(row) for row in c.fetchall()]
    
    return {"files": files}

@app.get("/student/lecture/{lecture_id}/suggestions")
async def get_lecture_suggestions(lecture_id: int, current_user: dict = Depends(get_current_user)):
    """Get AI-generated suggestions for a specific lecture"""
    if current_user['role'] != 'student':
        raise HTTPException(status_code=403, detail="Access denied")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT l.suggestions, c.name as subject 
            FROM lectures l
            JOIN courses c ON l.course_id = c.id
            WHERE l.id = ?
        """, (lecture_id,))
        row = c.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Lecture not found")
        
    suggestions = []
    if row['suggestions']:
        try:
            suggestions = json.loads(row['suggestions'])
        except:
            suggestions = []
            
    return {
        "lecture_id": lecture_id,
        "subject": row['subject'],
        "suggestions": suggestions
    }

# =======================
# Feedback Endpoints
# =======================
@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit or update feedback for a message"""
    with get_db_context() as conn:
        c = conn.cursor()
        # Check if feedback exists
        c.execute(
            "SELECT id FROM feedbacks WHERE message_id = ? AND user_id = ?",
            (feedback.message_id, current_user['user_id'])
        )
        existing = c.fetchone()
        
        if existing:
            # Update existing
            c.execute(
                """UPDATE feedbacks 
                SET feedback_type = ?, comment = ?, created_at = CURRENT_TIMESTAMP
                WHERE id = ?""",
                (feedback.feedback_type, feedback.comment, existing['id'])
            )
            print(f"✅ Updated feedback for message {feedback.message_id}")
        else:
            # Create new
            c.execute(
                """INSERT INTO feedbacks (message_id, user_id, feedback_type, comment)
                VALUES (?, ?, ?, ?)""",
                (feedback.message_id, current_user['user_id'], 
                 feedback.feedback_type, feedback.comment)
            )
            print(f"✅ Created feedback for message {feedback.message_id}")
        
        conn.commit()
        
        return {
            "success": True,
            "message": "Feedback submitted successfully"
        }

@app.get("/admin/get-feedbacks")
async def get_all_feedbacks(
    current_user: dict = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """Get all feedback with details"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT 
                f.id,
                f.message_id,
                f.feedback_type,
                f.comment,
                f.created_at,
                m.content as message_content,
                m.role as message_role,
                u.email as user_email,
                c.title as conversation_title,
                c.id as conversation_id
            FROM feedbacks f
            JOIN messages m ON f.message_id = m.id
            JOIN users u ON f.user_id = u.id
            JOIN conversations c ON m.conversation_id = c.id
            ORDER BY f.created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        feedbacks = [dict(row) for row in c.fetchall()]
        c.execute("SELECT COUNT(*) FROM feedbacks")
        total = c.fetchone()[0]

        return {"feedbacks": feedbacks, "total": total}

# =======================
# Admin Endpoints - Courses
# =======================
@app.post("/admin/create-course")
async def create_course(
    course: CourseCreate,
    current_user: dict = Depends(get_current_user)
):
    """Create a new course"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        # Check if exists
        c.execute("SELECT id FROM courses WHERE name = ?", (course.name,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Course already exists")
        
        # Insert
        c.execute(
            "INSERT INTO courses (name, description, admin_id) VALUES (?, ?, ?)",
            (course.name, course.description, current_user['user_id'])
        )
        course_id = c.lastrowid
        conn.commit()
        
        print(f"✅ Course created: {course.name} (ID: {course_id})")
        
        return {
            "success": True,
            "message": "Course created successfully",
            "course_id": course_id,
            "name": course.name
        }

@app.get("/admin/get-courses")
async def get_courses(
    current_user: dict = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """Get all courses with lecture counts"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT 
                c.id,
                c.name,
                c.description,
                c.created_at,
                COUNT(l.id) as lecture_count
            FROM courses c
            LEFT JOIN lectures l ON c.id = l.course_id
            GROUP BY c.id
            ORDER BY c.created_at DESC LIMIT ? OFFSET ?
        """)
        
        courses = []
        for row in c.fetchall():
            courses.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "lecture_count": row[4]
            })
        
        c.execute("SELECT COUNT(*) FROM courses")
        total = c.fetchone()[0]

        return {"courses": courses, "total": total}

@app.get("/admin/course/{course_name}/lectures")
async def get_course_lectures(
    course_name: str,
    current_user: dict = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0
):
    """Get lectures for a course"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        # First, get course_id from course_name
        c.execute("SELECT id FROM courses WHERE name = ?", (course_name,))
        course = c.fetchone()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        course_id = course['id']

        c.execute("""
            SELECT id, filename, filepath, subject, uploaded_at, 
                   processing_status, total_chunks, total_characters
            FROM lectures 
            WHERE course_id = ?
            ORDER BY uploaded_at DESC LIMIT ? OFFSET ?
        """, (course_id, limit, offset))
        
        lectures = [dict(row) for row in c.fetchall()]
        c.execute("SELECT COUNT(*) FROM lectures WHERE course_id = ?", (course_id,))
        total = c.fetchone()[0]
        
        return {
            "course_name": course_name,
            "lectures": lectures,
            "total": total
        }

@app.get("/admin/lecture-status/{lecture_id}")
async def lecture_status(
    lecture_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Poll for the status of a background lecture processing task."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT processing_status, total_chunks,
                   total_characters, error_message
            FROM lectures WHERE id = ?
        """, (lecture_id,))
        row = c.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Lecture not found")
    
    return {
        "lecture_id": lecture_id,
        "processing_status": row["processing_status"],
        "total_chunks": row["total_chunks"],
        "total_characters": row["total_characters"],
        "error_message": row["error_message"]
    }

@app.delete("/admin/delete-course/{course_id}")
async def delete_course(
    course_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a course"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM courses WHERE id = ?", (course_id,))
        course = c.fetchone()
        
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        course_name = course[0]
        c.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        conn.commit()
        
        print(f"🗑️ Course deleted: {course_name}")
        
        return {
            "success": True,
            "message": f"Course '{course_name}' deleted",
            "course_id": course_id
        }

async def process_lecture_background(filepath: str, subject: str, lecture_id: int):
    """Background task for processing lectures to avoid HTTP timeouts."""
    try:
        with get_db_context() as conn:
            c = conn.cursor()
            
            # Run the CPU-bound processing
            result = await asyncio.to_thread(process_new_pdf, filepath, subject)
            
            if not result['success']:
                raise Exception(result.get('error', 'Processing failed without a specific error message.'))
                
            suggestions_json = json.dumps(result.get('suggestions', []))
            
            # Update the lecture record with the results
            c.execute(
                """UPDATE lectures 
                SET processing_status = 'completed', 
                    total_chunks = ?, 
                    total_characters = ?,
                    suggestions = ?
                WHERE id = ?""",
                (result['total_chunks'], result['total_characters'], suggestions_json, lecture_id)
            )
            conn.commit()
            print(f"✅ Background processing completed for lecture {lecture_id}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Background processing FAILED for lecture {lecture_id}: {error_msg}")
        try:
            with get_db_context() as error_conn:
                # Log the error to the database
                error_conn.execute("UPDATE lectures SET processing_status = 'failed', error_message = ? WHERE id = ?", (error_msg, lecture_id))
                error_conn.commit()
        except Exception as db_err:
            print(f"⚠️ Additionally, failed to log the error to the DB: {db_err}")

# =======================
# Admin Endpoints - Lectures & Stats
# =======================
@app.get("/admin/get-users")
async def get_users(
    current_user: dict = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0
):
    """Get all users"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id, email, role, created_at, is_verified FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset))
        users = [dict(row) for row in c.fetchall()]
        c.execute("SELECT COUNT(*) FROM users")
        total = c.fetchone()[0]

    return {"users": users, "total": total}

@app.delete("/admin/delete-user/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, current_user: dict = Depends(get_current_user)):
    """Deletes a user."""
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Access denied")
    if current_user['user_id'] == user_id:
        raise HTTPException(status_code=400, detail="Admin cannot delete themselves.")

    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        if c.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/admin/get-lectures")
async def get_lectures(current_user: dict = Depends(get_current_user)):
    """Get all lectures"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, filename, subject, uploaded_at, 
                   processing_status, total_chunks, total_characters, error_message
            FROM lectures 
            ORDER BY uploaded_at DESC
        """)
        
        lectures = [dict(row) for row in c.fetchall()]
        
        return {"lectures": lectures}

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

@app.post("/admin/upload-lecture", status_code=status.HTTP_202_ACCEPTED)
async def upload_lecture(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process lecture"""
    
    print(f"\n{'='*60}")
    print(f"📤 Upload Request")
    print(f"   File: {file.filename}")
    print(f"   Course: {subject}")
    print(f"   User: {current_user['user_id']}")
    print(f"{'='*60}\n")
    
    if current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check file size (approximate via header)
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)}MB)")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != '.pdf':
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    if not subject or subject.strip() == "":
        raise HTTPException(status_code=400, detail="Course name required")

    # Verify course exists and get its ID
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT id FROM courses WHERE name = ?", (subject.strip(),))
        course = c.fetchone()
    
    if not course:
        raise HTTPException(status_code=400, detail=f"Course '{subject}' not found. Please create it first.")
    
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    filepath = os.path.join(LECTURES_DIR, unique_filename)
    
    try:
        # Read in chunks to avoid RAM spike
        with open(filepath, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024) # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
                if buffer.tell() > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="File exceeds size limit")
        print(f"✅ File saved: {filepath}")
    except Exception as e:
        print(f"❌ Failed to save: {e}")
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")
    
    lecture_id = None
    
    try:
        with get_db_context() as conn:
            c = conn.cursor()
            c.execute(
                """INSERT INTO lectures 
                (admin_id, course_id, filename, filepath, subject, uploaded_at, processing_status) 
                VALUES (?, ?, ?, ?, ?, datetime('now'), ?)""",
                (current_user['user_id'], course['id'], file.filename, filepath, subject.strip(), 'processing')
            )
            lecture_id = c.lastrowid
            conn.commit()
        
        print(f"✅ Lecture saved to DB: {lecture_id}")
        
        # Offload heavy processing to a background task
        background_tasks.add_task(process_lecture_background, filepath, subject.strip(), lecture_id)
        
        return {
            "success": True,
            "message": "Lecture upload accepted. Processing in background.",
            "lecture_id": lecture_id,
            "status": "processing"
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if lecture_id:
            try:
                with get_db_context() as conn:
                    c = conn.cursor()
                    c.execute(
                        """UPDATE lectures 
                        SET processing_status = 'failed', error_message = ? 
                        WHERE id = ?""",
                        (error_msg, lecture_id)
                    )
                    conn.commit()
            except Exception as db_error:
                print(f"⚠️ Failed to update error: {db_error}")
        
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Processing error: {error_msg}")

@app.delete("/admin/delete-lecture/{lecture_id}")
async def delete_lecture(
    lecture_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Delete a lecture"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        
        c.execute("SELECT filepath FROM lectures WHERE id = ?", (lecture_id,))
        lecture = c.fetchone()
        
        if not lecture:
            raise HTTPException(status_code=404, detail="Lecture not found")
        
        filepath = lecture[0]
        
        c.execute("DELETE FROM lectures WHERE id = ?", (lecture_id,))
        conn.commit()
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"🗑️ Deleted file: {filepath}")
        except Exception as e:
            print(f"⚠️ Could not delete file: {e}")
    
    return {
        "success": True,
        "message": "Lecture deleted",
        "lecture_id": lecture_id
    }

@app.get("/admin/get-stats")
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Get comprehensive statistics"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    with get_db_context() as conn:
        c = conn.cursor()
        # Users
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'student'")
        total_students = c.fetchone()[0]
        
        # Lectures
        c.execute("SELECT COUNT(*) FROM lectures")
        total_lectures = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM lectures WHERE processing_status = 'completed'")
        completed_lectures = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM lectures WHERE processing_status = 'failed'")
        failed_lectures = c.fetchone()[0]
        
        # Courses
        c.execute("SELECT COUNT(*) FROM courses")
        total_courses = c.fetchone()[0]
        
        # Activity
        c.execute("SELECT COUNT(*) FROM conversations WHERE is_deleted = 0")
        total_conversations = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM messages")
        total_messages = c.fetchone()[0]
        
        # Feedback
        c.execute("SELECT COUNT(*) FROM feedbacks WHERE feedback_type = 'positive'")
        positive_feedbacks = c.fetchone()[0]
        
        c.execute("SELECT COUNT(*) FROM feedbacks WHERE feedback_type = 'negative'")
        negative_feedbacks = c.fetchone()[0]
        
        # Content stats
        c.execute("SELECT SUM(total_chunks) FROM lectures WHERE processing_status = 'completed'")
        result = c.fetchone()[0]
        total_chunks = result if result else 0
        
        c.execute("SELECT SUM(total_characters) FROM lectures WHERE processing_status = 'completed'")
        result = c.fetchone()[0]
        total_characters = result if result else 0
        
        # Cache statistics
        cache_stats = {
            "enabled": False,
            "total_entries": 0,
            "hit_rate": 0,
            "total_requests": 0
        }
        
        if semantic_cache:
            try:
                # Get cache info
                info = await asyncio.to_thread(semantic_cache.get_cache_info)
                
                # Get analytics
                analytics = None
                if semantic_cache.enable_analytics:
                    analytics = await asyncio.to_thread(semantic_cache.analytics.get_stats, 7)
                
                cache_stats = {
                    "enabled": True,
                    "total_entries": info.get("total_entries", 0),
                    "threshold": info.get("threshold", 0.90),
                    "ttl_hours": info.get("ttl_hours", 24),
                    "hit_rate": analytics.get("hit_rate", 0) if analytics else 0,
                    "total_requests": analytics.get("total_requests", 0) if analytics else 0,
                    "cache_hits": analytics.get("cache_hits", 0) if analytics else 0,
                    "cache_misses": analytics.get("cache_misses", 0) if analytics else 0
                }
            except Exception as e:
                print(f"⚠️ Error getting cache stats: {e}")
        
        return {
            "stats": {
                "users": {
                    "total_students": total_students
                },
                "courses": {
                    "total": total_courses
                },
                "lectures": {
                    "total": total_lectures,
                    "completed": completed_lectures,
                    "failed": failed_lectures,
                    "processing": total_lectures - completed_lectures - failed_lectures
                },
                "content": {
                    "total_chunks": total_chunks,
                    "total_characters": total_characters
                },
                "activity": {
                    "total_conversations": total_conversations,
                    "total_messages": total_messages
                },
                "feedback": {
                    "positive": positive_feedbacks,
                    "negative": negative_feedbacks,
                    "total": positive_feedbacks + negative_feedbacks
                },
                "cache": cache_stats  # ✅ Enhanced cache stats
            }
        }

# =======================
# User Info
# =======================
@app.get("/user/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    with get_db_context() as conn:
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE id = ?", (current_user['user_id'],))
        user = c.fetchone()
    
    if user:
        current_user['email'] = user['email']
        
    return current_user

# =======================
# Health Check
# =======================
@app.get("/health")
async def health_check():
    services = {}
    overall = "healthy"
    
    # Real DB check
    try:
        with get_db_context() as conn:
            conn.execute("SELECT 1")
        services["database"] = "ok"
    except Exception as e:
        services["database"] = f"error: {str(e)}"
        overall = "degraded"
    
    # Real RAG check
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://127.0.0.1:8001/health")
            services["rag"] = "ok" if r.status_code == 200 else "error"
    except:
        services["rag"] = "offline"
        overall = "degraded"
    
    services["redis"] = "ok" if redis_available else "unavailable"
    
    return JSONResponse(
        status_code=200 if overall == "healthy" else 503,
        content={"status": overall, "services": services}
    )

# =======================
# Run Server
# =======================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting University AI Chatbot API")
    print("="*60)
    print(f"🌐 API URL: http://localhost:8080")
    print(f"📖 Docs: http://localhost:8080/docs")
    print(f"👤 Admin: {ADMIN_EMAIL} / [HIDDEN]")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=7860)