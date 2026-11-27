"""Email/password sign-up and revocable cookie sessions."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db, get_session_factory
from app.models import User, UserSession


router = APIRouter(prefix="/auth", tags=["Authentication"])
SESSION_DAYS = 7


def cookie_name() -> str:
    """Use the host-only cookie prefix when HTTPS is required in production."""
    return "__Host-source_atlas_session" if settings.app_env == "production" else "source_atlas_session"


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        """Normalize email before lookup and duplicate-account checks."""
        value = value.strip().lower()
        if value.count("@") != 1 or not value.split("@", 1)[0] or "." not in value.split("@", 1)[1]:
            raise ValueError("Enter a valid email address.")
        return value


class Account(BaseModel):
    id: UUID
    email: str


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Store the scrypt parameters and salt beside the password digest."""
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=5, dklen=32)
    return f"scrypt$16384$8$5${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Recompute the stored hash and compare it without timing leaks."""
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt),
                                n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    """Keep only a hash of the browser's session token in the database."""
    return hashlib.sha256(token.encode()).hexdigest()


DUMMY_PASSWORD_HASH = hash_password("dummy-password-123", b"source-atlas-demo")
# Unknown accounts still perform password work, so sign-in timing reveals less.


def check_origin(request: Request) -> None:
    """Reject cross-origin browser writes on the public HTTPS deployment."""
    if settings.app_env != "production" or request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != "https" or parsed.netloc != request.headers.get("host"):
            raise HTTPException(status_code=403, detail="Invalid request origin.")


async def get_current_user_id(
    request: Request,
) -> UUID | None:
    """Require a valid session in production; preserve local development tooling."""
    check_origin(request)
    token = request.cookies.get(cookie_name())
    if not token or len(token) > 256:
        # Local tools can run without an account; deployed users need a session.
        if settings.app_env != "production":
            return None
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    try:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(UserSession.user_id).where(
                    UserSession.token_hash == token_hash(token),
                    UserSession.expires_at > datetime.now(timezone.utc),
                )
            )
            user_id = result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Account service unavailable.") from exc
    if user_id is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user_id


async def issue_session(response: Response, session: AsyncSession, user: User) -> Account:
    """Persist a revocable session and send its raw token only in a cookie."""
    token = secrets.token_urlsafe(32)
    session.add(UserSession(token_hash=token_hash(token), user_id=user.id,
                            expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)))
    await session.commit()
    response.set_cookie(cookie_name(), token, max_age=SESSION_DAYS * 86400,
                        httponly=True, secure=settings.app_env == "production",
                        samesite="strict", path="/")
    response.headers["Cache-Control"] = "no-store"
    return Account(id=user.id, email=user.email)


@router.post("/signup", response_model=Account, status_code=201)
async def signup(credentials: Credentials, response: Response, request: Request,
                 session: Annotated[AsyncSession, Depends(get_db)]) -> Account:
    """Create an account and sign it in after the database accepts it."""
    check_origin(request)
    email = str(credentials.email).strip().lower()
    password = await anyio.to_thread.run_sync(hash_password, credentials.password)
    user = User(email=email, password_hash=password)
    try:
        session.add(user)
        await session.flush()
        return await issue_session(response, session, user)
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc


@router.post("/signin", response_model=Account)
async def signin(credentials: Credentials, response: Response, request: Request,
                 session: Annotated[AsyncSession, Depends(get_db)]) -> Account:
    """Authenticate credentials while doing equal work for unknown emails."""
    check_origin(request)
    result = await session.execute(select(User).where(User.email == str(credentials.email).strip().lower()))
    user = result.scalar_one_or_none()
    valid = await anyio.to_thread.run_sync(verify_password, credentials.password,
                                           user.password_hash if user else DUMMY_PASSWORD_HASH)
    if user is None or not valid:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return await issue_session(response, session, user)


@router.get("/me", response_model=Account)
async def me(user_id: Annotated[UUID | None, Depends(get_current_user_id)],
             session: Annotated[AsyncSession, Depends(get_db)]) -> Account:
    """Return the signed-in account used to restore the browser session."""
    if user_id is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return Account(id=user.id, email=user.email)


@router.post("/signout", status_code=204)
async def signout(request: Request, response: Response,
                  session: Annotated[AsyncSession, Depends(get_db)]) -> None:
    """Revoke the stored token before clearing the browser cookie."""
    check_origin(request)
    token = request.cookies.get(cookie_name())
    if token:
        row = await session.get(UserSession, token_hash(token))
        if row is not None:
            await session.delete(row)
            await session.commit()
    response.delete_cookie(cookie_name(), path="/")
