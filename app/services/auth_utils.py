# app/services/auth_utils.py

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# Use pbkdf2_sha256 instead of bcrypt to avoid bcrypt/version issues on Railway
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """
    Hash a plain-text password for storage.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain-text password against the stored hash.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: str,
    tenant: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    subject: typically user_id or email
    tenant: the tenant code/schema identifier you use (e.g. 'tenant_test_client')
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.utcnow() + expires_delta
    to_encode = {
        "sub": subject,
        "tenant": tenant,
        "exp": expire,
    }
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Raises JWTError if invalid/expired.
    """
    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    return payload