import argparse
import asyncio
import os

from sqlalchemy import select

from app.db import SessionFactory
from app.models import User
from app.security import hash_password


async def create_initial_admin() -> None:
    email = os.environ.get("INITIAL_ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("INITIAL_ADMIN_PASSWORD", "")
    if not email or len(password) < 12:
        raise SystemExit(
            "INITIAL_ADMIN_EMAIL and a 12+ character "
            "INITIAL_ADMIN_PASSWORD are required"
        )
    async with SessionFactory() as session:
        existing = await session.scalar(
            select(User).where(User.email == email)
        )
        if existing:
            print("Initial administrator already exists; no changes made.")
            return
        session.add(
            User(
                email=email,
                password_hash=hash_password(password),
                role="admin",
                is_active=True,
                must_change_password=False,
            )
        )
        await session.commit()
        print("Initial administrator created.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create-initial-admin"])
    args = parser.parse_args()
    if args.command == "create-initial-admin":
        asyncio.run(create_initial_admin())


if __name__ == "__main__":
    main()
