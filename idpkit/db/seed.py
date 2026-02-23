"""Seed default admin user on first startup."""

import logging
import os

from sqlalchemy import func, select

from idpkit.db.models import User, UserRole

logger = logging.getLogger(__name__)


async def seed_default_admin(session_factory):
    """Create a default admin user if no users exist in the database."""
    async with session_factory() as session:
        result = await session.execute(select(func.count(User.id)))
        user_count = result.scalar()

        if user_count > 0:
            logger.debug("Users already exist (%d), skipping admin seed.", user_count)
            return

        from idpkit.api.deps import hash_password

        password = os.getenv("IDP_ADMIN_PASSWORD", "admin123")
        admin = User(
            username="admin",
            hashed_password=hash_password(password),
            email=None,
            role=UserRole.ADMIN.value,
            is_active=1,
        )
        session.add(admin)
        await session.commit()
        logger.info(
            "Default admin user created (username=admin). "
            "Change the password via IDP_ADMIN_PASSWORD env var."
        )
