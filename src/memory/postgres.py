import logging
from typing import Optional
from urllib.parse import quote

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore

from core.settings import settings

logger = logging.getLogger(__name__)


class PostgresConnectionManager:
    """Singleton class to manage PostgreSQL connection pool and related components"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.pool: Optional[AsyncConnectionPool] = None
            self.saver: Optional[AsyncPostgresSaver] = None
            self.store: Optional[AsyncPostgresStore] = None
            self.initialized = True
    
    def get_connection_string(self) -> str:
        """Build PostgreSQL connection string.

        POSTGRES_URL is used as given when set (the host decides its own SSL
        settings). Otherwise the URL is built from the five POSTGRES_* parts.
        """
        if settings.POSTGRES_URL is not None and settings.POSTGRES_URL.get_secret_value():
            return settings.POSTGRES_URL.get_secret_value()

        if settings.POSTGRES_PASSWORD is None:
            raise ValueError("POSTGRES_PASSWORD is not set")

        encoded_password = quote(settings.POSTGRES_PASSWORD.get_secret_value(), safe='')

        return (
            f"postgresql://{settings.POSTGRES_USER}:"
            f"{encoded_password}@"
            f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
            f"{settings.POSTGRES_DB}?sslmode={settings.POSTGRES_SSLMODE}"
        )

    async def check_encoding(self) -> None:
        """Refuse to run on a database that is not UTF8.

        On a SQL_ASCII database the driver returns thread ids as bytes. The
        checkpointer then writes the next turn under a different thread id, so
        every turn after the first is silently lost. Failing at startup is safer.
        """
        async with self.pool.connection() as conn:
            cur = await conn.execute("SHOW server_encoding")
            row = await cur.fetchone()
        encoding = row["server_encoding"] if isinstance(row, dict) else row[0]
        if isinstance(encoding, bytes):  # exactly what a SQL_ASCII database returns
            encoding = encoding.decode("ascii", "replace")
        if encoding.upper() != "UTF8":
            raise ValueError(
                f"PostgreSQL database encoding is {encoding}, but UTF8 is required. "
                "Create the database with ENCODING 'UTF8'."
            )
    
    async def setup(self):
        """Initialize connection pool and related components"""
        if self.pool is not None:
            logger.warning("Connection pool already initialized")
            return
        
        validate_postgres_config()
        conn_string = self.get_connection_string()
        
        logger.info("Initializing PostgreSQL connection pool...")

        # Connection pool configuration
        connection_kwargs = {
            "autocommit": True,
            "row_factory": dict_row,
            # Add connection health check parameters
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        # Create connection pool - use open=False to avoid deprecation warning
        self.pool = AsyncConnectionPool(
            conninfo=conn_string,
            min_size=2,
            max_size=10,
            timeout=30.0,
            max_lifetime=3600.0,  # 1 hour (increased from 5 minutes)
            max_idle=600.0,  # 10 minutes (increased from 60 seconds)
            kwargs=connection_kwargs,
            open=False,  # Don't open connection pool in constructor
            # Add connection check callback
            check=AsyncConnectionPool.check_connection,
        )
        
        # Explicitly open connection pool
        await self.pool.open()
        try:
            await self.check_encoding()
        except Exception:
            await self.pool.close()
            self.pool = None
            raise
        
        # Initialize saver and store
        self.saver = AsyncPostgresSaver(self.pool)
        self.store = AsyncPostgresStore(self.pool)

        # Set up database tables
        # Note: Skip setup if tables already exist and user lacks CREATE permission
        try:
            await self.saver.setup()
            await self.store.setup()
            logger.info("Database tables setup completed")
        except Exception as e:
            if "permission denied" in str(e).lower():
                logger.warning("Skipping table setup (user lacks CREATE permission, checking if tables exist)")
                # Verify required tables exist
                async with self.pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute("""
                            SELECT COUNT(*) as count FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name IN ('checkpoints', 'checkpoint_migrations', 'checkpoint_writes', 'checkpoint_blobs', 'store', 'store_migrations')
                        """)
                        result = await cur.fetchone()
                        count = result['count']
                        if count >= 6:
                            logger.info(f"✓ Verified {count}/6 required tables exist, proceeding without setup")
                        else:
                            raise ValueError(f"Only {count}/6 required tables exist, but cannot create missing tables due to insufficient permissions")
            else:
                raise
        
        logger.info("PostgreSQL connection pool initialized successfully")
    
    async def cleanup(self):
        """Clean up connection pool"""
        if self.pool:
            logger.info("Closing PostgreSQL connection pool...")
            await self.pool.close()
            self.pool = None
            self.saver = None
            self.store = None
            logger.info("PostgreSQL connection pool closed")
    
    def get_saver(self) -> AsyncPostgresSaver:
        """Get saver instance"""
        if self.saver is None:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")
        return self.saver
    
    def get_store(self) -> AsyncPostgresStore:
        """Get store instance"""
        if self.store is None:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")
        return self.store


# Keep original functions for backward compatibility
def validate_postgres_config() -> None:
    """Validate PostgreSQL configuration"""
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD", 
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]
    
    if settings.POSTGRES_URL is not None and settings.POSTGRES_URL.get_secret_value():
        return  # one URL replaces the five parts

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "Set POSTGRES_URL, or all five of these, to use PostgreSQL persistence."
        )


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    # Use connection manager's method
    return pg_manager.get_connection_string()


# Global connection manager instance
pg_manager = PostgresConnectionManager()


def get_postgres_saver():
    """Get PostgreSQL saver (maintain backward compatibility)"""
    return pg_manager.get_saver()


def get_postgres_store():
    """Get PostgreSQL store (maintain backward compatibility)"""
    return pg_manager.get_store()