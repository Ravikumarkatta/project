"""
Database connection manager for Bible-AI.
Handles connection pooling, transaction management, and connection configuration.
"""
import os
import time
import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Union, Any, Generator, Tuple

import sqlalchemy
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Thread-local storage for connection pools
_engine_store = threading.local()


class DBConnectionManager:
    """
    Manages database connections with connection pooling and transaction support.
    Supports both SQLite and other SQL databases through SQLAlchemy.
    """
    
    def __init__(self, db_config: Dict[str, Any] = None):
        """
        Initialize the connection manager with the given configuration.
        
        Args:
            db_config: Database configuration containing connection parameters
                       If None, will look for configuration in environment variables
        """
        self.db_config = db_config or self._get_default_config()
        self.db_type = self.db_config.get('type', 'sqlite')
        self.engine = None
        self.Session = None
        
        # Initialize connection pool
        self._initialize_engine()
        
    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine based on configuration."""
        if hasattr(_engine_store, 'engine') and _engine_store.engine:
            self.engine = _engine_store.engine
            self.Session = _engine_store.Session
            return
            
        connection_string = self._build_connection_string()
        
        # Configure pooling options
        pool_size = self.db_config.get('pool_size', 5)
        max_overflow = self.db_config.get('max_overflow', 10)
        pool_recycle = self.db_config.get('pool_recycle', 3600)
        
        # Create engine with appropriate settings
        if self.db_type == 'sqlite':
            # SQLite specific settings
            self.engine = create_engine(
                connection_string,
                connect_args={'check_same_thread': False},
                poolclass=QueuePool,
                pool_pre_ping=True
            )
            
            # Set SQLite pragmas for better performance
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
                
        else:
            # Other database engines (PostgreSQL, MySQL, etc.)
            self.engine = create_engine(
                connection_string,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=pool_recycle,
                pool_pre_ping=True
            )
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        ))
        
        # Store in thread-local storage
        _engine_store.engine = self.engine
        _engine_store.Session = self.Session
        
        logger.info(f"Initialized {self.db_type} connection pool")
    
    def _build_connection_string(self) -> str:
        """
        Build SQLAlchemy connection string from configuration.
        
        Returns:
            Connection string appropriate for the configured database
        """
        if self.db_type == 'sqlite':
            db_path = self.db_config.get('path', 'data/processed/bible.db')
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite:///{db_path}"
            
        elif self.db_type == 'postgresql':
            host = self.db_config.get('host', 'localhost')
            port = self.db_config.get('port', 5432)
            user = self.db_config.get('user', 'postgres')
            password = self.db_config.get('password', '')
            database = self.db_config.get('database', 'bible_ai')
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
            
        elif self.db_type == 'mysql':
            host = self.db_config.get('host', 'localhost')
            port = self.db_config.get('port', 3306)
            user = self.db_config.get('user', 'root')
            password = self.db_config.get('password', '')
            database = self.db_config.get('database', 'bible_ai')
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default database configuration from environment variables.
        
        Returns:
            Dictionary with database configuration
        """
        return {
            'type': os.environ.get('DB_TYPE', 'sqlite'),
            'path': os.environ.get('DB_PATH', 'data/processed/bible.db'),
            'host': os.environ.get('DB_HOST', 'localhost'),
            'port': int(os.environ.get('DB_PORT', 5432)),
            'user': os.environ.get('DB_USER', ''),
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'bible_ai'),
            'pool_size': int(os.environ.get('DB_POOL_SIZE', 5)),
            'max_overflow': int(os.environ.get('DB_MAX_OVERFLOW', 10)),
            'pool_recycle': int(os.environ.get('DB_POOL_RECYCLE', 3600)),
        }
    
    @contextmanager
    def get_session(self) -> Generator:
        """
        Get a database session with transaction support.
        
        Usage:
            with db_manager.get_session() as session:
                session.query(...)
        
        Yields:
            SQLAlchemy session object
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a raw database connection.
        
        Yields:
            SQLAlchemy connection object
        """
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Tuple[bool, Union[Any, Exception]]:
        """
        Execute a raw SQL query with error handling.
        
        Args:
            query: SQL query string
            params: Optional parameters for the query
            
        Returns:
            Tuple of (success, result)
                If success is True, result contains query results
                If success is False, result contains the exception
        """
        params = params or {}
        try:
            with self.get_connection() as conn:
                result = conn.execute(sqlalchemy.text(query), params)
                if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
                    return True, result.fetchall()
                return True, result.rowcount
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return False, e
    
    def execute_transaction(self, queries: list) -> Tuple[bool, Union[Any, Exception]]:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of (query_string, params_dict) tuples
            
        Returns:
            Tuple of (success, result)
                If success is True, result is None
                If success is False, result contains the exception
        """
        try:
            with self.get_connection() as conn:
                with conn.begin():  # Start transaction
                    for query, params in queries:
                        conn.execute(sqlalchemy.text(query), params or {})
            return True, None
        except Exception as e:
            logger.error(f"Transaction execution error: {str(e)}")
            return False, e
    
    def ping(self) -> bool:
        """
        Check if database connection is alive.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                if self.db_type == 'sqlite':
                    conn.execute(sqlalchemy.text("SELECT 1"))
                else:
                    conn.execute(sqlalchemy.text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database ping failed: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("Closed database connection pool")


# Global connection manager instance
_db_manager = None


def get_db_manager(db_config: Dict[str, Any] = None) -> DBConnectionManager:
    """
    Get or create the global database connection manager.
    
    Args:
        db_config: Optional database configuration
        
    Returns:
        Database connection manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DBConnectionManager(db_config)
    return _db_manager


def close_db_connections() -> None:
    """Close all database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None