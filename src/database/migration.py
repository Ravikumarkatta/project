"""
Database migration utilities for Bible-AI.

Provides functionality for schema versioning and database migrations.
"""
import os
import logging
import importlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from sqlalchemy import Column, Integer, String, DateTime, MetaData, Table, text
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Migration table definition
MIGRATION_TABLE_NAME = 'schema_migrations'
MIGRATION_TABLE_DEFINITION = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version VARCHAR(100) NOT NULL UNIQUE,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);
"""

def ensure_migration_table(engine: Engine) -> None:
    """
    Ensure that the migration tracking table exists.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    with engine.connect() as conn:
        conn.execute(text(MIGRATION_TABLE_DEFINITION))
        conn.commit()


def get_applied_migrations(engine: Engine) -> List[str]:
    """
    Get list of already applied migrations.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        List of applied migration versions
    """
    ensure_migration_table(engine)
    
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT version FROM {MIGRATION_TABLE_NAME} ORDER BY id")
        )
        return [row[0] for row in result]


def record_migration(engine: Engine, version: str, description: str) -> None:
    """
    Record a successfully applied migration.
    
    Args:
        engine: SQLAlchemy engine instance
        version: Migration version identifier
        description: Migration description
    """
    with engine.connect() as conn:
        conn.execute(
            text(f"INSERT INTO {MIGRATION_TABLE_NAME} (version, description) VALUES (:version, :description)"),
            {"version": version, "description": description}
        )
        conn.commit()


def get_available_migrations() -> List[Tuple[str, str]]:
    """
    Get list of available migrations from the migrations directory.
    
    Returns:
        List of tuples (version, description)
    """
    migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
    if not os.path.exists(migrations_dir):
        return []
    
    migrations = []
    for filename in sorted(os.listdir(migrations_dir)):
        if filename.endswith('.py') and filename != '__init__.py':
            version = filename[:-3]  # Remove .py extension
            
            # Try to import the migration to get its description
            try:
                module_path = f"src.database.migrations.{version}"
                migration_module = importlib.import_module(module_path)
                description = getattr(migration_module, 'description', 'No description')
                migrations.append((version, description))
            except ImportError:
                logger.warning(f"Could not import migration: {version}")
                migrations.append((version, 'Unknown description'))
    
    return migrations


def run_migrations(engine: Engine) -> List[str]:
    """
    Run all pending migrations.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        List of applied migration versions
    """
    applied_migrations = get_applied_migrations(engine)
    available_migrations = get_available_migrations()
    
    applied = []
    
    for version, description in available_migrations:
        if version in applied_migrations:
            logger.debug(f"Migration {version} already applied, skipping")
            continue
        
        logger.info(f"Applying migration {version}: {description}")
        try:
            # Import and run the migration
            module_path = f"src.database.migrations.{version}"
            migration_module = importlib.import_module(module_path)
            
            # Call the upgrade function
            if hasattr(migration_module, 'upgrade'):
                migration_module.upgrade(engine)
                record_migration(engine, version, description)
                applied.append(version)
                logger.info(f"Migration {version} successfully applied")
            else:
                logger.error(f"Migration {version} has no upgrade function")
        except Exception as e:
            logger.error(f"Failed to apply migration {version}: {str(e)}")
            raise
    
    return applied


def get_migration_status() -> Dict[str, Any]:
    """
    Get the current migration status.
    
    Returns:
        Dictionary with migration status information
    """
    from src.database.connection import get_db_manager
    
    db_manager = get_db_manager()
    engine = db_manager.engine
    
    applied_migrations = get_applied_migrations(engine)
    available_migrations = get_available_migrations()
    
    pending_migrations = [
        version for version, _ in available_migrations 
        if version not in applied_migrations
    ]
    
    return {
        "applied_migrations": applied_migrations,
        "pending_migrations": pending_migrations,
        "available_migrations": available_migrations,
        "is_up_to_date": len(pending_migrations) == 0
    }


def create_migration(name: str) -> str:
    """
    Create a new migration file with the given name.
    
    Args:
        name: Name of the migration (will be prefixed with timestamp)
        
    Returns:
        Path to the created migration file
    """
    migrations_dir = os.path.join(os.path.dirname(__file__), 'migrations')
    os.makedirs(migrations_dir, exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = os.path.join(migrations_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Migrations package for Bible-AI."""\n')
    
    # Create the migration file
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"{timestamp}_{name}.py"
    filepath = os.path.join(migrations_dir, filename)
    
    template = '''"""
{name} migration.
"""

description = "{name}"

def upgrade(engine):
    """Apply the migration."""
    # Apply schema changes
    # Example:
    # from sqlalchemy import text
    # with engine.connect() as conn:
    #     conn.execute(text("ALTER TABLE my_table ADD COLUMN new_column TEXT"))
    #     conn.commit()
    pass


def downgrade(engine):
    """Revert the migration."""
    # Revert schema changes
    # Example:
    # from sqlalchemy import text
    # with engine.connect() as conn:
    #     conn.execute(text("ALTER TABLE my_table DROP COLUMN new_column"))
    #     conn.commit()
    pass
'''.format(name=name)
    
    with open(filepath, 'w') as f:
        f.write(template)
    
    logger.info(f"Created migration file: {filepath}")
    return filepath