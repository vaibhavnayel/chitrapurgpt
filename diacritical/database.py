"""
Database module for the Diacritical application.
Handles database models and connections.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize SQLAlchemy
db = SQLAlchemy()

class DiacriticMapping(db.Model):
    """
    Database model for diacritic mappings.
    Maps plain text to text with diacritics.
    """
    __tablename__ = 'diacritic_mappings'
    
    id = db.Column(db.Integer, primary_key=True)
    plain_text = db.Column(db.String(255), unique=True, nullable=False)
    diacritic_text = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<DiacriticMapping {self.plain_text} -> {self.diacritic_text}>"
    
    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'plain_text': self.plain_text,
            'diacritic_text': self.diacritic_text,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Feedback(db.Model):
    """
    Database model for user feedback.
    """
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    email = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<Feedback {self.id}: {self.message[:30]}...>"
    
    def to_dict(self):
        """Convert the model instance to a dictionary"""
        return {
            'id': self.id,
            'message': self.message,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

def init_db(app, retries=None, delay=None):
    """Initialize the database with the Flask app.

    The table check connects to Postgres at import time. On Render a resumed service can start
    before its database has finished resuming, so retry for a while instead of crashing the
    worker; if the database still isn't reachable, start anyway (the tables already exist in
    production, and pool_pre_ping reconnects once the database is up).
    """
    import os
    import time
    import sqlalchemy as sa

    retries = int(os.environ.get('DB_CONNECT_RETRIES', 12)) if retries is None else retries
    delay = float(os.environ.get('DB_CONNECT_DELAY', 5)) if delay is None else delay

    app.config.setdefault('SQLALCHEMY_ENGINE_OPTIONS', {}).setdefault('pool_pre_ping', True)
    db.init_app(app)

    with app.app_context():
        for attempt in range(1, retries + 1):
            try:
                inspector = sa.inspect(db.engine)
                if not inspector.has_table('diacritic_mappings') or not inspector.has_table('feedback'):
                    db.create_all()
                    logger.info("Created database tables")
                else:
                    logger.info("Database tables already exist")
                return
            except sa.exc.OperationalError as exc:
                db.engine.dispose()
                if attempt == retries:
                    logger.error("Database still unreachable after %d attempts; starting without the "
                                 "table check: %s", retries, exc.__class__.__name__)
                    return
                logger.warning("Database not reachable (attempt %d/%d), retrying in %.0fs",
                               attempt, retries, delay)
                time.sleep(delay)

def get_db():
    """Get the database instance"""
    return db 