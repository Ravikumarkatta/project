"""
SQL schema definitions for Bible-AI.
Contains SQLAlchemy model definitions for all database tables.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, 
    DateTime, ForeignKey, Index, UniqueConstraint, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.schema import MetaData

# SQLAlchemy naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=convention)
Base = declarative_base(metadata=metadata)


class Bible(Base):
    """Bible translation metadata."""
    __tablename__ = 'bibles'
    
    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    abbreviation = Column(String(20), nullable=False, unique=True)
    language = Column(String(50), nullable=False)
    version = Column(String(50), nullable=False)
    copyright = Column(Text)
    publisher = Column(String(255))
    year_published = Column(Integer)
    description = Column(Text)
    is_public_domain = Column(Boolean, default=False)
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    books = relationship("Book", back_populates="bible", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Bible(id={self.id}, name='{self.name}', abbreviation='{self.abbreviation}')>"


class Book(Base):
    """Book of the Bible."""
    __tablename__ = 'books'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bible_id = Column(String(36), ForeignKey('bibles.id', ondelete='CASCADE'), nullable=False)
    position = Column(Integer, nullable=False)  # Book order in the Bible
    name = Column(String(100), nullable=False)
    short_name = Column(String(20), nullable=False)
    testament = Column(String(10), nullable=False)  # 'old' or 'new'
    category = Column(String(50))  # e.g., 'Law', 'History', 'Pauline Epistles'
    chapters_count = Column(Integer, nullable=False)
    
    # Relationships
    bible = relationship("Bible", back_populates="books")
    chapters = relationship("Chapter", back_populates="book", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('bible_id', 'position', name='uq_book_bible_position'),
        UniqueConstraint('bible_id', 'name', name='uq_book_bible_name'),
        Index('ix_books_bible_id', 'bible_id'),
    )
    
    def __repr__(self) -> str:
        return f"<Book(id={self.id}, name='{self.name}', testament='{self.testament}')>"


class Chapter(Base):
    """Chapter in a book of the Bible."""
    __tablename__ = 'chapters'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(Integer, ForeignKey('books.id', ondelete='CASCADE'), nullable=False)
    number = Column(Integer, nullable=False)
    title = Column(String(255))
    verses_count = Column(Integer, nullable=False)
    
    # Relationships
    book = relationship("Book", back_populates="chapters")
    verses = relationship("Verse", back_populates="chapter", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('book_id', 'number', name='uq_chapter_book_number'),
        Index('ix_chapters_book_id', 'book_id'),
    )
    
    def __repr__(self) -> str:
        return f"<Chapter(id={self.id}, book_id={self.book_id}, number={self.number})>"


class Verse(Base):
    """Individual verse in the Bible."""
    __tablename__ = 'verses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chapter_id = Column(Integer, ForeignKey('chapters.id', ondelete='CASCADE'), nullable=False)
    number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    is_words_of_christ = Column(Boolean, default=False)
    
    # Relationships
    chapter = relationship("Chapter", back_populates="verses")
    cross_references_from = relationship(
        "CrossReference", 
        foreign_keys="[CrossReference.from_verse_id]", 
        back_populates="from_verse"
    )
    cross_references_to = relationship(
        "CrossReference", 
        foreign_keys="[CrossReference.to_verse_id]", 
        back_populates="to_verse"
    )
    
    __table_args__ = (
        UniqueConstraint('chapter_id', 'number', name='uq_verse_chapter_number'),
        Index('ix_verses_chapter_id', 'chapter_id'),
    )
    
    def __repr__(self) -> str:
        return f"<Verse(id={self.id}, chapter_id={self.chapter_id}, number={self.number})>"


class CrossReference(Base):
    """Cross-reference between verses."""
    __tablename__ = 'cross_references'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    from_verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    to_verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    reference_type = Column(String(50))  # e.g., 'parallel', 'quotation', 'allusion'
    confidence = Column(Float, default=1.0)  # 0.0-1.0 confidence score
    source = Column(String(100))  # e.g., 'manual', 'computed', 'openBible'
    notes = Column(Text)
    
    # Relationships
    from_verse = relationship("Verse", foreign_keys=[from_verse_id], back_populates="cross_references_from")
    to_verse = relationship("Verse", foreign_keys=[to_verse_id], back_populates="cross_references_to")
    
    __table_args__ = (
        UniqueConstraint('from_verse_id', 'to_verse_id', name='uq_crossref_from_to'),
        Index('ix_cross_references_from_verse_id', 'from_verse_id'),
        Index('ix_cross_references_to_verse_id', 'to_verse_id'),
    )
    
    def __repr__(self) -> str:
        return f"<CrossReference(id={self.id}, from={self.from_verse_id}, to={self.to_verse_id})>"


class VerseEmbedding(Base):
    """Vector embeddings for verses to support semantic search."""
    __tablename__ = 'verse_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False, unique=True)
    embedding = Column(Text, nullable=False)  # Serialized embedding vector
    model_version = Column(String(100), nullable=False)  # Model used to generate embedding
    dimensions = Column(Integer, nullable=False)  # Number of dimensions in embedding
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_verse_embeddings_verse_id', 'verse_id'),
    )
    
    def __repr__(self) -> str:
        return f"<VerseEmbedding(id={self.id}, verse_id={self.verse_id}, model='{self.model_version}')>"


class Commentary(Base):
    """Commentary metadata."""
    __tablename__ = 'commentaries'
    
    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=False)
    year_published = Column(Integer)
    tradition = Column(String(100))  # e.g., 'Reformed', 'Catholic', 'Orthodox'
    publisher = Column(String(255))
    description = Column(Text)
    is_public_domain = Column(Boolean, default=False)
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    entries = relationship("CommentaryEntry", back_populates="commentary", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Commentary(id={self.id}, title='{self.title}', author='{self.author}')>"


class CommentaryEntry(Base):
    """Entry in a commentary for a specific verse or passage."""
    __tablename__ = 'commentary_entries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    commentary_id = Column(String(36), ForeignKey('commentaries.id', ondelete='CASCADE'), nullable=False)
    start_verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    end_verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    text = Column(Text, nullable=False)
    
    # Relationships
    commentary = relationship("Commentary", back_populates="entries")
    start_verse = relationship("Verse", foreign_keys=[start_verse_id])
    end_verse = relationship("Verse", foreign_keys=[end_verse_id])
    
    __table_args__ = (
        Index('ix_commentary_entries_commentary_id', 'commentary_id'),
        Index('ix_commentary_entries_start_verse_id', 'start_verse_id'),
    )
    
    def __repr__(self) -> str:
        return f"<CommentaryEntry(id={self.id}, commentary_id='{self.commentary_id}', start_verse_id={self.start_verse_id})>"


class LexiconEntry(Base):
    """Original language lexicon entry."""
    __tablename__ = 'lexicon_entries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strong_number = Column(String(10), nullable=False, unique=True)
    language = Column(String(20), nullable=False)  # 'hebrew' or 'greek'
    transliteration = Column(String(100))
    original_word = Column(String(100))
    definition = Column(Text)
    part_of_speech = Column(String(50))
    
    # Relationships
    occurrences = relationship("WordOccurrence", back_populates="lexicon_entry", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_lexicon_entries_strong_number', 'strong_number'),
        Index('ix_lexicon_entries_language', 'language'),
    )
    
    def __repr__(self) -> str:
        return f"<LexiconEntry(id={self.id}, strong='{self.strong_number}', word='{self.original_word}')>"


class WordOccurrence(Base):
    """Occurrence of an original language word in a verse."""
    __tablename__ = 'word_occurrences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    lexicon_entry_id = Column(Integer, ForeignKey('lexicon_entries.id', ondelete='CASCADE'), nullable=False)
    position = Column(Integer, nullable=False)  # Position in the verse
    form = Column(String(100))  # Actual form as it appears in text
    morphology = Column(String(100))  # Grammatical form
    
    # Relationships
    lexicon_entry = relationship("LexiconEntry", back_populates="occurrences")
    
    __table_args__ = (
        Index('ix_word_occurrences_verse_id', 'verse_id'),
        Index('ix_word_occurrences_lexicon_entry_id', 'lexicon_entry_id'),
    )
    
    def __repr__(self) -> str:
        return f"<WordOccurrence(id={self.id}, verse_id={self.verse_id}, entry_id={self.lexicon_entry_id})>"


class HistoricalContext(Base):
    """Historical context information for passages."""
    __tablename__ = 'historical_contexts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    start_verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    end_verse_id = Column(Integer, ForeignKey('verses.id', ondelete='CASCADE'), nullable=False)
    time_period = Column(String(100))
    location = Column(String(100))
    culture = Column(String(100))
    historical_event = Column(String(255))
    description = Column(Text)
    
    __table_args__ = (
        Index('ix_historical_contexts_start_verse_id', 'start_verse_id'),
    )
    
    def __repr__(self) -> str:
        return f"<HistoricalContext(id={self.id}, start_verse_id={self.start_verse_id}, period='{self.time_period}')>"


class SearchLog(Base):
    """Log of user searches for analytics."""
    __tablename__ = 'search_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    query_type = Column(String(50))  # e.g., 'text', 'verse', 'topic'
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_hash = Column(String(64))  # Hashed IP for privacy
    session_id = Column(String(64))
    result_count = Column(Integer)
    execution_time_ms = Column(Integer)
    
    __table_args__ = (
        Index('ix_search_logs_timestamp', 'timestamp'),
        Index('ix_search_logs_query_type', 'query_type'),
    )
    
    def __repr__(self) -> str:
        return f"<SearchLog(id={self.id}, query='{self.query[:20]}...', timestamp={self.timestamp})>"


class TheologicalTopic(Base):
    """Theological topic or doctrine."""
    __tablename__ = 'theological_topics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    parent_id = Column(Integer, ForeignKey('theological_topics.id', ondelete='SET NULL'))
    
    # Self-referential relationship for hierarchical topics
    subtopics = relationship(
        "TheologicalTopic",
        backref="parent",
        remote_side=[id]
    )
    
    # Many-to-many relationship with verses
    verses = relationship(
        "Verse",
        secondary="topic_verse_associations",
        backref="topics"
    )
    
    def __repr__(self) -> str:
        return f"<TheologicalTopic(id={self.id}, name='{self.name}')>"


# Association table for many-to-many relationship between topics and verses
topic_verse_association = Table(
    'topic_verse_associations',
    Base.metadata,
    Column('topic_id', Integer, ForeignKey('theological_topics.id', ondelete='CASCADE'), primary_key=True),
    Column('verse_id', Integer, ForeignKey('verses.id', ondelete='CASCADE'), primary_key=True),
    Column('relevance', Float, default=1.0),  # 0.0-1.0 relevance score
)


# Functions to create and drop all tables

def create_tables(engine):
    """Create all tables defined in the schema."""
    Base.metadata.create_all(engine)


def drop_tables(engine):
    """Drop all tables defined in the schema."""
    Base.metadata.drop_all(engine)