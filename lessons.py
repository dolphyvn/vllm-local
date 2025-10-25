"""
lessons.py - Self-improving lesson storage and retrieval system
Stores lessons learned, corrections, and feedback for reasoning improvement
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)

# Lesson data models
class Lesson(BaseModel):
    id: str
    title: str
    content: str
    category: str
    confidence: float
    created_at: str
    updated_at: str
    source_conversation_id: Optional[str] = None
    tags: List[str] = []
    applications_count: int = 0
    success_rate: float = 0.0

class UserFeedback(BaseModel):
    id: str
    lesson_id: str
    rating: int  # 1-5 stars
    feedback_text: str
    helpful: bool
    created_at: str
    user_context: Dict[str, Any] = {}

class Correction(BaseModel):
    id: str
    original_response: str
    corrected_response: str
    correction_reason: str
    lesson_derived: str
    created_at: str
    conversation_id: str
    effectiveness_score: float = 0.0

class LessonApplication(BaseModel):
    id: str
    lesson_id: str
    conversation_id: str
    application_context: str
    outcome: str  # "success", "partial", "failure"
    effectiveness_rating: Optional[int] = None
    created_at: str

class LessonManager:
    """
    Manages lesson storage, retrieval, and self-improvement logic
    """

    def __init__(self, db_path: str = "./lessons.db"):
        self.db_path = db_path
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of database"""
        if self._initialized:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS lessons (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        confidence REAL DEFAULT 0.0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        source_conversation_id TEXT,
                        tags TEXT DEFAULT '[]',
                        applications_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id TEXT PRIMARY KEY,
                        lesson_id TEXT NOT NULL,
                        rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                        feedback_text TEXT,
                        helpful BOOLEAN NOT NULL,
                        created_at TEXT NOT NULL,
                        user_context TEXT DEFAULT '{}',
                        FOREIGN KEY (lesson_id) REFERENCES lessons (id)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS corrections (
                        id TEXT PRIMARY KEY,
                        original_response TEXT NOT NULL,
                        corrected_response TEXT NOT NULL,
                        correction_reason TEXT NOT NULL,
                        lesson_derived TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        effectiveness_score REAL DEFAULT 0.0
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS lesson_applications (
                        id TEXT PRIMARY KEY,
                        lesson_id TEXT NOT NULL,
                        conversation_id TEXT NOT NULL,
                        application_context TEXT NOT NULL,
                        outcome TEXT NOT NULL CHECK (outcome IN ('success', 'partial', 'failure')),
                        effectiveness_rating INTEGER CHECK (effectiveness_rating >= 1 AND effectiveness_rating <= 5),
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (lesson_id) REFERENCES lessons (id)
                    )
                """)

                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lessons_category ON lessons(category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lessons_created_at ON lessons(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_lessons_confidence ON lessons(confidence)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_lesson_id ON user_feedback(lesson_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_corrections_conversation_id ON corrections(conversation_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_applications_lesson_id ON lesson_applications(lesson_id)")

                conn.commit()

            self._initialized = True
            logger.info(f"Lesson database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize lesson database: {e}")
            self._initialized = False

    def add_lesson(self,
                   title: str,
                   content: str,
                   category: str,
                   confidence: float = 0.5,
                   source_conversation_id: Optional[str] = None,
                   tags: List[str] = None) -> str:
        """Add a new lesson to the database"""
        try:
            self._ensure_initialized()
            if not self._initialized:
                logger.warning("Lesson database not available")
                return ""

            lesson_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO lessons
                    (id, title, content, category, confidence, created_at, updated_at,
                     source_conversation_id, tags, applications_count, success_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    lesson_id, title, content, category, confidence, now, now,
                    source_conversation_id, json.dumps(tags or []), 0, 0.0
                ))
                conn.commit()

            logger.info(f"Added lesson: {title} (ID: {lesson_id})")
            return lesson_id

        except Exception as e:
            logger.error(f"Failed to add lesson: {e}")
            return ""

    def add_feedback(self, lesson_id: str, rating: int, feedback_text: str = "",
                    helpful: bool = True, user_context: Dict[str, Any] = None) -> str:
        """Add user feedback for a lesson"""
        try:
            self._ensure_initialized()
            if not self._initialized:
                return ""

            feedback_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_feedback
                    (id, lesson_id, rating, feedback_text, helpful, created_at, user_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_id, lesson_id, rating, feedback_text, helpful, now,
                    json.dumps(user_context or {})
                ))
                conn.commit()

            logger.info(f"Added feedback for lesson {lesson_id}: {rating}/5")
            return feedback_id

        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return ""

    def add_correction(self, original_response: str, corrected_response: str,
                      correction_reason: str, lesson_derived: str,
                      conversation_id: str) -> str:
        """Add a correction with derived lesson"""
        try:
            self._ensure_initialized()
            if not self._initialized:
                return ""

            correction_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO corrections
                    (id, original_response, corrected_response, correction_reason,
                     lesson_derived, created_at, conversation_id, effectiveness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    correction_id, original_response, corrected_response,
                    correction_reason, lesson_derived, now, conversation_id, 0.0
                ))
                conn.commit()

            logger.info(f"Added correction: {correction_reason[:50]}...")
            return correction_id

        except Exception as e:
            logger.error(f"Failed to add correction: {e}")
            return ""

    def record_lesson_application(self, lesson_id: str, conversation_id: str,
                                application_context: str, outcome: str,
                                effectiveness_rating: Optional[int] = None) -> str:
        """Record when a lesson is applied and its effectiveness"""
        try:
            self._ensure_initialized()
            if not self._initialized:
                return ""

            application_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO lesson_applications
                    (id, lesson_id, conversation_id, application_context, outcome,
                     effectiveness_rating, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (application_id, lesson_id, conversation_id,
                      application_context, outcome, effectiveness_rating, now))

                # Update lesson statistics
                cursor.execute("""
                    UPDATE lessons
                    SET applications_count = applications_count + 1,
                        updated_at = ?
                    WHERE id = ?
                """, (now, lesson_id))

                conn.commit()

            logger.info(f"Recorded lesson application: {lesson_id} -> {outcome}")
            return application_id

        except Exception as e:
            logger.error(f"Failed to record lesson application: {e}")
            return ""

    def get_relevant_lessons(self, query_text: str, category: str = None,
                            max_lessons: int = 5) -> List[Dict[str, Any]]:
        """Get relevant lessons based on query and optional category filter"""
        try:
            self._ensure_initialized()
            if not self._initialized:
                return []

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                base_query = """
                    SELECT l.*,
                           AVG(uf.rating) as avg_rating,
                           COUNT(uf.id) as feedback_count,
                           AVG(la.effectiveness_rating) as avg_effectiveness
                    FROM lessons l
                    LEFT JOIN user_feedback uf ON l.id = uf.lesson_id
                    LEFT JOIN lesson_applications la ON l.id = la.lesson_id
                """

                conditions = []
                params = []

                if category:
                    conditions.append("l.category = ?")
                    params.append(category)

                # Add text search (simple SQLite FTS-like behavior)
                if query_text:
                    # Split query into keywords and search in title and content
                    keywords = query_text.lower().split()
                    for keyword in keywords:
                        if len(keyword) > 2:  # Skip very short words
                            conditions.append("(LOWER(l.title) LIKE ? OR LOWER(l.content) LIKE ?)")
                            params.extend([f"%{keyword}%", f"%{keyword}%"])

                if conditions:
                    base_query += " WHERE " + " AND ".join(conditions)

                base_query += """
                    GROUP BY l.id
                    ORDER BY l.confidence DESC, avg_rating DESC, l.success_rate DESC
                    LIMIT ?
                """
                params.append(max_lessons)

                cursor.execute(base_query, params)
                rows = cursor.fetchall()

                lessons = []
                for row in rows:
                    lesson = dict(row)
                    # Parse JSON fields
                    lesson['tags'] = json.loads(lesson['tags'])
                    lesson['relevance_score'] = self._calculate_relevance_score(query_text, lesson)
                    lessons.append(lesson)

                # Sort by relevance score
                lessons.sort(key=lambda x: x['relevance_score'], reverse=True)

                logger.info(f"Retrieved {len(lessons)} relevant lessons for query: {query_text[:50]}...")
                return lessons

        except Exception as e:
            logger.error(f"Failed to get relevant lessons: {e}")
            return []

    def _calculate_relevance_score(self, query_text: str, lesson: Dict[str, Any]) -> float:
        """Calculate relevance score for lesson ranking"""
        score = 0.0
        query_words = set(query_text.lower().split())

        # Title relevance (higher weight)
        title_words = set(lesson['title'].lower().split())
        title_overlap = len(query_words & title_words)
        score += title_overlap * 0.3

        # Content relevance
        content_words = set(lesson['content'].lower().split())
        content_overlap = len(query_words & content_words)
        score += content_overlap * 0.2

        # Category relevance
        if lesson['category'].lower() in query_text.lower():
            score += 0.5

        # Quality factors
        score += lesson['confidence'] * 0.3
        score += lesson.get('avg_rating', 0) * 0.1
        score += lesson.get('success_rate', 0) * 0.2

        return score

    def get_lesson_statistics(self) -> Dict[str, Any]:
        """Get comprehensive lesson statistics"""
        try:
            self._ensure_initialized()
            if not self._initialized:
                return {}

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM lessons")
                total_lessons = cursor.fetchone()[0]

                cursor.execute("SELECT category, COUNT(*) FROM lessons GROUP BY category")
                lessons_by_category = dict(cursor.fetchall())

                cursor.execute("""
                    SELECT AVG(rating), COUNT(*)
                    FROM user_feedback
                """)
                avg_feedback = cursor.fetchone()

                cursor.execute("""
                    SELECT AVG(effectiveness_rating), COUNT(*)
                    FROM lesson_applications
                    WHERE effectiveness_rating IS NOT NULL
                """)
                avg_effectiveness = cursor.fetchone()

                cursor.execute("""
                    SELECT outcome, COUNT(*)
                    FROM lesson_applications
                    GROUP BY outcome
                """)
                application_outcomes = dict(cursor.fetchall())

                return {
                    "total_lessons": total_lessons,
                    "lessons_by_category": lessons_by_category,
                    "average_feedback_rating": avg_feedback[0] if avg_feedback[0] else 0,
                    "total_feedback_count": avg_feedback[1] if avg_feedback[1] else 0,
                    "average_effectiveness": avg_effectiveness[0] if avg_effectiveness[0] else 0,
                    "total_applications": avg_effectiveness[1] if avg_effectiveness[1] else 0,
                    "application_outcomes": application_outcomes,
                    "most_effective_categories": self._get_top_categories()
                }

        except Exception as e:
            logger.error(f"Failed to get lesson statistics: {e}")
            return {}

    def _get_top_categories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing categories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT l.category,
                           AVG(la.effectiveness_rating) as avg_effectiveness,
                           COUNT(la.id) as application_count,
                           AVG(l.confidence) as avg_confidence
                    FROM lessons l
                    JOIN lesson_applications la ON l.id = la.lesson_id
                    WHERE la.effectiveness_rating IS NOT NULL
                    GROUP BY l.category
                    HAVING application_count >= 3
                    ORDER BY avg_effectiveness DESC, avg_confidence DESC
                    LIMIT ?
                """, (limit,))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        "category": row[0],
                        "avg_effectiveness": row[1],
                        "application_count": row[2],
                        "avg_confidence": row[3]
                    })

                return results

        except Exception as e:
            logger.error(f"Failed to get top categories: {e}")
            return []

    def is_healthy(self) -> bool:
        """Check if lesson system is healthy"""
        try:
            self._ensure_initialized()
            return self._initialized
        except Exception:
            return False