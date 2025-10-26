"""
knowledge_feeder.py - API models and utilities for programmatic knowledge feeding
Provides Pydantic models and validation for knowledge ingestion APIs
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class KnowledgeCategory(str, Enum):
    """Categories for knowledge storage"""
    GENERAL = "general"
    TRADING = "trading"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    BUSINESS = "business"
    MARKET = "market"
    RISK = "risk_management"
    STRATEGY = "strategy"
    DEFINITIONS = "definitions"
    CORRECTIONS = "corrections"

class KnowledgeEntry(BaseModel):
    """Single knowledge entry for API"""
    topic: str = Field(..., description="Main topic or keyword")
    content: str = Field(..., description="Detailed content about the topic")
    category: KnowledgeCategory = Field(KnowledgeCategory.GENERAL, description="Knowledge category")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence level (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source: Optional[str] = Field(None, description="Source of this knowledge")
    priority: int = Field(1, ge=1, le=10, description="Priority level (1-10, higher = more important)")

class BulkKnowledgeRequest(BaseModel):
    """Request for bulk knowledge upload"""
    knowledge_entries: List[KnowledgeEntry] = Field(..., description="List of knowledge entries")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier for tracking")
    overwrite: bool = Field(False, description="Whether to overwrite existing knowledge")

class LessonEntry(BaseModel):
    """Lesson entry for structured learning"""
    title: str = Field(..., description="Lesson title")
    situation: str = Field(..., description="Situation or context")
    lesson: str = Field(..., description="What was learned")
    category: KnowledgeCategory = Field(KnowledgeCategory.TRADING, description="Lesson category")
    correct_approach: Optional[str] = Field(None, description="The correct approach or answer")
    wrong_approach: Optional[str] = Field(None, description="What was done incorrectly")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Confidence in this lesson")
    tags: List[str] = Field(default_factory=list, description="Tags for lesson categorization")

class CorrectionEntry(BaseModel):
    """Correction entry for fixing misconceptions"""
    incorrect_statement: str = Field(..., description="The incorrect statement or belief")
    correct_statement: str = Field(..., description="The correct information")
    topic: str = Field(..., description="Main topic being corrected")
    explanation: Optional[str] = Field(None, description="Why the correction is important")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this correction")
    category: KnowledgeCategory = Field(KnowledgeCategory.CORRECTIONS, description="Correction category")

class DefinitionEntry(BaseModel):
    """Definition entry for terms and concepts"""
    term: str = Field(..., description="Term or acronym")
    definition: str = Field(..., description="Clear definition")
    expanded_form: Optional[str] = Field(None, description="If acronym, the expanded form")
    context: Optional[str] = Field(None, description="Context where this definition applies")
    examples: List[str] = Field(default_factory=list, description="Examples of usage")
    category: KnowledgeCategory = Field(KnowledgeCategory.DEFINITIONS, description="Definition category")

class ApiResponse(BaseModel):
    """Standard API response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class KnowledgeStats(BaseModel):
    """Knowledge statistics response"""
    total_entries: int
    entries_by_category: Dict[str, int]
    last_updated: str
    total_lessons: int
    total_corrections: int
    total_definitions: int