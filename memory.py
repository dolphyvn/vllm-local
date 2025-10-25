"""
memory.py - Enhanced ChromaDB-based persistent semantic memory system
Provides vector storage and retrieval for conversations, explicit memories, and lessons
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages persistent semantic memory using ChromaDB
    Handles storage and retrieval of conversations and explicit memories
    """

    def __init__(self, collection_name: str = "financial_memory", persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB memory manager (lazy initialization)

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB"""
        if self._initialized:
            return

        try:
            # Import ChromaDB only when needed
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Financial assistant memory storage"}
            )

            self._initialized = True
            logger.info(f"ChromaDB initialized with collection '{self.collection_name}' in '{self.persist_directory}'")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            # Continue without memory if ChromaDB fails
            self._initialized = False

    def add_memory(self, user_input: str, model_reply: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a conversation pair to memory

        Args:
            user_input: User's message
            model_reply: AI's response
            metadata: Additional metadata to store
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                logger.warning("Memory not available, skipping add_memory")
                return
            # Create unique ID
            timestamp = datetime.now().isoformat()
            memory_id = f"conv_{timestamp}_{hash(user_input) % 10000}"

            # Prepare metadata
            full_metadata = {
                "timestamp": timestamp,
                "type": "conversation",
                "user_input_length": len(user_input),
                "model_reply_length": len(model_reply)
            }

            if metadata:
                full_metadata.update(metadata)

            # Combine user input and model reply for embedding
            combined_text = f"User: {user_input}\nAssistant: {model_reply}"

            # Add to collection
            self.collection.add(
                documents=[combined_text],
                metadatas=[full_metadata],
                ids=[memory_id]
            )

            logger.debug(f"Added memory entry: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    def get_memory(self, query: str, n: int = 3, where: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Retrieve relevant memory entries based on query

        Args:
            query: Query text to search for
            n: Number of results to return
            where: Optional filter conditions

        Returns:
            List of relevant memory text entries
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                logger.warning("Memory not available, returning empty list")
                return []

            # Build query parameters
            query_params = {
                "query_texts": [query],
                "n_results": n
            }

            if where:
                query_params["where"] = where

            # Query the collection
            results = self.collection.query(**query_params)

            # Extract and format results
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}

                    # Format memory entry with timestamp if available
                    timestamp = metadata.get('timestamp', 'Unknown time')
                    memory_type = metadata.get('type', 'conversation')

                    formatted_memory = f"[{timestamp}] {memory_type}: {doc}"
                    memories.append(formatted_memory)

            logger.debug(f"Retrieved {len(memories)} memory entries for query: {query[:50]}...")
            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return []

    def search_memories_by_category(self, category: str, n: int = 10) -> List[str]:
        """
        Search memories by specific category

        Args:
            category: Category to filter by
            n: Maximum number of results

        Returns:
            List of memory entries from the specified category
        """
        try:
            results = self.collection.query(
                query_texts=[""],
                n_results=n,
                where={"category": category}
            )

            memories = []
            if results['documents'] and results['documents'][0]:
                memories = results['documents'][0]

            return memories

        except Exception as e:
            logger.error(f"Failed to search memories by category: {e}")
            return []

    def get_recent_memories(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent memory entries

        Args:
            n: Number of recent entries to retrieve

        Returns:
            List of memory entries with metadata
        """
        try:
            # Get all entries (ChromaDB doesn't have direct time-based sorting)
            results = self.collection.query(
                query_texts=["recent"],
                n_results=n * 2  # Get more to account for potential filtering
            )

            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    memories.append({
                        "text": doc,
                        "metadata": metadata
                    })

            # Sort by timestamp (most recent first) and limit
            memories.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            return memories[:n]

        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    def clear_old_memories(self, days_old: int = 30, keep_explicit: bool = True) -> int:
        """
        Clear old memories to manage storage

        Args:
            days_old: Age threshold in days
            keep_explicit: Whether to keep explicit memories

        Returns:
            Number of memories removed
        """
        try:
            # Calculate cutoff timestamp
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_timestamp = cutoff_date.isoformat()

            # This would require custom implementation as ChromaDB doesn't
            # have built-in date-based deletion
            # For now, this is a placeholder
            logger.info(f"Memory cleanup requested for entries older than {cutoff_timestamp}")
            return 0

        except Exception as e:
            logger.error(f"Failed to clear old memories: {e}")
            return 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()

            return {
                "collection_name": self.collection_name,
                "total_entries": count,
                "persist_directory": self.persist_directory
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def is_healthy(self) -> bool:
        """
        Check if the memory system is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                return False

            # Try a simple query to check connectivity
            self.collection.query(query_texts=["health_check"], n_results=1)
            return True
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False

    def backup_memory(self, backup_path: str) -> bool:
        """
        Create a backup of the memory collection

        Args:
            backup_path: Path to save backup

        Returns:
            True if successful, False otherwise
        """
        try:
            # This would require exporting all data from ChromaDB
            # For now, this is a placeholder
            logger.info(f"Memory backup requested to: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup memory: {e}")
            return False

    def search_conversations(self, keyword: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Search conversations by keyword

        Args:
            keyword: Keyword to search for
            n: Maximum number of results

        Returns:
            List of matching conversations with metadata
        """
        try:
            results = self.collection.query(
                query_texts=[keyword],
                n_results=n,
                where={"type": "conversation"}
            )

            conversations = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    conversations.append({
                        "text": doc,
                        "metadata": metadata
                    })

            return conversations

        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            return []

    def add_explicit_memory(self, key: str, value: str, category: str = "general"):
        """
        Add an explicit memory entry (key-value pair)

        Args:
            key: Memory key/identifier
            value: Memory content
            category: Category for organization (e.g., 'trading', 'strategy', 'risk')
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                logger.warning("Memory not available, skipping add_explicit_memory")
                return

            # Create unique ID
            timestamp = datetime.now().isoformat()
            memory_id = f"explicit_{timestamp}_{hash(key) % 10000}"

            # Prepare metadata
            metadata = {
                "timestamp": timestamp,
                "type": "explicit",
                "key": key,
                "category": category,
                "value_length": len(value)
            }

            # Add to collection
            self.collection.add(
                documents=[f"{key}: {value}"],
                metadatas=[metadata],
                ids=[memory_id]
            )

            logger.info(f"Added explicit memory: {key} (category: {category})")

        except Exception as e:
            logger.error(f"Failed to add explicit memory: {e}")
            raise

    def add_lesson_memory(self, lesson_title: str, lesson_content: str,
                        category: str = "trading", confidence: float = 0.7,
                        tags: List[str] = None, source_conversation: str = None):
        """
        Add a lesson to semantic memory for vector-based retrieval

        Args:
            lesson_title: Title of the lesson
            lesson_content: Detailed lesson content
            category: Lesson category
            confidence: Confidence level (0-1)
            tags: List of tags for categorization
            source_conversation: ID of conversation where lesson was learned
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                logger.warning("Memory not available, skipping add_lesson_memory")
                return

            # Create unique ID
            timestamp = datetime.now().isoformat()
            memory_id = f"lesson_{timestamp}_{hash(lesson_title) % 10000}"

            # Prepare metadata
            metadata = {
                "timestamp": timestamp,
                "type": "lesson",
                "title": lesson_title,
                "category": category,
                "confidence": confidence,
                "tags": tags or [],
                "source_conversation": source_conversation,
                "content_length": len(lesson_content)
            }

            # Add to ChromaDB for semantic search
            self.collection.add(
                documents=[f"Lesson: {lesson_title}. {lesson_content}"],
                metadatas=[metadata],
                ids=[memory_id]
            )

            logger.info(f"Added lesson to semantic memory: {lesson_title} (category: {category})")

        except Exception as e:
            logger.error(f"Failed to add lesson memory: {e}")
            raise

    def search_lessons(self, query: str, category: str = None, n: int = 5) -> List[Dict[str, Any]]:
        """
        Search for lessons using semantic similarity

        Args:
            query: Search query
            category: Filter by category (optional)
            n: Maximum number of results

        Returns:
            List of lesson documents with metadata
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                logger.warning("Memory not available, returning empty lesson list")
                return []

            # Build query parameters
            query_params = {
                "query_texts": [query],
                "n_results": n
            }

            # Add category filter if specified
            if category:
                query_params["where"] = {"category": category}

            # Query the collection
            results = self.collection.query(**query_params)

            # Extract and format results
            lessons = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    if metadata.get('type') == 'lesson':
                        lessons.append({
                            "document": doc,
                            "metadata": metadata,
                            "similarity_score": 1.0  # ChromaDB doesn't return scores by default
                        })

            logger.info(f"Found {len(lessons)} relevant lessons for query: {query[:50]}...")
            return lessons

        except Exception as e:
            logger.error(f"Failed to search lessons: {e}")
            return []

    def get_combined_context(self, query: str, memory_context: int = 3,
                           lesson_context: int = 2) -> Dict[str, List[str]]:
        """
        Get combined context from both conversation memory and lessons

        Args:
            query: The user's query
            memory_context: Number of conversation memories to retrieve
            lesson_context: Number of lessons to retrieve

        Returns:
            Dictionary with 'conversations' and 'lessons' keys
        """
        try:
            self._ensure_initialized()
            if not self._initialized or self.collection is None:
                logger.warning("Memory not available for combined context")
                return {"conversations": [], "lessons": []}

            # Get conversation memories
            conversation_memories = self.get_memory(query, n=memory_context)

            # Get lesson memories
            lesson_memories = self.search_lessons(query, n=lesson_context)

            # Format lesson memories as text
            lesson_texts = [lesson["document"] for lesson in lesson_memories]

            logger.info(f"Retrieved combined context: {len(conversation_memories)} conversations, {len(lesson_texts)} lessons")

            return {
                "conversations": conversation_memories,
                "lessons": lesson_texts
            }

        except Exception as e:
            logger.error(f"Failed to get combined context: {e}")
            return {"conversations": [], "lessons": []}