"""
rag_enhancer.py - Memory-Augmented RAG System
Enhances the chat system with automatic lesson extraction and improved RAG capabilities
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class RAGEnhancer:
    """
    Enhanced RAG system that automatically detects corrections and stores lessons
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

        # Correction indicators
        self.correction_patterns = [
            r'\bno[,\s]+you\'?re?\s+wrong\b',
            r'\bthat\'?s?\s+not\s+(?:correct|right)\b',
            r'\bincorrect\b',
            r'\byou\'?re?\s+mistaken\b',
            r'\bactually\s+(?:it\s+is|it\'s?)\s+\w+',
            r'\bthe\s+correct\s+(?:answer|term|definition)\s+is\b',
            r'\blet\s+me\s+correct\s+you\b',
            r'\bthat\'?s?\s+wrong\b',
            r'\byou\s+got\s+it\s+wrong\b'
        ]

        # Definition patterns
        self.definition_patterns = [
            r'\b(\w+(?:\s+\w+)*)\s+(?:is|stands?\s+for)\s+([^.,!?]+)',
            r'\b(\w+(?:\s+\w+)*)\s+=\s+([^.,!?]+)',
            r'\b(\w+(?:\s+\w+)*)\s+means?\s+([^.,!?]+)',
            r'\b(\w+(?:\s+\w+)*)\s+is\s+(?:a|an|the)\s+([^.,!?]+)'
        ]

    def detect_correction(self, user_message: str, ai_response: str) -> bool:
        """
        Detect if user message contains a correction to the AI response

        Args:
            user_message: The user's input
            ai_response: The AI's previous response

        Returns:
            True if a correction is detected
        """
        user_lower = user_message.lower()

        # Check for correction indicators
        for pattern in self.correction_patterns:
            if re.search(pattern, user_lower):
                logger.info(f"Correction detected with pattern: {pattern}")
                return True

        # Check if user is providing the "correct" answer after AI was wrong
        if re.search(r'\bcorrect\b.*?\bis\b', user_lower):
            return True

        return False

    def extract_lesson_from_correction(self, user_message: str, ai_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract a lesson from a correction interaction

        Args:
            user_message: User's correction
            ai_response: AI's incorrect response

        Returns:
            Lesson dictionary or None if no lesson extracted
        """
        try:
            # Find what the user is correcting
            topic = self._extract_topic_from_interaction(user_message, ai_response)
            if not topic:
                return None

            # Extract the correct information from user message
            correct_info = self._extract_correct_information(user_message, topic)
            if not correct_info:
                return None

            # Create lesson title and content
            lesson_title = f"Correction: {topic}"
            lesson_content = f"Topic: {topic}\nCorrect Information: {correct_info}\nPrevious Incorrect Response: {ai_response[:200]}...\nUser Correction: {user_message}"

            return {
                'title': lesson_title,
                'content': lesson_content,
                'category': 'corrections',
                'confidence': 0.9,  # High confidence for user corrections
                'tags': [topic, 'correction', 'user_feedback'],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to extract lesson from correction: {e}")
            return None

    def _extract_topic_from_interaction(self, user_message: str, ai_response: str) -> Optional[str]:
        """
        Extract the main topic being discussed from the interaction

        Args:
            user_message: User's message
            ai_response: AI's response

        Returns:
            Topic string or None
        """
        # Try to find acronyms or terms being discussed
        patterns = [
            r'\b([A-Z]{2,})\b',  # Acronyms
            r'\bwhat\s+is\s+(\w+(?:\s+\w+)*)\b',  # "What is X" questions
            r'\b(\w+(?:\s+\w+)*)\s+(?:is|stands?\s+for)\b'  # Definitions
        ]

        # Check both user message and AI response
        combined_text = f"{user_message} {ai_response}"

        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                # Return the most common/frequent match
                return max(set(matches), key=matches.count)

        return None

    def _extract_correct_information(self, user_message: str, topic: str) -> Optional[str]:
        """
        Extract the correct information about a topic from user message

        Args:
            user_message: User's message
            topic: The topic being corrected

        Returns:
            Correct information string or None
        """
        # Look for definition patterns
        for pattern in self.definition_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match and match.group(1).lower() in topic.lower():
                return match.group(2).strip()

        # Look for explicit correction phrases
        correction_patterns = [
            rf'{re.escape(topic)}\s+(?:is|stands?\s+for)\s+([^.,!?]+)',
            rf'(?:the\s+correct\s+(?:answer|definition)|correct)\s+(?:is|for\s+{re.escape(topic)})\s+is\s+([^.,!?]+)',
            rf'actually\s+(?:it\s+is|it\'s?)\s+([^.,!?]+)',
        ]

        for pattern in correction_patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # If no specific pattern found, try to extract the key phrase
        # Look for sentences that contain the topic
        sentences = re.split(r'[.!?]', user_message)
        for sentence in sentences:
            if topic.lower() in sentence.lower():
                return sentence.strip()

        return None

    def store_correction_as_lesson(self, user_message: str, ai_response: str) -> bool:
        """
        Detect correction and store it as a lesson

        Args:
            user_message: User's message
            ai_response: AI's previous response

        Returns:
            True if lesson was stored successfully
        """
        try:
            if not self.detect_correction(user_message, ai_response):
                return False

            lesson = self.extract_lesson_from_correction(user_message, ai_response)
            if not lesson:
                return False

            # Store the lesson
            self.memory_manager.add_lesson_memory(
                lesson_title=lesson['title'],
                lesson_content=lesson['content'],
                category=lesson['category'],
                confidence=lesson['confidence'],
                tags=lesson['tags']
            )

            logger.info(f"Stored correction lesson: {lesson['title']}")
            return True

        except Exception as e:
            logger.error(f"Failed to store correction as lesson: {e}")
            return False

    def enhance_query_with_rag(self, query: str, max_context: int = 5) -> Dict[str, Any]:
        """
        Enhanced RAG retrieval with better query processing

        Args:
            query: User query
            max_context: Maximum context items to retrieve

        Returns:
            Enhanced context dictionary
        """
        try:
            # Get basic context (now includes enhanced memory retrieval)
            context = self.memory_manager.get_combined_context(query,
                                                              memory_context=max_context,
                                                              lesson_context=max_context)

            # Log what was found
            logger.info(f"Enhanced RAG context: {len(context['conversations'])} memories, {len(context['lessons'])} lessons")

            # If no lessons found but query might benefit from them, try broader search
            if not context['lessons']:
                # Try searching for related terms
                related_terms = self._extract_related_terms(query)
                for term in related_terms[:2]:  # Limit to avoid too many searches
                    term_context = self.memory_manager.get_combined_context(term,
                                                                         memory_context=1,
                                                                         lesson_context=2)
                    if term_context['lessons']:
                        context['lessons'].extend(term_context['lessons'])
                        break

            # Additional enhancement for personal information queries and cross-model continuity
            if self._is_personal_info_query(query):
                # Ensure we have some context even if semantic search fails
                if len(context['conversations']) == 0:
                    # Force retrieval of recent conversations
                    recent_memories = self.memory_manager.get_recent_memories(n=10)
                    for memory in recent_memories:
                        if memory.get('metadata', {}).get('type') == 'conversation':
                            formatted_memory = f"[{memory['metadata'].get('timestamp', 'Unknown')}] {memory['text']}"
                            context['conversations'].append(formatted_memory)

                logger.info(f"Personal info query detected - enhanced context with {len(context['conversations'])} memories")
            else:
                # For non-personal queries, also check if we need cross-model continuity
                # If very few conversations found, search more broadly for continuity
                if len(context['conversations']) < 2:
                    logger.info("Limited conversation context found, searching for cross-model continuity...")
                    # Force broader search to ensure continuity across model switches
                    personal_memories = self.memory_manager.search_personal_information()
                    context['conversations'].extend(personal_memories[:3])  # Add up to 3 personal memories
                    logger.info(f"Added {min(3, len(personal_memories))} continuity memories for cross-model context")

            return context

        except Exception as e:
            logger.error(f"Failed to enhance query with RAG: {e}")
            return {"conversations": [], "lessons": []}

    def _is_personal_info_query(self, query: str) -> bool:
        """
        Check if the query is asking for personal information

        Args:
            query: User query

        Returns:
            True if this is a personal information query
        """
        personal_patterns = [
            r'\bwhat(?:\'|s| is)?\s+my\s+name\b',
            r'\bwho\s+am\s+i\b',
            r'\bdo\s+you\s+remember\s+me\b',
            r'\bwhat(?:\'|s| is)?\s+my\s+(?:name|info|information)\b',
            r'\bremember\s+my\s+name\b',
            r'\babout\s+me\b',
            r'\bmy\s+details?\b'
        ]

        query_lower = query.lower()
        for pattern in personal_patterns:
            if re.search(pattern, query_lower):
                return True
        return False

    def _extract_related_terms(self, query: str) -> List[str]:
        """
        Extract related terms from query for broader search

        Args:
            query: Original query

        Returns:
            List of related terms
        """
        # Simple extraction of potential keywords
        # Look for acronyms, important terms, etc.
        words = re.findall(r'\b\w+\b', query)

        # Filter out common stop words
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'please', 'tell', 'me', 'again', 'short', 'so'}

        related_terms = []
        for word in words:
            if len(word) > 2 and word.lower() not in stop_words:
                related_terms.append(word)

        return related_terms[:5]  # Limit to top 5 terms