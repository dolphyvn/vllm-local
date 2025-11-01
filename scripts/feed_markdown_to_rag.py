#!/usr/bin/env python3
"""
Feed Markdown/Text Documents to RAG System
Chunks large documents and feeds them to the financial_memory collection

Usage:
    python scripts/feed_markdown_to_rag.py --file data/vwap.md --category trading
    python scripts/feed_markdown_to_rag.py --file docs/ebook.md --chunk-size 1000
"""

import os
import sys
import argparse
import requests
import re
from pathlib import Path
from typing import List, Dict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from knowledge_feeder import KnowledgeEntry, KnowledgeCategory
except ImportError:
    print("ERROR: knowledge_feeder module not found")
    sys.exit(1)


class MarkdownFeeder:
    """Feed markdown/text documents to the RAG system"""

    def __init__(self, base_url="http://localhost:8080", password="admin123"):
        self.base_url = base_url
        self.password = password
        self.token = None
        self.session = requests.Session()

    def authenticate(self):
        """Authenticate with the API"""
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"password": self.password}
            )

            if response.status_code == 200:
                self.token = response.json().get("session_token")
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                print("[OK] Authentication successful")
                return True
            else:
                print(f"[ERROR] Authentication failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"[ERROR] Authentication error: {e}")
            return False

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
            text: Full text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n\n')
                break_point = max(last_period, last_newline)

                if break_point > chunk_size // 2:  # Only break if it's reasonably close to end
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Extract sections from markdown based on headers

        Args:
            text: Markdown text

        Returns:
            List of sections with title and content
        """
        sections = []

        # Split by markdown headers (# Header, ## Header, etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')

        current_section = {"title": "Introduction", "content": "", "level": 0}

        for line in lines:
            match = re.match(header_pattern, line)

            if match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = {"title": title, "content": "", "level": level}
            else:
                current_section["content"] += line + "\n"

        # Add final section
        if current_section["content"].strip():
            sections.append(current_section)

        return sections

    def feed_document(self, file_path: str, category: str = "trading",
                     chunk_size: int = 1000, use_sections: bool = True) -> bool:
        """
        Feed a document to the RAG system

        Args:
            file_path: Path to markdown/text file
            category: Knowledge category
            chunk_size: Size of text chunks (if not using sections)
            use_sections: Use markdown sections instead of fixed chunks

        Returns:
            True if successful
        """
        print(f"\n[INFO] Reading file: {file_path}")

        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"[ERROR] Failed to read file: {e}")
            return False

        print(f"[INFO] File size: {len(content)} characters")

        # Get filename for tagging
        filename = Path(file_path).stem

        # Extract sections or chunks
        if use_sections:
            print("[INFO] Extracting markdown sections...")
            sections = self.extract_sections(content)
            print(f"[INFO] Found {len(sections)} sections")

            entries = []
            for i, section in enumerate(sections, 1):
                # If section is too large, chunk it
                if len(section["content"]) > chunk_size * 2:
                    chunks = self.chunk_text(section["content"], chunk_size)
                    for j, chunk in enumerate(chunks, 1):
                        entry = {
                            "topic": f"{section['title']} (Part {j}/{len(chunks)})",
                            "content": chunk,
                            "category": category,
                            "confidence": 0.9,
                            "tags": [filename, "ebook", section['title'].lower().replace(' ', '_')],
                            "source": file_path,
                            "priority": max(1, 7 - section["level"])  # Higher priority for main sections
                        }
                        entries.append(entry)
                else:
                    entry = {
                        "topic": section["title"],
                        "content": section["content"].strip(),
                        "category": category,
                        "confidence": 0.9,
                        "tags": [filename, "ebook", section['title'].lower().replace(' ', '_')],
                        "source": file_path,
                        "priority": max(1, 7 - section["level"])
                    }
                    entries.append(entry)

        else:
            print(f"[INFO] Chunking text into {chunk_size} character chunks...")
            chunks = self.chunk_text(content, chunk_size)
            print(f"[INFO] Created {len(chunks)} chunks")

            entries = []
            for i, chunk in enumerate(chunks, 1):
                # Extract first line as topic
                first_line = chunk.split('\n')[0][:100]
                entry = {
                    "topic": f"{filename} - Part {i}/{len(chunks)}",
                    "content": chunk,
                    "category": category,
                    "confidence": 0.8,
                    "tags": [filename, "ebook", "chunk"],
                    "source": file_path,
                    "priority": 5
                }
                entries.append(entry)

        # Feed in batches
        batch_size = 50
        total = len(entries)
        success_count = 0

        print(f"\n[INFO] Feeding {total} entries to RAG system...")

        for i in range(0, total, batch_size):
            batch = entries[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"[INFO] Processing batch {batch_num}/{total_batches} ({len(batch)} entries)...")

            try:
                response = self.session.post(
                    f"{self.base_url}/api/knowledge/bulk",
                    json={
                        "knowledge_entries": batch,
                        "batch_id": f"{filename}_batch_{batch_num}",
                        "overwrite": False
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        success_count += len(batch)
                        print(f"[OK] Batch {batch_num} added successfully")
                    else:
                        print(f"[ERROR] Batch {batch_num} failed: {result.get('message')}")
                else:
                    print(f"[ERROR] Batch {batch_num} HTTP {response.status_code}: {response.text[:200]}")

            except Exception as e:
                print(f"[ERROR] Failed to send batch {batch_num}: {e}")

        print(f"\n{'='*60}")
        print(f"[SUMMARY] Successfully added {success_count}/{total} entries")
        print(f"{'='*60}")

        return success_count == total


def main():
    parser = argparse.ArgumentParser(
        description='Feed markdown/text documents to RAG system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feed VWAP ebook
  python scripts/feed_markdown_to_rag.py --file data/vwap.md --category trading

  # Feed with custom chunk size
  python scripts/feed_markdown_to_rag.py --file docs/guide.md --chunk-size 1500

  # Use fixed chunks instead of sections
  python scripts/feed_markdown_to_rag.py --file data/book.txt --no-sections

  # Custom API endpoint
  python scripts/feed_markdown_to_rag.py --file data/vwap.md --url http://192.168.1.100:8080
        """
    )

    parser.add_argument('--file', required=True, help='Path to markdown/text file')
    parser.add_argument('--category', default='trading',
                       choices=['trading', 'technical', 'strategy', 'risk_management', 'general'],
                       help='Knowledge category (default: trading)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Character chunk size (default: 1000)')
    parser.add_argument('--no-sections', action='store_true',
                       help='Use fixed chunks instead of markdown sections')
    parser.add_argument('--url', default='http://localhost:8080',
                       help='API base URL (default: http://localhost:8080)')
    parser.add_argument('--password', default='admin123',
                       help='Admin password (default: admin123)')

    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.file):
        print(f"[ERROR] File not found: {args.file}")
        return 1

    # Create feeder
    feeder = MarkdownFeeder(base_url=args.url, password=args.password)

    # Authenticate
    if not feeder.authenticate():
        print("[ERROR] Failed to authenticate")
        return 1

    # Feed document
    success = feeder.feed_document(
        file_path=args.file,
        category=args.category,
        chunk_size=args.chunk_size,
        use_sections=not args.no_sections
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
