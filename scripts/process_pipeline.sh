#!/bin/bash
#
# Complete MT5 to RAG Processing Pipeline
# Runs all steps from CSV to ChromaDB
#
# Usage:
#   ./scripts/process_pipeline.sh data/XAUUSD_PERIOD_M15_0.csv
#   ./scripts/process_pipeline.sh data/XAUUSD_PERIOD_M15_200.csv --live
#

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
STRUCTURED_DIR="data/structured"
PATTERNS_DIR="data/patterns"
CHROMA_DIR="./chroma_db"
COLLECTION="trading_patterns"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   MT5 TO RAG PROCESSING PIPELINE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}❌ Error: No input file specified${NC}"
    echo ""
    echo "Usage:"
    echo "  $0 <input_csv_file> [--live]"
    echo ""
    echo "Examples:"
    echo "  $0 data/XAUUSD_PERIOD_M15_0.csv          # Process full history"
    echo "  $0 data/XAUUSD_PERIOD_M15_200.csv --live # Process live data"
    exit 1
fi

INPUT_FILE="$1"
IS_LIVE="${2:-}"

# Validate input file
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}❌ Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Extract metadata from filename
FILENAME=$(basename "$INPUT_FILE")
FILENAME_NO_EXT="${FILENAME%.csv}"

# Try to extract symbol and timeframe from filename
# Expected format: SYMBOL_PERIOD_TIMEFRAME_CANDLES.csv
if [[ $FILENAME =~ ^([A-Z_]+)_PERIOD_([A-Z0-9]+)_([0-9]+)\.csv$ ]]; then
    SYMBOL="${BASH_REMATCH[1]}"
    TIMEFRAME="${BASH_REMATCH[2]}"
    CANDLES="${BASH_REMATCH[3]}"
    echo -e "${GREEN}✓ Detected: Symbol=$SYMBOL, Timeframe=$TIMEFRAME, Candles=$CANDLES${NC}"
else
    # Fallback defaults
    SYMBOL="XAUUSD"
    TIMEFRAME="M15"
    CANDLES="unknown"
    echo -e "${YELLOW}⚠ Could not parse filename, using defaults: Symbol=$SYMBOL, Timeframe=$TIMEFRAME${NC}"
fi

# Create output directories
mkdir -p "$STRUCTURED_DIR"
mkdir -p "$PATTERNS_DIR"

# Define output files
STRUCTURED_FILE="$STRUCTURED_DIR/${SYMBOL}_${TIMEFRAME}_structured.json"
PATTERNS_FILE="$PATTERNS_DIR/${SYMBOL}_${TIMEFRAME}_patterns.json"

# Live data uses smaller lookforward
if [ "$IS_LIVE" == "--live" ]; then
    LOOKFORWARD=10
    echo -e "${YELLOW}📊 Live mode: Using lookforward=$LOOKFORWARD bars${NC}"
else
    LOOKFORWARD=20
    echo -e "${BLUE}📊 Full history mode: Using lookforward=$LOOKFORWARD bars${NC}"
fi

echo ""
echo -e "${BLUE}Pipeline Configuration:${NC}"
echo "  Input:      $INPUT_FILE"
echo "  Symbol:     $SYMBOL"
echo "  Timeframe:  $TIMEFRAME"
echo "  Structured: $STRUCTURED_FILE"
echo "  Patterns:   $PATTERNS_FILE"
echo "  ChromaDB:   $CHROMA_DIR"
echo "  Collection: $COLLECTION"
echo ""

# ==================== STEP 1: CSV → Structured JSON ====================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 1/4: Converting CSV to Structured JSON${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

python scripts/mt5_to_structured_json.py \
    --input "$INPUT_FILE" \
    --output "$STRUCTURED_FILE" \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 1 failed: CSV conversion${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Step 1 complete${NC}"
echo ""

# ==================== STEP 2: Detect Patterns ====================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 2/4: Detecting Trading Patterns${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

python scripts/pattern_detector.py \
    --input "$STRUCTURED_FILE" \
    --output "$PATTERNS_FILE" \
    --lookforward "$LOOKFORWARD" \
    --min-quality 0.6

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 2 failed: Pattern detection${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Step 2 complete${NC}"
echo ""

# ==================== STEP 3: Feed to ChromaDB ====================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 3/4: Feeding Patterns to ChromaDB${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

python scripts/rag_structured_feeder.py \
    --input "$PATTERNS_FILE" \
    --chroma-dir "$CHROMA_DIR" \
    --collection "$COLLECTION" \
    --batch-size 100

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Step 3 failed: ChromaDB feeding${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Step 3 complete${NC}"
echo ""

# ==================== STEP 4: Verify & Test ====================
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}STEP 4/4: Verification & Testing${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Show collection stats
echo -e "${BLUE}📊 Collection Statistics:${NC}"
python scripts/rag_structured_feeder.py \
    --chroma-dir "$CHROMA_DIR" \
    --collection "$COLLECTION" \
    --stats-only

echo ""

# Test retrieval
echo -e "${BLUE}🔍 Testing Pattern Retrieval:${NC}"
python scripts/pattern_retriever.py \
    --query "bullish reversal" \
    --symbol "$SYMBOL" \
    --timeframe "$TIMEFRAME" \
    --limit 3 \
    --format llm \
    --chroma-dir "$CHROMA_DIR" \
    --collection "$COLLECTION"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠ Retrieval test failed (non-fatal)${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ PIPELINE COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📁 Generated Files:${NC}"
echo "  ✓ Structured: $STRUCTURED_FILE"
echo "  ✓ Patterns:   $PATTERNS_FILE"
echo ""
echo -e "${BLUE}💾 ChromaDB:${NC}"
echo "  ✓ Directory:  $CHROMA_DIR"
echo "  ✓ Collection: $COLLECTION"
echo ""
echo -e "${BLUE}💡 Next Steps:${NC}"
echo "  • Query patterns: python scripts/pattern_retriever.py --query \"your query\""
echo "  • View stats: python scripts/rag_structured_feeder.py --stats-only"
echo "  • Process more files: $0 <another_csv_file>"
echo ""
