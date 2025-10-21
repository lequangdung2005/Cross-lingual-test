#!/usr/bin/env python3
"""
Few-Shot Test Generation Pipeline - Main Entry Point

Production-ready pipeline for generating unit tests using few-shot learning
with RAG (Retrieval-Augmented Generation) and UniXcoder embeddings.

This is the main entry point that uses the modular pipeline package.

Usage:
    # Build database from data
    python pipeline_cli.py build --input examples.json --output database.pkl

    # Construct prompt from query code
    python pipeline_cli.py construct --database database.pkl --query "def foo(): pass"
    
    # Construct prompt from query file
    python pipeline_cli.py construct --database database.pkl --query-file query.py
    
    # Construct prompts for benchmark
    python pipeline_cli.py benchmark --database database.pkl --benchmark-file benchmark.json --output results.json
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import and run the CLI
from pipeline.cli import main

if __name__ == "__main__":
    main()
