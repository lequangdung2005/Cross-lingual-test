"""
Command-line interface module for Few-Shot Test Generation Pipeline.

This module handles argument parsing and command execution.
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional

from .database_builder import build_database
from .prompt_constructor import construct_single_prompt, construct_benchmark_prompt

logger = logging.getLogger(__name__)


def setup_parsers():
    """Set up argument parsers for all commands."""
    
    parser = argparse.ArgumentParser(
        description="Few-Shot Unit Test Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build database from JSON file
  %(prog)s build --input examples.json --output database.pkl
  
  # Build from HuggingFace dataset
  %(prog)s build --input username/dataset --format huggingface --output db.pkl
  
  # Construct prompt from query string
  %(prog)s construct --database database.pkl --query "def foo(): pass"
  
  # Construct prompt from query file
  %(prog)s construct --database database.pkl --query-file query.py
  
  # Construct prompts for benchmark
  %(prog)s benchmark --database database.pkl --benchmark-file benchmark.json
  
  # Save output to file
  %(prog)s construct --database database.pkl --query-file query.py --output prompt.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build database from data')
    build_parser.add_argument('--input', '-i', required=True, 
                             help='Input data path (file or HuggingFace dataset)')
    build_parser.add_argument('--output', '-o', required=True,
                             help='Output database path (.pkl)')
    build_parser.add_argument('--format', '-f', 
                             choices=['json', 'jsonl', 'csv', 'huggingface', 'auto'],
                             default='auto',
                             help='Input data format (default: auto-detect)')
    build_parser.add_argument('--model', '-m', default='microsoft/unixcoder-base',
                             help='UniXcoder model name')
    build_parser.add_argument('--focal-key', default='focal_method',
                             help='Key for focal method in data')
    build_parser.add_argument('--test-key', default='unit_test',
                             help='Key for unit test in data')
    build_parser.add_argument('--metadata-keys', nargs='+',
                             help='Keys to preserve as metadata')
    build_parser.add_argument('--max-examples', type=int,
                             help='Maximum number of examples to load')
    build_parser.add_argument('--batch-size', type=int, default=8,
                             help='Batch size for embedding generation')
    
    # Construct command (single prompt)
    construct_parser = subparsers.add_parser('construct', help='Construct prompt for single query')
    construct_parser.add_argument('--database', '-d', required=True,
                                 help='Path to database file (.pkl)')
    construct_parser.add_argument('--query', '-q',
                                 help='Query code as string')
    construct_parser.add_argument('--query-file', '-qf',
                                 help='Path to file containing query code')
    construct_parser.add_argument('--output', '-o',
                                 help='Output file path (optional)')
    construct_parser.add_argument('--top-k', type=int, default=3,
                                 help='Number of examples to retrieve')
    construct_parser.add_argument('--threshold', type=float, default=0.4,
                                 help='Similarity threshold')
    construct_parser.add_argument('--no-details', action='store_true',
                                 help='Hide detailed statistics')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Construct prompts for benchmark')
    benchmark_parser.add_argument('--database', '-d', required=True,
                                 help='Path to database file (.pkl)')
    benchmark_parser.add_argument('--benchmark-file', '-b', required=True,
                                 help='Path to benchmark file (JSON/JSONL)')
    benchmark_parser.add_argument('--output', '-o', required=True,
                                 help='Output file path')
    benchmark_parser.add_argument('--top-k', type=int, default=3,
                                 help='Number of examples to retrieve')
    benchmark_parser.add_argument('--threshold', type=float, default=0.4,
                                 help='Similarity threshold')
    benchmark_parser.add_argument('--format', '-f',
                                 choices=['standard', 'jsonl', 'csv'],
                                 default='standard',
                                 help='Output format (default: standard JSON)')
    benchmark_parser.add_argument('--no-details', action='store_true',
                                 help='Hide detailed statistics')
    
    return parser


def load_benchmark_file(benchmark_path: str) -> Optional[list]:
    """Load benchmark queries from file."""
    
    path = Path(benchmark_path)
    
    if not path.exists():
        logger.error(f"Benchmark file not found: {benchmark_path}")
        return None
    
    try:
        # Try JSON first
        if path.suffix == '.json':
            with open(benchmark_path, 'r') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'queries' in data:
                return data['queries']
            elif isinstance(data, dict) and 'benchmark' in data:
                return data['benchmark']
            else:
                logger.error("Invalid JSON structure. Expected list or dict with 'queries'/'benchmark' key")
                return None
        
        # Try JSONL
        elif path.suffix == '.jsonl':
            queries = []
            with open(benchmark_path, 'r') as f:
                for line in f:
                    if line.strip():
                        queries.append(json.loads(line))
            return queries
        
        else:
            logger.error(f"Unsupported benchmark file format: {path.suffix}")
            logger.info("Supported formats: .json, .jsonl")
            return None
            
    except Exception as e:
        logger.error(f"Failed to load benchmark file: {e}")
        import traceback
        traceback.print_exc()
        return None


def execute_command(args):
    """Execute the specified command."""
    
    if args.command == 'build':
        success = build_database(
            input_path=args.input,
            output_path=args.output,
            data_format=args.format,
            model_name=args.model,
            focal_key=args.focal_key,
            test_key=args.test_key,
            metadata_keys=args.metadata_keys,
            max_examples=args.max_examples,
            batch_size=args.batch_size
        )
        return 0 if success else 1
        
    elif args.command == 'construct':
        if not args.query and not args.query_file:
            logger.error("Either --query or --query-file must be provided")
            return 1
        
        result = construct_single_prompt(
            database_path=args.database,
            query_code=args.query,
            query_file=args.query_file,
            output_file=args.output,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            show_details=not args.no_details
        )
        return 0 if result else 1
    
    elif args.command == 'benchmark':
        # Load benchmark queries
        benchmark_queries = load_benchmark_file(args.benchmark_file)
        
        if not benchmark_queries:
            return 1
        
        logger.info(f"Loaded {len(benchmark_queries)} queries from benchmark file")
        
        # Construct prompts
        result = construct_benchmark_prompt(
            database_path=args.database,
            benchmark_queries=benchmark_queries,
            output_file=args.output,
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            show_details=not args.no_details,
            benchmark_format=args.format
        )
        return 0 if result else 1
    
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


def main():
    """Main entry point."""
    
    parser = setup_parsers()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    exit_code = execute_command(args)
    sys.exit(exit_code)
