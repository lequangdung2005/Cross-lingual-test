"""
Database builder module for Few-Shot Test Generation Pipeline.

This module handles building and saving indexed databases from training data.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_database(
    input_path: str,
    output_path: str,
    data_format: Optional[str] = None,
    model_name: str = "microsoft/unixcoder-base",
    focal_key: str = "focal_method",
    test_key: str = "unit_test",
    metadata_keys: Optional[list] = None,
    max_examples: Optional[int] = None,
    batch_size: int = 8
) -> bool:
    """
    Build and save a database from input data.
    
    Args:
        input_path: Path to input data file or directory
        output_path: Path to save the database (.pkl)
        data_format: Data format (json, jsonl, csv, huggingface, auto)
        model_name: UniXcoder model name
        focal_key: Key for focal method in data
        test_key: Key for unit test in data
        metadata_keys: Keys to preserve as metadata
        max_examples: Maximum number of examples to load
        batch_size: Batch size for embedding generation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from dense import create_pipeline
        from retrievers.database_data_loader import DataLoader
        
        logger.info("=" * 80)
        logger.info("BUILDING DATABASE")
        logger.info("=" * 80)
        
        # Detect format if not specified
        if data_format is None or data_format == "auto":
            data_format = "auto"
            logger.info(f"Auto-detecting format for: {input_path}")
        
        # Load data
        logger.info(f"Loading data from: {input_path}")
        logger.info(f"Format: {data_format}")
        
        if metadata_keys is None:
            metadata_keys = ["language", "category", "source"]
        
        if data_format == "json":
            examples = DataLoader.load_from_json(
                input_path, focal_key, test_key, metadata_keys
            )
        elif data_format == "jsonl":
            examples = DataLoader.load_from_jsonl(
                input_path, focal_key, test_key, metadata_keys, max_examples
            )
        elif data_format == "csv":
            examples = DataLoader.load_from_csv(
                input_path, focal_key, test_key, metadata_keys
            )
        elif data_format == "huggingface":
            examples = DataLoader.load_from_huggingface(
                input_path, split="train", max_examples=max_examples,
                focal_key=focal_key, test_key=test_key, metadata_keys=metadata_keys
            )
        elif data_format == "auto":
            examples = DataLoader.auto_load(
                input_path, focal_key, test_key, metadata_keys
            )
        else:
            logger.error(f"Unsupported format: {data_format}")
            return False
        
        logger.info(f"✓ Loaded {len(examples)} examples")
        
        if not examples:
            logger.error("No examples loaded. Cannot build database.")
            return False
        
        # Create pipeline
        logger.info(f"Initializing pipeline with model: {model_name}")
        logger.info("(This may take a moment to download the model...)")
        
        pipeline, database = create_pipeline(model_name=model_name)
        logger.info("✓ Pipeline initialized")
        
        # Add examples and build index
        logger.info("Adding examples to database...")
        database.add_examples_bulk(examples)
        
        logger.info(f"Building index (batch_size={batch_size})...")
        database.build_index(batch_size=batch_size)
        logger.info(f"✓ Database ready with {len(database.examples)} examples")
        
        # Save database
        logger.info(f"Saving database to: {output_path}")
        database.save_index(output_path)
        logger.info("✓ Database saved successfully")
        
        logger.info("=" * 80)
        logger.info("BUILD COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nDatabase info:")
        logger.info(f"  - Path: {output_path}")
        logger.info(f"  - Examples: {len(database.examples)}")
        logger.info(f"  - Model: {model_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build database: {e}")
        import traceback
        traceback.print_exc()
        return False
