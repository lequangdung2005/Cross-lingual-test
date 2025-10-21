"""
Data loader module for loading focal method and unit test pairs.

This module provides utilities to load code examples from various sources:
- Local JSON/JSONL files
- Local CSV files
- HuggingFace datasets
- Directory of code files
- Custom formats

The loaded data is used to populate the CodeExampleDatabase for retrieval.
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load focal method and unit test pairs from various sources.
    
    Supported formats:
    - JSON/JSONL files
    - CSV files
    - HuggingFace datasets
    - Directory structures
    - Python dictionaries
    """
    
    @staticmethod
    def load_from_json(
        file_path: str,
        focal_key: str = "focal_method",
        test_key: str = "unit_test",
        metadata_keys: Optional[List[str]] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load examples from a JSON file.
        
        Expected JSON format:
        [
            {
                "focal_method": "def add(a, b): ...",
                "unit_test": "def test_add(): ...",
                "language": "python",
                ...
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            focal_key: Key name for focal method in JSON
            test_key: Key name for unit test in JSON
            metadata_keys: List of additional keys to include as metadata
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        logger.info(f"Loading data from JSON: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of examples")
            
            examples = []
            for idx, item in enumerate(data):
                if focal_key not in item:
                    logger.warning(f"Item {idx} missing '{focal_key}', skipping")
                    continue
                if test_key not in item:
                    logger.warning(f"Item {idx} missing '{test_key}', skipping")
                    continue
                
                focal = item[focal_key]
                test = item[test_key]
                
                # Collect metadata
                metadata = {}
                if metadata_keys:
                    for key in metadata_keys:
                        if key in item:
                            metadata[key] = item[key]
                else:
                    # Include all other keys as metadata
                    metadata = {k: v for k, v in item.items() 
                               if k not in [focal_key, test_key]}
                
                examples.append((focal, test, metadata))
            
            logger.info(f"Loaded {len(examples)} examples from JSON")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise
    
    @staticmethod
    def load_from_jsonl(
        file_path: str,
        focal_key: str = "focal_method",
        test_key: str = "unit_test",
        metadata_keys: Optional[List[str]] = None,
        max_examples: Optional[int] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load examples from a JSONL (JSON Lines) file.
        
        Each line should be a valid JSON object.
        
        Args:
            file_path: Path to JSONL file
            focal_key: Key name for focal method
            test_key: Key name for unit test
            metadata_keys: List of additional keys to include as metadata
            max_examples: Maximum number of examples to load
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        logger.info(f"Loading data from JSONL: {file_path}")
        
        try:
            examples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_examples and len(examples) >= max_examples:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Line {line_num}: Invalid JSON, skipping")
                        continue
                    
                    if focal_key not in item or test_key not in item:
                        logger.warning(f"Line {line_num}: Missing required keys, skipping")
                        continue
                    
                    focal = item[focal_key]
                    test = item[test_key]
                    
                    # Collect metadata
                    metadata = {}
                    if metadata_keys:
                        for key in metadata_keys:
                            if key in item:
                                metadata[key] = item[key]
                    else:
                        metadata = {k: v for k, v in item.items() 
                                   if k not in [focal_key, test_key]}
                    
                    examples.append((focal, test, metadata))
            
            logger.info(f"Loaded {len(examples)} examples from JSONL")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load JSONL file: {e}")
            raise
    
    @staticmethod
    def load_from_csv(
        file_path: str,
        focal_column: str = "focal_method",
        test_column: str = "unit_test",
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ",",
        skip_header: bool = True
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load examples from a CSV file.
        
        Args:
            file_path: Path to CSV file
            focal_column: Column name for focal method
            test_column: Column name for unit test
            metadata_columns: List of columns to include as metadata
            delimiter: CSV delimiter
            skip_header: Whether to skip the first row
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        logger.info(f"Loading data from CSV: {file_path}")
        
        try:
            examples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row_num, row in enumerate(reader, 1):
                    if focal_column not in row or test_column not in row:
                        logger.warning(f"Row {row_num}: Missing required columns, skipping")
                        continue
                    
                    focal = row[focal_column]
                    test = row[test_column]
                    
                    # Collect metadata
                    metadata = {}
                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row:
                                metadata[col] = row[col]
                    else:
                        metadata = {k: v for k, v in row.items() 
                                   if k not in [focal_column, test_column]}
                    
                    examples.append((focal, test, metadata))
            
            logger.info(f"Loaded {len(examples)} examples from CSV")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    @staticmethod
    def load_from_huggingface(
        dataset_name: str,
        split: str = "train",
        focal_key: str = "focal_method",
        test_key: str = "unit_test",
        metadata_keys: Optional[List[str]] = None,
        max_examples: Optional[int] = None,
        cache_dir: Optional[str] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load examples from a HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "user/dataset")
            split: Dataset split to load ("train", "test", "validation")
            focal_key: Key name for focal method
            test_key: Key name for unit test
            metadata_keys: List of additional keys to include as metadata
            max_examples: Maximum number of examples to load
            cache_dir: Directory to cache the dataset
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        logger.info(f"Loading data from HuggingFace: {dataset_name} (split={split})")
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required to load HuggingFace datasets. "
                "Install with: pip install datasets"
            )
        
        try:
            # Load dataset
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir
            )
            
            # Limit examples if specified
            if max_examples:
                dataset = dataset.select(range(min(max_examples, len(dataset))))
            
            examples = []
            for idx, item in enumerate(dataset):
                if focal_key not in item or test_key not in item:
                    logger.warning(f"Item {idx}: Missing required keys, skipping")
                    continue
                
                focal = item[focal_key]
                test = item[test_key]
                
                # Collect metadata
                metadata = {}
                if metadata_keys:
                    for key in metadata_keys:
                        if key in item:
                            metadata[key] = item[key]
                else:
                    metadata = {k: v for k, v in item.items() 
                               if k not in [focal_key, test_key]}
                
                examples.append((focal, test, metadata))
            
            logger.info(f"Loaded {len(examples)} examples from HuggingFace")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise
    
    @staticmethod
    def load_from_directory(
        directory: str,
        focal_pattern: str = "*_focal.py",
        test_pattern: str = "*_test.py",
        max_examples: Optional[int] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load examples from a directory structure.
        
        Expected structure:
        directory/
            example1_focal.py
            example1_test.py
            example2_focal.py
            example2_test.py
            ...
        
        Args:
            directory: Path to directory
            focal_pattern: Glob pattern for focal files
            test_pattern: Glob pattern for test files
            max_examples: Maximum number of examples to load
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        logger.info(f"Loading data from directory: {directory}")
        
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            # Find all focal files
            focal_files = sorted(dir_path.glob(focal_pattern))
            
            examples = []
            for focal_file in focal_files:
                if max_examples and len(examples) >= max_examples:
                    break
                
                # Derive test file name
                base_name = focal_file.stem.replace('_focal', '')
                test_file = focal_file.parent / f"{base_name}_test{focal_file.suffix}"
                
                if not test_file.exists():
                    logger.warning(f"Test file not found for {focal_file.name}, skipping")
                    continue
                
                # Read files
                with open(focal_file, 'r', encoding='utf-8') as f:
                    focal_code = f.read()
                
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_code = f.read()
                
                metadata = {
                    'source': 'directory',
                    'focal_file': str(focal_file),
                    'test_file': str(test_file),
                    'base_name': base_name
                }
                
                examples.append((focal_code, test_code, metadata))
            
            logger.info(f"Loaded {len(examples)} examples from directory")
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load from directory: {e}")
            raise
    
    @staticmethod
    def load_from_dict_list(
        data: List[Dict[str, Any]],
        focal_key: str = "focal_method",
        test_key: str = "unit_test",
        metadata_keys: Optional[List[str]] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load examples from a list of dictionaries.
        
        Args:
            data: List of dictionaries containing focal and test code
            focal_key: Key name for focal method
            test_key: Key name for unit test
            metadata_keys: List of additional keys to include as metadata
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        logger.info(f"Loading data from dictionary list ({len(data)} items)")
        
        examples = []
        for idx, item in enumerate(data):
            if focal_key not in item or test_key not in item:
                logger.warning(f"Item {idx}: Missing required keys, skipping")
                continue
            
            focal = item[focal_key]
            test = item[test_key]
            
            # Collect metadata
            metadata = {}
            if metadata_keys:
                for key in metadata_keys:
                    if key in item:
                        metadata[key] = item[key]
            else:
                metadata = {k: v for k, v in item.items() 
                           if k not in [focal_key, test_key]}
            
            examples.append((focal, test, metadata))
        
        logger.info(f"Loaded {len(examples)} examples from dictionary list")
        return examples
    
    @staticmethod
    def auto_load(
        source: Union[str, List[Dict]],
        **kwargs
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Automatically detect source type and load data.
        
        Args:
            source: Path to file/directory, HuggingFace dataset name, or list of dicts
            **kwargs: Additional arguments passed to specific loader
            
        Returns:
            List of (focal_method, unit_test, metadata) tuples
        """
        # If it's a list of dicts
        if isinstance(source, list):
            return DataLoader.load_from_dict_list(source, **kwargs)
        
        # If it's a string path or dataset name
        if isinstance(source, str):
            # Check if it's a local path
            path = Path(source)
            
            if path.exists():
                # It's a local path
                if path.is_file():
                    # Determine file type
                    if path.suffix == '.json':
                        return DataLoader.load_from_json(source, **kwargs)
                    elif path.suffix == '.jsonl':
                        return DataLoader.load_from_jsonl(source, **kwargs)
                    elif path.suffix == '.csv':
                        return DataLoader.load_from_csv(source, **kwargs)
                    else:
                        raise ValueError(f"Unsupported file type: {path.suffix}")
                elif path.is_dir():
                    return DataLoader.load_from_directory(source, **kwargs)
            else:
                # Assume it's a HuggingFace dataset
                logger.info(f"Path not found locally, trying HuggingFace: {source}")
                return DataLoader.load_from_huggingface(source, **kwargs)
        
        raise ValueError(f"Unsupported source type: {type(source)}")


def create_sample_data(output_path: str, num_examples: int = 10, format: str = "json"):
    """
    Create sample data file for testing.
    
    Args:
        output_path: Path to save the sample data
        num_examples: Number of examples to generate
        format: Output format ("json", "jsonl", or "csv")
    """
    examples = [
        {
            "focal_method": f"""def function_{i}(x, y):
    \"\"\"Example function {i}.\"\"\"
    return x + y + {i}""",
            "unit_test": f"""def test_function_{i}():
    assert function_{i}(1, 2) == {3 + i}
    assert function_{i}(0, 0) == {i}""",
            "language": "python",
            "category": "arithmetic",
            "example_id": i
        }
        for i in range(num_examples)
    ]
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)
    
    elif format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
    
    elif format == "csv":
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=examples[0].keys())
            writer.writeheader()
            writer.writerows(examples)
    
    logger.info(f"Created sample data: {output_path} ({num_examples} examples)")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Data Loader Module - Example Usage")
    print("=" * 80)
    
    # Create sample data
    print("\n1. Creating sample data files...")
    create_sample_data("sample_data.json", num_examples=5, format="json")
    create_sample_data("sample_data.jsonl", num_examples=5, format="jsonl")
    create_sample_data("sample_data.csv", num_examples=5, format="csv")
    print("✓ Sample files created")
    
    # Load from JSON
    print("\n2. Loading from JSON...")
    examples_json = DataLoader.load_from_json("sample_data.json")
    print(f"✓ Loaded {len(examples_json)} examples from JSON")
    
    # Load from JSONL
    print("\n3. Loading from JSONL...")
    examples_jsonl = DataLoader.load_from_jsonl("sample_data.jsonl", max_examples=3)
    print(f"✓ Loaded {len(examples_jsonl)} examples from JSONL")
    
    # Load from CSV
    print("\n4. Loading from CSV...")
    examples_csv = DataLoader.load_from_csv("sample_data.csv")
    print(f"✓ Loaded {len(examples_csv)} examples from CSV")
    
    # Auto-load
    print("\n5. Auto-loading...")
    examples_auto = DataLoader.auto_load("sample_data.json")
    print(f"✓ Auto-loaded {len(examples_auto)} examples")
    
    print("\n" + "=" * 80)
    print("Example data structure:")
    print("=" * 80)
    if examples_json:
        focal, test, metadata = examples_json[0]
        print(f"\nFocal Method:\n{focal[:100]}...")
        print(f"\nUnit Test:\n{test[:100]}...")
        print(f"\nMetadata: {metadata}")
    
    print("\n✓ All examples completed successfully!")
