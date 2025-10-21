"""
Pipeline package for Few-Shot Test Generation.

This package provides a modular command-line interface for building databases
and constructing prompts for unit test generation.
"""

from .database_builder import build_database
from .prompt_constructor import construct_single_prompt, construct_benchmark_prompt
from .cli import main

__all__ = [
    'build_database',
    'construct_single_prompt',
    'construct_benchmark_prompt',
    'main'
]
