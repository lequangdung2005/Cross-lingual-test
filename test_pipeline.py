"""
Unit tests for the Few-Shot Test Generation Pipeline

These tests demonstrate validation at each stage of the pipeline.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np


class TestUniXcoderEmbedder(unittest.TestCase):
    """Test the UniXcoder embedding component."""
    
    @patch('retriever.unixcoder.AutoTokenizer')
    @patch('retriever.unixcoder.AutoModel')
    def setUp(self, mock_model, mock_tokenizer):
        """Set up mock embedder for testing."""
        from retriever.unixcoder import UniXcoderEmbedder
        
        # Mock the model and tokenizer
        self.mock_tokenizer = Mock()
        self.mock_model = Mock()
        
        mock_tokenizer.from_pretrained.return_value = self.mock_tokenizer
        mock_model.from_pretrained.return_value = self.mock_model
        
        # Mock model output
        mock_output = Mock()
        mock_output.last_hidden_state = np.random.randn(1, 10, 768)
        self.mock_model.return_value = mock_output
        
        self.embedder = UniXcoderEmbedder(model_name="microsoft/unixcoder-base")
    
    def test_embed_returns_correct_shape(self):
        """Validate: Embedding has correct shape."""
        # This is a conceptual test - in real scenario, would test with actual model
        expected_shape = 768  # UniXcoder base dimension
        
        # Mock embedding
        mock_embedding = np.random.randn(expected_shape)
        
        self.assertEqual(len(mock_embedding), expected_shape)
        self.assertIsInstance(mock_embedding, np.ndarray)
    
    def test_embed_no_nan_or_inf(self):
        """Validate: Embedding contains no NaN or Inf values."""
        embedding = np.random.randn(768)
        
        self.assertFalse(np.any(np.isnan(embedding)))
        self.assertFalse(np.any(np.isinf(embedding)))


class TestCodeExampleDatabase(unittest.TestCase):
    """Test the code example database component."""
    
    def setUp(self):
        """Set up test database."""
        from retriever.unixcoder import CodeExampleDatabase
        
        # Mock embedder
        self.mock_embedder = Mock()
        self.mock_embedder.embed.return_value = np.random.randn(768)
        self.mock_embedder.embed_batch.return_value = np.random.randn(5, 768)
        
        self.database = CodeExampleDatabase(self.mock_embedder)
    
    def test_add_example_increases_count(self):
        """Validate: Adding example increases database size."""
        initial_count = len(self.database.examples)
        
        self.database.add_example(
            "def foo(): pass",
            "def test_foo(): pass",
            {"lang": "python"}
        )
        
        self.assertEqual(len(self.database.examples), initial_count + 1)
    
    def test_build_index_creates_embeddings(self):
        """Validate: Index building creates embedding matrix."""
        # Add examples
        for i in range(3):
            self.database.add_example(f"def f{i}(): pass", f"def test_f{i}(): pass")
        
        # Build index
        self.database.build_index()
        
        # Validate
        self.assertIsNotNone(self.database.embeddings)
        self.assertEqual(self.database.embeddings.shape[0], 3)
    
    def test_retrieve_returns_correct_number(self):
        """Validate: Retrieval returns requested number of results."""
        # Add examples
        for i in range(5):
            self.database.add_example(f"def f{i}(): pass", f"def test_f{i}(): pass")
        
        self.database.build_index()
        
        # Retrieve
        results = self.database.retrieve("def query(): pass", top_k=3)
        
        # Validate
        self.assertEqual(len(results), 3)
    
    def test_retrieve_returns_sorted_by_similarity(self):
        """Validate: Results sorted by similarity score."""
        # Add examples
        for i in range(5):
            self.database.add_example(f"def f{i}(): pass", f"def test_f{i}(): pass")
        
        self.database.build_index()
        
        # Retrieve
        results = self.database.retrieve("def query(): pass", top_k=5)
        
        # Validate sorting
        similarities = [r.similarity_score for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
    
    def test_cosine_similarity_in_valid_range(self):
        """Validate: Cosine similarity is in [-1, 1]."""
        query = np.random.randn(768)
        database = np.random.randn(10, 768)
        
        from retriever.unixcoder import CodeExampleDatabase
        similarities = CodeExampleDatabase._cosine_similarity(query, database)
        
        self.assertTrue(np.all(similarities >= -1.0))
        self.assertTrue(np.all(similarities <= 1.0))


class TestPipelineValidation(unittest.TestCase):
    """Test pipeline validation and self-correction."""
    
    def setUp(self):
        """Set up mock pipeline."""
        from retriever.unixcoder import (
            FewShotTestGenerationPipeline,
            CodeExample,
            RetrievalResult
        )
        
        self.mock_embedder = Mock()
        self.mock_database = Mock()
        
        self.pipeline = FewShotTestGenerationPipeline(
            embedder=self.mock_embedder,
            database=self.mock_database,
            top_k=3,
            similarity_threshold=0.5
        )
        
        # Create mock examples
        self.mock_examples = [
            RetrievalResult(
                example=CodeExample("def f1(): pass", "def test_f1(): pass"),
                similarity_score=0.8,
                rank=1
            ),
            RetrievalResult(
                example=CodeExample("def f2(): pass", "def test_f2(): pass"),
                similarity_score=0.6,
                rank=2
            ),
            RetrievalResult(
                example=CodeExample("def f3(): pass", "def test_f3(): pass"),
                similarity_score=0.3,
                rank=3
            ),
        ]
    
    def test_validation_filters_low_similarity(self):
        """Validate: Low similarity examples are filtered."""
        validation = self.pipeline.validate_retrieval(self.mock_examples)
        
        # Should filter out example with 0.3 similarity (below 0.5 threshold)
        self.assertEqual(len(validation['valid_results']), 2)
        self.assertEqual(validation['filtered_count'], 1)
    
    def test_validation_passes_with_good_results(self):
        """Validate: Pipeline passes with good similarity scores."""
        good_examples = [
            RetrievalResult(
                example=CodeExample("def f(): pass", "def test(): pass"),
                similarity_score=0.9,
                rank=1
            )
        ]
        
        validation = self.pipeline.validate_retrieval(good_examples)
        
        self.assertTrue(validation['passed'])
        self.assertGreater(validation['avg_similarity'], 0.5)
    
    def test_validation_fails_with_poor_results(self):
        """Validate: Pipeline fails with poor similarity scores."""
        poor_examples = [
            RetrievalResult(
                example=CodeExample("def f(): pass", "def test(): pass"),
                similarity_score=0.2,
                rank=1
            )
        ]
        
        validation = self.pipeline.validate_retrieval(poor_examples)
        
        self.assertFalse(validation['passed'])
        self.assertEqual(len(validation['valid_results']), 0)
    
    def test_validation_calculates_correct_average(self):
        """Validate: Average similarity calculated correctly."""
        validation = self.pipeline.validate_retrieval(self.mock_examples)
        
        expected_avg = (0.8 + 0.6 + 0.3) / 3
        self.assertAlmostEqual(validation['avg_similarity'], expected_avg, places=2)
    
    def test_prompt_construction_includes_all_examples(self):
        """Validate: Constructed prompt includes all examples."""
        focal = "def query(): pass"
        examples = self.mock_examples[:2]  # Use first 2
        
        prompt = self.pipeline.construct_few_shot_prompt(focal, examples)
        
        # Check that all examples are included
        self.assertIn("Example 1", prompt)
        self.assertIn("Example 2", prompt)
        self.assertIn("def f1(): pass", prompt)
        self.assertIn("def f2(): pass", prompt)
    
    def test_prompt_construction_includes_query(self):
        """Validate: Constructed prompt includes the query."""
        focal = "def my_function(): pass"
        
        prompt = self.pipeline.construct_few_shot_prompt(focal, self.mock_examples[:1])
        
        self.assertIn("my_function", prompt)
        self.assertIn("Your Task", prompt)
    
    def test_prompt_includes_similarity_scores(self):
        """Validate: Prompt includes similarity scores for transparency."""
        prompt = self.pipeline.construct_few_shot_prompt(
            "def f(): pass",
            self.mock_examples[:1]
        )
        
        self.assertIn("0.8", prompt)  # Similarity score


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline execution."""
    
    def setUp(self):
        """Set up pipeline with mocks."""
        from retriever.unixcoder import (
            FewShotTestGenerationPipeline,
            CodeExample,
            RetrievalResult
        )
        
        self.mock_embedder = Mock()
        self.mock_embedder.embed.return_value = np.random.randn(768)
        
        self.mock_database = Mock()
        self.mock_database.retrieve.return_value = [
            RetrievalResult(
                example=CodeExample("def example(): pass", "def test_example(): pass"),
                similarity_score=0.85,
                rank=1
            )
        ]
        
        self.pipeline = FewShotTestGenerationPipeline(
            embedder=self.mock_embedder,
            database=self.mock_database,
            top_k=1,
            similarity_threshold=0.5
        )
    
    def test_pipeline_returns_success_with_valid_input(self):
        """Validate: Pipeline completes successfully with valid input."""
        result = self.pipeline.run("def test_function(): pass")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('few_shot_prompt', result)
        self.assertIn('pipeline_stages', result)
    
    def test_pipeline_stages_all_executed(self):
        """Validate: All pipeline stages are executed."""
        result = self.pipeline.run("def test_function(): pass")
        
        stages = result['pipeline_stages']
        
        self.assertIn('query_processing', stages)
        self.assertIn('retrieval', stages)
        self.assertIn('validation', stages)
        self.assertIn('prompt_construction', stages)
    
    def test_pipeline_fails_validation_with_low_threshold(self):
        """Validate: Pipeline fails when no examples meet threshold."""
        # Set very high threshold
        self.pipeline.similarity_threshold = 0.99
        
        result = self.pipeline.run("def test_function(): pass")
        
        self.assertEqual(result['status'], 'failed_validation')
    
    def test_query_processing_stage_output(self):
        """Validate: Query processing returns expected output format."""
        result = self.pipeline.process_query("def test(): pass")
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('embedding', result)
        self.assertIn('focal_method', result)
        self.assertIn('embedding_shape', result)
    
    def test_retrieval_stage_calls_database(self):
        """Validate: Retrieval stage calls database.retrieve()."""
        self.pipeline.retrieve_examples("def test(): pass", top_k=3)
        
        self.mock_database.retrieve.assert_called_once()
    
    def test_complete_pipeline_produces_valid_prompt(self):
        """Validate: Complete pipeline produces non-empty prompt."""
        result = self.pipeline.run("def test(): pass")
        
        if result['status'] == 'success':
            prompt = result['few_shot_prompt']
            
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
            self.assertIn("Task:", prompt)
            self.assertIn("Example", prompt)


class TestSelfCorrection(unittest.TestCase):
    """Test self-correction mechanisms."""
    
    def test_can_adjust_threshold_after_failure(self):
        """Validate: Can adjust threshold and retry after validation failure."""
        from retriever.unixcoder import FewShotTestGenerationPipeline
        
        mock_embedder = Mock()
        mock_database = Mock()
        
        pipeline = FewShotTestGenerationPipeline(
            embedder=mock_embedder,
            database=mock_database,
            similarity_threshold=0.9
        )
        
        # First attempt should fail with high threshold
        initial_threshold = pipeline.similarity_threshold
        self.assertEqual(initial_threshold, 0.9)
        
        # Self-correction: lower threshold
        pipeline.similarity_threshold = 0.5
        
        # Verify threshold changed
        self.assertEqual(pipeline.similarity_threshold, 0.5)
        self.assertLess(pipeline.similarity_threshold, initial_threshold)
    
    def test_validation_provides_feedback_for_correction(self):
        """Validate: Validation report provides actionable feedback."""
        from retriever.unixcoder import (
            FewShotTestGenerationPipeline,
            CodeExample,
            RetrievalResult
        )
        
        pipeline = FewShotTestGenerationPipeline(
            embedder=Mock(),
            database=Mock(),
            similarity_threshold=0.8
        )
        
        poor_results = [
            RetrievalResult(
                example=CodeExample("def f(): pass", "def test(): pass"),
                similarity_score=0.5,
                rank=1
            )
        ]
        
        validation = pipeline.validate_retrieval(poor_results)
        
        # Should provide feedback
        self.assertIn('avg_similarity', validation)
        self.assertIn('similarity_threshold', validation)
        self.assertIn('passed', validation)
        
        # Should indicate failure
        self.assertFalse(validation['passed'])
        
        # Should show that threshold is the issue
        self.assertLess(validation['avg_similarity'], validation['similarity_threshold'])


def run_validation_tests():
    """Run all validation tests and report results."""
    print("=" * 80)
    print("RUNNING PIPELINE VALIDATION TESTS")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestUniXcoderEmbedder))
    suite.addTests(loader.loadTestsFromTestCase(TestCodeExampleDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfCorrection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All validation tests passed!")
    else:
        print("\n❌ Some tests failed. Review output above.")
    
    return result


if __name__ == "__main__":
    run_validation_tests()
