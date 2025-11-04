"""
Test Generation Module with OpenAI API

This module handles test generation using OpenAI API with multiple prompt strategies:

1. Standard Generation:
   - Loads from HuggingFace: Tessera2025/Tessera2025
   - Uses language-specific splits (rust, go, julia)
   - Builds prompts from templates

2. Few-Shot Generation (--use_fewshot_jsonl):
   - Loads JSONL files with retrieved examples
   - JSONL format: {"id": ..., "retrieved_context": {...}}
   - The 'retrieved_context' field contains focal_method and retrieved results
   - Combines retrieved examples + focal code to create full prompt

3. RAG Generation (--rag_type):
   - Uses pre-constructed prompts with RAG context
   - Supports BM25 and dense retrieval methods
"""

from typing import Any, List, Dict
from string import ascii_uppercase
import argparse
import os
import json
import textwrap
import time
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, Dataset
from utils import delete_test_from_filecontext, get_import_context, rust_parser, go_parser, julia_parser

from constant import prompts, suffixes, datapath, bm_25_rag_datapath, dense_rag_datapath, fewshot_prompt_dir, fewshot_datapath

lang_parsers = {
    "rust": rust_parser,
    "go": go_parser,
    "julia": julia_parser
}


class TestGenOpenAI:
    def __init__(
        self, 
        task_name, 
        dataset_path, 
        model_name,
        api_key=None,
        base_url=None,
        split="train", 
        instruct=False, 
        system_prompt=None, 
        template=None, 
        suffix=None, 
        arg_context=False, 
        is_rag=False, 
        file_context=False, 
        use_fewshot_jsonl=False
    ) -> None:
        """
        Initialize TestGenOpenAI.
        
        Args:
            task_name: Language task name (Rust, Go, Julia)
            dataset_path: Dataset path configuration
            model_name: OpenAI model name (e.g., gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional custom API base URL
            split: Dataset split
            instruct: Use instruction template
            system_prompt: System prompt for chat models
            template: Prompt template
            suffix: Code suffix template
            arg_context: Include argument context
            is_rag: Using RAG context
            file_context: Include file context
            use_fewshot_jsonl: Use few-shot JSONL with retrieved examples
        """
        self.TASK_NAME = task_name
        self.model_name = model_name
        self.instruct = instruct
        self.template = template
        self.suffix = suffix
        self.arg_context = arg_context
        self.is_rag = is_rag
        self.file_context = file_context
        self.use_fewshot_jsonl = use_fewshot_jsonl
        self.system_prompt = system_prompt
        
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        # Always load dataset from HuggingFace
        if dataset_path.get("split"):
            self.dataset = load_dataset(dataset_path["repo"], split=dataset_path["split"], trust_remote_code=True).to_list()
        else:
            self.dataset = load_dataset(dataset_path["repo"], split=split, trust_remote_code=True).to_list()

        print(f"Loaded dataset with {len(self.dataset)} examples")
        
        # Load few-shot examples if needed and merge with dataset
        if use_fewshot_jsonl and dataset_path.get("fewshot_file"):
            self.fewshot_data = self._load_fewshot_jsonl(dataset_path["fewshot_file"])
            for i, example in enumerate(self.dataset):
                example["retrieved_context"] = self.fewshot_data[i]["retrieved_context"]
                
            print(f"Loaded {len(self.fewshot_data)} few-shot examples to merge with dataset")
    
    def _load_fewshot_jsonl(self, file_path):
        """Load few-shot JSONL file and create a mapping by id."""
        print(f"Loading few-shot JSONL from: {file_path}")
        
        fewshots = []
        with open(file_path, 'r', encoding='utf-8') as f:
            fewshots = [json.loads(line.strip()) for line in f]
                
        return fewshots
    
    def _get_fewshot_examples(self, example):
        """Get few-shot examples from the example's retrieved_context field."""
        return example.get("retrieved_context")
    
    def _build_standard_prompt(self, example):
        """Build standard prompt from template for HuggingFace dataset."""
        # Prepare code with class signature if needed
        if "class_signature" in example and example["class_signature"]:
            code = "{} {{\n{}\n}}".format(
                example["class_signature"], 
                textwrap.indent(text=example["focal_code"], prefix='    ')
            )
        else:
            code = example["focal_code"]
        
        # Build base prompt
        if self.instruct:
            prompt = self.template.format(
                function_code=code,
                function_name=example["function_name"],
                file_path=example["file_path"]
            )
        else:
            prompt = self.template.format(
                function_code=code,
                function_name=example["function_name"]
            )
        
        return code, prompt
    
    def _build_fewshot_prompt(self, example, retrieved_context):
        """Build few-shot prompt from HuggingFace example + retrieved context."""
        focal_method = retrieved_context["focal_method"]
        retrieved_results = retrieved_context.get("results", [])
        
        # Format the retrieved examples as few-shot context
        few_shot_pairs = []
        for i, result in enumerate(retrieved_results[:5], 1):  # Use top 5 examples
            result_example = result["example"]
            pair = f"Example {i}:\n{result_example['focal_method']}\n\n{result_example['unit_test']}"
            few_shot_pairs.append(pair)
        
        if few_shot_pairs:
            few_shot_context = "Given the following related examples:\n\n" + "\n\n".join(few_shot_pairs) + "\n\n"
        else:
            few_shot_context = ""
        
        # Build base prompt with focal code
        if self.instruct:
            base_prompt = self.template.format(
                function_code=focal_method,
                function_name=example["function_name"],
                file_path=example["file_path"]
            )
        else:
            base_prompt = self.template.format(
                function_code=focal_method,
                function_name=example["function_name"]
            )
        
        # Combine few-shot context with base prompt
        prompt = few_shot_context + base_prompt
        
        return focal_method, prompt
    
    def _add_context_to_prompt(self, example, prompt):
        """Add various contexts to the prompt (for HuggingFace dataset only)."""
        # Add RAG context
        if self.is_rag and "context" in example:
            prompt = "#RELEVANT CONTEXT\n" + example["context"] + "\n#END_OF_CONTEXT\n\n" + prompt
        
        # Add file context
        elif self.file_context:
            if self.TASK_NAME.lower() == "rust":
                file_content = delete_test_from_filecontext(
                    lang_parsers[self.TASK_NAME.lower()], 
                    example["file_content"]
                )
            else:
                file_content = example["file_content"]
            prompt = "\n#CURRENT FILE: {}\n{}\n#ENDFILE\n".format(
                example["file_path"], 
                file_content
            ) + prompt
        
        # Add argument context
        if self.arg_context and example.get("function_component", {}).get("argument_definitions"):
            pre_prompt = "\n".join([
                y for x in example["function_component"]["argument_definitions"] 
                for y in x["definitions"]
            ]).strip()
            
            if example.get("struct_class"):
                pre_prompt = "\n\n" + example["struct_class"]
            
            prompt = "{}\n{}\n".format(
                get_import_context(self.TASK_NAME.lower(), example["file_content"]), 
                pre_prompt
            ) + "\n\n" + prompt
        
        return prompt
    
    def _add_suffix(self, prompt, function_name):
        """Add suffix to prompt if configured."""
        if self.suffix:
            return prompt + self.suffix.format(function_name=function_name)
        return prompt
        
    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        """Prepare dataset with prompts."""
        dataset = self.dataset
        
        # Process dataset directly without map (avoid serialization issues)
        processed_examples = []
        
        for i, example in enumerate(dataset):
            # Check if we should use few-shot prompts
            if self.use_fewshot_jsonl:
                retrieved_context = self._get_fewshot_examples(example)
                
                if retrieved_context:
                    # Build few-shot prompt with retrieved examples
                    code, prompt = self._build_fewshot_prompt(example, retrieved_context)
                    
                    example['question'] = self._add_suffix(prompt, example["function_name"])
                    example['task_id'] = i
                    processed_examples.append(example)
                    continue
            
            # Standard prompt construction (no few-shot or few-shot not available)
            code, prompt = self._build_standard_prompt(example)
            
            # Add contexts if needed
            if self.instruct:
                prompt = self._add_context_to_prompt(example, prompt)
            else:
                # For non-instruct models, add arg context if needed
                if self.arg_context and example.get("function_component", {}).get("argument_definitions"):
                    pre_prompt = "\n".join([
                        y for x in example["function_component"]["argument_definitions"] 
                        for y in x["definitions"]
                    ]).strip()
                    prompt = pre_prompt + "\n\n" + prompt
            
            example['question'] = self._add_suffix(prompt, example["function_name"])
            example['task_id'] = i
            processed_examples.append(example)
        
        return processed_examples
    
    def generate_single(
        self,
        prompt: str,
        num_return_sequences: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        retry_count: int = 3,
        retry_delay: float = 2.0
    ) -> List[str]:
        """
        Generate test code for a single prompt using OpenAI API.
        
        Args:
            prompt: Input prompt
            num_return_sequences: Number of sequences to generate
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            List of generated test code strings
        """
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        results = []
        
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences
                )
                
                # Extract generated texts
                for choice in response.choices:
                    results.append(choice.message.content)
                
                return results
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{retry_count}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)
                else:
                    # Return empty results if all retries failed
                    print(f"All retries failed. Returning empty results.")
                    return ["" for _ in range(num_return_sequences)]
        
        return results
    
    def generate(
        self,
        save_dir: str,
        num_return_sequences: int = 5,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        batch_size: int = 1,  # OpenAI API processes one at a time
        request_delay: float = 0.5  # Delay between API calls to avoid rate limits
    ):
        """
        Generate test code for all examples in the dataset.
        
        Args:
            save_dir: Directory to save generated tests
            num_return_sequences: Number of sequences to generate per example
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            batch_size: Not used (kept for compatibility)
            request_delay: Delay between API requests in seconds
        """
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        print(f"\n{'='*80}")
        print(f"Starting generation with OpenAI API")
        print(f"Model: {self.model_name}")
        print(f"Total examples: {len(dataset)}")
        print(f"Sequences per example: {num_return_sequences}")
        print(f"{'='*80}\n")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate for each example
        all_results = []
        
        for idx, example in enumerate(tqdm(dataset, desc="Generating tests")):
            prompt = example['question']
            
            # Generate
            generated_texts = self.generate_single(
                prompt=prompt,
                num_return_sequences=num_return_sequences,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Save result
            result = {
                'task_id': example['task_id'],
                'prompt': prompt,
                'generated': generated_texts,
                'function_name': example['function_name'],
                'file_path': example.get('file_path', ''),
            }
            all_results.append(result)
            
            # Save intermediate results every 10 examples
            if (idx + 1) % 10 == 0:
                self._save_results(all_results, save_dir)
            
            # Add delay to avoid rate limits
            if idx < len(dataset) - 1:
                time.sleep(request_delay)
        
        # Save final results
        self._save_results(all_results, save_dir)
        
        print(f"\n{'='*80}")
        print(f"Generation complete!")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*80}\n")
    
    def _save_results(self, results: List[Dict], save_dir: str):
        """Save generation results to JSON file."""
        output_file = os.path.join(save_dir, "generated_tests.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save in JSONL format
        output_jsonl = os.path.join(save_dir, "generated_tests.jsonl")
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="OpenAI model name (e.g., gpt-4, gpt-3.5-turbo)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default=None, help="Custom API base URL (optional)")
    parser.add_argument("--task_name", type=str, default="Rust", help="Task name")
    parser.add_argument("--split", type=str, required=True, help="Dataset split")
    parser.add_argument("--lang", type=str, choices=["Rust", "Go", "Julia"], default="Rust", help="Programming language")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--save_dir", type=str, default="", required=True, help="Output directory")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to generate")
    
    # Prompt type
    parser.add_argument('--instruct_model', action="store_true", help="Use instruction template")
    parser.add_argument('--arg_context', action="store_true", help="Include argument context")
    parser.add_argument('--file_context', action="store_true", help="Include file context")
    
    # RAG options
    parser.add_argument('--rag_type', type=str, choices=["bm25", "dense"], default=None, help="RAG retrieval type")
    
    # Few-shot option (JSONL)
    parser.add_argument('--use_fewshot_jsonl', action="store_true", help="Use few-shot JSONL with retrieved examples")

    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.2, help="Sampling temperature")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling")
    parser.add_argument('--request_delay', type=float, default=0.5, help="Delay between API requests (seconds)")
    parser.add_argument('--retry_count', type=int, default=3, help="Number of retries on API failure")

    opt = parser.parse_args()

    # System prompt for instruction models
    system_prompt = "You are a helpful coding assistant. Your task is to generate unittest for a given function."

    # Determine template
    if opt.instruct_model:
        template = prompts[opt.lang]["instruct"]
    else:
        template = prompts[opt.lang]["base"]

    # Determine data path
    if opt.use_fewshot_jsonl:
        # Use HuggingFace dataset + few-shot JSONL examples
        data_path = fewshot_datapath[opt.lang]
        print(f"Using HuggingFace dataset: {data_path['repo']} (split: {data_path['split']})")
        print(f"With few-shot examples from: {data_path['fewshot_file']}")
    elif opt.rag_type:
        # Use RAG data
        if opt.rag_type == "bm25":
            data_path = bm_25_rag_datapath[opt.lang]
        else:
            data_path = dense_rag_datapath[opt.lang]
        print(f"Using RAG data: {opt.rag_type}")
    else:
        # Use standard dataset from HuggingFace
        data_path = datapath[opt.lang]
        print(f"Using standard HuggingFace dataset: {data_path['repo']} (split: {data_path['split']})")
    
    suffix = suffixes[opt.lang]
    task_name = opt.task_name

    # Create output directory
    os.makedirs(opt.save_dir, exist_ok=True)
    print(f"Output directory: {opt.save_dir}")

    # Initialize task
    task = TestGenOpenAI(
        task_name=task_name,
        dataset_path=data_path,
        model_name=opt.model,
        api_key=opt.api_key,
        base_url=opt.base_url,
        split=opt.split,
        instruct=opt.instruct_model,
        system_prompt=system_prompt,
        template=template,
        suffix=suffix,
        arg_context=opt.arg_context,
        is_rag=opt.rag_type is not None,
        file_context=opt.file_context,
        use_fewshot_jsonl=opt.use_fewshot_jsonl
    )

    # Print sample
    dataset = task.prepare_dataset()
    print("=" * 25 + " Test Sample " + "=" * 25)
    sample_idx = min(50, len(dataset) - 1)
    print(dataset[sample_idx]['question'])
    print(f"\nTotal examples: {len(dataset)}")
    print("=" * 63)

    # Generate
    task.generate(
        save_dir=opt.save_dir,
        max_tokens=opt.max_tokens,
        num_return_sequences=opt.num_return_sequences,
        temperature=opt.temperature,
        top_p=opt.top_p,
        request_delay=opt.request_delay
    )
