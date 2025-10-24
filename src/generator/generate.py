"""
Test Generation Module

This module handles test generation with multiple prompt strategies:

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

from typing import Any
from string import ascii_uppercase
import argparse
import os
import json
import textwrap
from code_eval import Evaluator
from datasets import load_dataset, Dataset
from utils import delete_test_from_filecontext, get_import_context, rust_parser, go_parser, julia_parser

from code_eval.tasks.base import TaskBase
from transformers import AutoTokenizer
from constant import prompts, suffixes, datapath, bm_25_rag_datapath, dense_rag_datapath, fewshot_prompt_dir, fewshot_datapath

lang_parsers = {
    "rust": rust_parser,
    "go": go_parser,
    "julia": julia_parser
}


class TestGen(TaskBase):
    def __init__(
        self, 
        task_name, 
        dataset_path, 
        model_name, 
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
        Initialize TestGen.
        
        Args:
            task_name: Language task name (Rust, Go, Julia)
            dataset_path: Dataset path configuration
            model_name: Model name for tokenizer
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
        self.instruct = instruct
        self.template = template
        self.suffix = suffix
        self.arg_context = arg_context
        self.is_rag = is_rag
        self.file_context = file_context
        self.use_fewshot_jsonl = use_fewshot_jsonl
        self.system_prompt = system_prompt
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
    
    def _apply_chat_template(self, prompt):
        """Apply chat template for instruction models."""
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        
        messages.append({"role": "user", "content": prompt})
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
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
                    
                    if self.instruct:
                        prompt = self._apply_chat_template(prompt)
                    
                    example['question'] = self._add_suffix(prompt, example["function_name"])
                    example['task_id'] = i
                    processed_examples.append(example)
                    continue
            
            # Standard prompt construction (no few-shot or few-shot not available)
            code, prompt = self._build_standard_prompt(example)
            
            # Add contexts if needed
            if self.instruct:
                prompt = self._add_context_to_prompt(example, prompt)
                prompt = self._apply_chat_template(prompt)
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
        
        # Convert to Dataset
        return Dataset.from_list(processed_examples)
   
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--task_name", type=str, default="Rust", help="Task name")
    parser.add_argument("--split", type=str, required=True, help="Dataset split")
    parser.add_argument("--lang", type=str, choices=["Rust", "Go", "Julia"], default="Rust", help="Programming language")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--cache_dir", type=str, default="", help="Cache directory")
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
    parser.add_argument('--use_beam_search', action="store_true", help="Use beam search")
    parser.add_argument('--num_beam', type=int, default=10, help="Number of beams")
    parser.add_argument('--do_sample', action="store_true", help="Use sampling")
    parser.add_argument('--top_p', type=float, default=0.95, help="Top-p sampling")
    parser.add_argument('--top_k', type=int, default=0, help="Top-k sampling")
    parser.add_argument('--temperature', type=float, default=0.2, help="Sampling temperature")
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help="Repetition penalty")

    opt = parser.parse_args()

    # System prompt for instruction models
    system_prompt = "You are a helpful coding assistant. Your task is to generate unittest for a given function."

    # Determine template
    if opt.instruct_model:
        template = prompts[opt.lang]["instruct"]
    else:
        template = prompts[opt.lang]["base"]

    # Special template for LoRA fine-tuned models
    if opt.lora_path and not opt.instruct_model:
        template = """{function_code}\n""" + f"<{opt.lang.lower()}_test>\n"

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
    task = TestGen(
        task_name=task_name,
        dataset_path=data_path,
        split=opt.split,
        instruct=opt.instruct_model,
        system_prompt=system_prompt,
        template=template,
        suffix=suffix,
        arg_context=opt.arg_context,
        model_name=opt.model,
        is_rag=opt.rag_type is not None,
        file_context=opt.file_context,
        use_fewshot_jsonl=opt.use_fewshot_jsonl
    )

    # Initialize evaluator
    save_dir = opt.save_dir
    evaluator = Evaluator(
        task=task,
        model_name=opt.model,
        batch_size=opt.batch_size,
        save_dir=save_dir,
        cache_dir=opt.cache_dir,
        trust_remote_code=True,
        peft_model=opt.lora_path
    )
   
    # Print sample
    print("=" * 25 + " Test Sample " + "=" * 25)
    sample_idx = min(50, len(evaluator.dataset['question']) - 1)
    print(evaluator.dataset['question'][sample_idx])
    print(f"\nTotal examples: {len(evaluator.dataset['question'])}")
    print("=" * 63)

    # Generate
    evaluator.generate(
        backend='vllm',
        max_tokens=opt.max_tokens,
        num_return_sequences=opt.num_return_sequences,
        temperature=opt.temperature,
        do_sample=opt.do_sample,
        top_p=opt.top_p,
        top_k=opt.top_k,
        use_beam_search=opt.use_beam_search,
        num_beam=opt.num_beam,
        repetition_penalty=opt.repetition_penalty
    )