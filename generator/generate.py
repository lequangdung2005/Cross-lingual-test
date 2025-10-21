
from typing import Any
from string import ascii_uppercase
import argparse
import os
import textwrap
from code_eval import Evaluator
from datasets import load_dataset
from utils import delete_test_from_filecontext, get_import_context, rust_parser, go_parser, julia_parser

from code_eval.tasks.base import TaskBase
from transformers import AutoTokenizer
from constant import *

lang_parsers = {
    "rust": rust_parser,
    "go": go_parser,
    "julia": julia_parser
}


class TestGen(TaskBase):
    def __init__(self, task_name, dataset_path, model_name, split="train", instruct=False, system_prompt= None, template=None, suffix=None, arg_context= False, is_rag=False, file_context=False) -> None:
        self.TASK_NAME=task_name
        self.instruct = instruct
        self.template = template
        self.suffix = suffix
        self.arg_context = arg_context
        self.is_rag=is_rag
        self.file_context= file_context
       
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # super().__init__()
        
        if dataset_path["file"] is None:
            self.dataset = load_dataset(dataset_path["repo"], split = split)
        else:
            self.dataset = load_dataset(dataset_path["repo"], data_files = dataset_path["file"], split = split)
        print(self.dataset)
        
        
    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:

        dataset = self.dataset

        def _preprocess(example):


            if "class_signature" in example and example["class_signature"]: # Class wrapper for Rust
                code = "{} {{\n{}\n}}".format(example["class_signature"], textwrap.indent(text=example["focal_code"], prefix='    '))
            else:
                code = example["focal_code"]

            
            if self.instruct:
                prompt = self.template.format(function_code=code, 
                                                function_name=example["function_name"], 
                                                file_path=example["file_path"])
                
                if self.is_rag:
                    prompt = "#RELEVANT CONTEXT\n" + example["context"] + "\n#END_OF_CONTEXT\n\n" + prompt
                
                elif self.file_context:
                    if self.TASK_NAME.lower() == "rust":
                        file_content = delete_test_from_filecontext(lang_parsers[self.TASK_NAME.lower()], example["file_content"])
                    else:
                        file_content = example["file_content"]
                    prompt = "\n#CURRENT FILE: {}\n{}\n#ENDFILE\n".format(example["file_path"], file_content) + prompt
                

                # if self.TASK_NAME.lower() == "julia" and self.arg_context:
                #     if example["argumennt_context"]:
                #         prompt = "#ARGUMENT CONTEXT\n{}\n#END_ARG_CONTEXT".format("\n".join([x["definition_content"] for x in example["argumennt_context"]]).strip()) + "\n\n" + prompt
                    
                if self.arg_context and example["function_component"]["argument_definitions"]:
                    # if self.TASK_NAME.lower() == "julia":
                    #     pre_prompt = "\n".join([x["definition_content"] for x in example["function_component"]["argument_definitions"]]).strip() 
                    # else:
                    pre_prompt = "\n".join([y for x in example["function_component"]["argument_definitions"] for y in x["definitions"]]).strip() 
                    if example["struct_class"]:
                        pre_prompt = "\n\n" + example["struct_class"]
                    prompt = "{}\n{}\n".format(get_import_context(self.TASK_NAME.lower(), example["file_content"]), pre_prompt) + "\n\n" + prompt
                messages = []

                if self.system_prompt:
                    messages = [
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                    ]

                
                messages.append({"role": "user", "content": prompt})
                example['question'] = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self.template.format(function_code=code, 
                                              function_name=example["function_name"])
                # if self.TASK_NAME.lower() == "julia" and self.arg_context:
                #     if example["argumennt_context"]:
                #         prompt = "\n".join([x["definition_content"] for x in example["argumennt_context"]]).strip() + "\n\n" + prompt
                if self.arg_context and example["function_component"]["argument_definitions"]:
                    prompt = "\n".join([y for x in example["function_component"]["argument_definitions"] for y in x["definitions"]]).strip() + "\n\n" + prompt
                
                example['question'] = prompt

            if self.suffix:
                example['question'] = example['question'] + self.suffix.format(function_name=example["function_name"])

        
           
            return example
       
        updated_dataset = dataset.map(_preprocess)
        updated_dataset = updated_dataset.add_column('task_id', list(range(len(dataset))))
        return updated_dataset
   
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="Rust")
    parser.add_argument("--split", type=str)
    parser.add_argument("--lang", type=str, choices= ["Rust", "Go", "Julia"], default="Rust")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument('--instruct_model', action="store_true")
    parser.add_argument('--arg_context', action="store_true")
    parser.add_argument('--file_context', action="store_true")
    parser.add_argument('--rag_type', type=str, choices=["bm25", "dense"], default=None)

    parser.add_argument('--use_beam_search', action="store_true")
    parser.add_argument('--num_beam', type=int, default=10)
    parser.add_argument('--do_sample', action="store_true")
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)


    system_prompt = "You are a helpful coding assistant. Your task is to generate unittest for a given function."

    opt = parser.parse_args()

    if opt.instruct_model:
        template=prompts[opt.lang]["instruct"]
    else:
        template=prompts[opt.lang]["base"]

    if opt.lora_path and not opt.instruct_model:
        template = """{function_code}\n""" + f"<{opt.lang.lower()}_test>\n"


    data_path = datapath[opt.lang]
    if opt.rag_type:
        if opt.rag_type == "bm25":
            data_path = bm_25_rag_datapath[opt.lang]
        else:
            data_path = dense_rag_datapath[opt.lang]
    suffix = suffixes[opt.lang]
    task_name = opt.task_name

    os.makedirs(opt.save_dir, exist_ok=True)
    print(opt.save_dir)

    task = TestGen(task_name=task_name, dataset_path=data_path, split = opt.split, instruct=opt.instruct_model,
                   system_prompt= system_prompt, template=template, suffix=suffix, arg_context=opt.arg_context,
                   model_name = opt.model, is_rag=opt.rag_type is not None, file_context= opt.file_context)

    save_dir = opt.save_dir
    evaluator = Evaluator(task=task,
                        model_name=opt.model,
                        batch_size=opt.batch_size,
                        save_dir=save_dir,
                        cache_dir=opt.cache_dir,
                        trust_remote_code=True,
                        peft_model = opt.lora_path)
   
    print("="*25 + "Test sample" + "="*25)
    print(evaluator.dataset['question'][50])
    print(len(evaluator.dataset['question']))
    print("="*61 )

    evaluator.generate(backend='vllm',
                    max_tokens=opt.max_tokens,
                    num_return_sequences=opt.num_return_sequences,
                    temperature=opt.temperature,
                    do_sample= opt.do_sample,
                    top_p = opt.top_p,
                    top_k = opt.top_k,
                    use_beam_search=opt.use_beam_search,
                    num_beam=opt.num_beam,
                    repetition_penalty=opt.repetition_penalty)