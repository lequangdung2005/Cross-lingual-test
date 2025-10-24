suffixes = {
    "Rust": """    #[test]\n    fn test_{function_name}() {{""",
    "Julia": """@testset \"{function_name} Tests\" begin""",
    "Go": """func Test{function_name}("""
}

prompts = {
    "Rust": {
        "instruct": """{function_code}\nGenerate Rust unittest for `{function_name}` function in module `{file_path}`:\n""",
        "base": """{function_code}\n// Check the correcness for {function_name} function in Rust\n"""
    },
    "Julia": {
        "instruct": """{function_code}\nGenerate Julia unittest for `{function_name}` function in module `{file_path}`:\n""",
        "base": """{function_code}\n# Check the correcness for {function_name} function in Julia\n"""
    },
    "Go": {
        "instruct": """{function_code}\nGenerate Go unittest for `{function_name}` function in module `{file_path}`:\n""",
        "base": """{function_code}\n// Check the correcness for {function_name} function in Go\n"""
    },
}

datapath = {
    "Rust": {"repo": "Tessera2025/Tessera2025", "split": "rust"},
    "Go": {"repo": "Tessera2025/Tessera2025", "split": "go"},
    "Julia": {"repo": "Tessera2025/Tessera2025", "split": "julia"}
}

bm_25_rag_datapath = {
    "Rust": {"repo": "json", "file": None},
    "Go": {"repo": "json", "file": None},
    "Julia": {"repo": "json", "file": None}
}

dense_rag_datapath = {
    "Rust": {"repo": "json", "file": None},
    "Go": {"repo": "json", "file": None},
    "Julia": {"repo": "json", "file": None}
}

# Few-shot prompt directories (constructed prompts from retrieval)
fewshot_prompt_dir = {
    "Rust": "data/constructed_prompt/unixcoder/rust",
    "Go": "data/constructed_prompt/unixcoder/go",
    "Julia": "data/constructed_prompt/unixcoder/julia"
}

# Few-shot JSONL data paths (prompts embedded in data)
fewshot_datapath = {
    "Rust": {"repo": "Tessera2025/Tessera2025", "split": "rust", "fewshot_file": "data/constructed_prompt/unixcoder/rust/data_with_fewshot.jsonl"},
    "Go": {"repo": "Tessera2025/Tessera2025", "split": "go", "fewshot_file": "data/constructed_prompt/unixcoder/go/data_with_fewshot.jsonl"},
    "Julia": {"repo": "Tessera2025/Tessera2025", "split": "julia", "fewshot_file": "data/constructed_prompt/unixcoder/julia/data_with_fewshot.jsonl"}
}