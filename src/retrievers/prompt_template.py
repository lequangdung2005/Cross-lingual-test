suffixes = {
    "Rust": """    #[test]\n    fn test_{function_name}() {{""",
    "Julia": """@testset \"{function_name} Tests\" begin""",
    "Go": """func Test{function_name}("""
}

# prompts = {
#     "Rust": {
#         "instruct": """{function_code}\nGenerate Rust unittest for `{function_name}` function in module `{file_path}`:\n""",
#         "base": """{function_code}\n// Check the correcness for {function_name} function in Rust\n"""
#     },
#     "Julia": {
#         "instruct": """{function_code}\nGenerate Julia unittest for `{function_name}` function in module `{file_path}`:\n""",
#         "base": """{function_code}\n# Check the correcness for {function_name} function in Julia\n"""
#     },
#     "Go": {
#         "instruct": """{function_code}\nGenerate Go unittest for `{function_name}` function in module `{file_path}`:\n""",
#         "base": """{function_code}\n// Check the correcness for {function_name} function in Go\n"""
#     },
# }

# datapath = {
#     "Rust": {"repo": "Tessera2025/Tessera2025", "file": None},
#     "Go": {"repo": "Tessera2025/Tessera2025", "file": None},
#     "Julia": {"repo": "Tessera2025/Tessera2025", "file": None}
# }

# bm_25_rag_datapath = {
#     "Rust": {"repo": "json", "file": None},
#     "Go": {"repo": "json", "file": None},
#     "Julia": {"repo": "json", "file": None}
# }

# dense_rag_datapath = {
#     "Rust": {"repo": "json", "file": None},
#     "Go": {"repo": "json", "file": None},
#     "Julia": {"repo": "json", "file": None}
# }