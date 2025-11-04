import json
import re
import tree_sitter_rust as tsrust
import tree_sitter_go as ts_go
import tree_sitter_julia as ts_julia
from datasets import load_dataset
from tree_sitter import Language, Parser
from collections import deque

rust_lang=Language(tsrust.language())
go_lang=Language(ts_go.language())
julia_lang= Language(ts_julia.language())

PARSERS = {
    "go": Parser(go_lang),
    "rust": Parser(rust_lang),
    "julia":Parser(julia_lang),
}


def rust_traverse_function_node(node, ignore_kind=None, avoid_nested=True):
    results=[]
    if ignore_kind is None:
        ignore_kind = []
    queue =deque([node])
    while queue:
        cur_node = queue.popleft()
        if (cur_node.type=='attribute_item')&(cur_node.text.decode()=="#[test]"):
            next_node = queue.popleft()             
            if next_node.type=='function_item':
                test_functions=("    #[test]\n    " + next_node.text.decode("utf-8"))
                results.append(test_functions)
            else:
                queue.appendleft(next_node)

        if cur_node.type == 'function_item':
            if ("test_" in cur_node.text.decode("utf-8").splitlines()[0]) or ("_test" in cur_node.text.decode("utf-8").splitlines()[0]):
                test_functions=("    #[test]\n    " + cur_node.text.decode("utf-8"))
                results.append(test_functions)
            if avoid_nested:
                continue
        if not cur_node.children:
            continue
        for child in cur_node.children:
            if child.type not in ignore_kind:
                queue.append(child)
    return results



def go_traverse_function_node(node, ignore_kind=None, avoid_nested=True):
    results=[]
    if ignore_kind is None:
        ignore_kind = []
    queue =deque([node])
    while queue:
        cur_node = queue.popleft()
   
        if cur_node.type == 'function_declaration' and "{" in cur_node.text.decode("utf-8") and "}" in cur_node.text.decode("utf-8"):
            if ("Test" in cur_node.text.decode("utf-8").splitlines()[0]):
                test_functions=cur_node.text.decode("utf-8")
                results.append(test_functions)
            if avoid_nested:
                continue
        if not cur_node.children:
            continue
        for child in cur_node.children:
            if child.type not in ignore_kind:
                queue.append(child)
    return results

def get_test_functions(raw_test_list,lang):
    lang=lang.lower()
    parser= PARSERS[lang]
    test_functions=[]
    for test in raw_test_list:
        if lang =="rust":
            tree = parser.parse(bytes(test, 'utf-8'))
            test_functions.extend(rust_traverse_function_node(tree.root_node))
        elif lang =="go":
            
            split=test.split("func Test",1)[-1]
            test="func Test"+split[0].upper() + split[1:]
            tree = parser.parse(bytes(test, 'utf-8'))
            test_functions.extend(go_traverse_function_node(tree.root_node))
        else:
            split=test.split("@testset",1)[-1]
            test="@testset"+split
            test_functions.append(test)
    return test_functions

def sanitize_output(response,lang):
    test_function_list=get_test_functions(response,lang)
    if len(test_function_list)>10:
       test_function_list=test_function_list[:10]
    
    return test_function_list

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, required=True,help='Input JSONL file ')
    parser.add_argument('--processed_input_directory', type=str,default="",help='Output JSONL file to save processed data')
    parser.add_argument('--lang', type=str, required=True,help='Programming language (go, rust, julia)')

    args = parser.parse_args()
    lang=args.lang.lower()
    processed_input_directory=f"evaluation/data/processed_input/{lang}/processed_input.jsonl" if args.processed_input_directory=="" else args.processed_input_directory
    with open(args.input_directory, 'r') as f:
        response_data = [json.loads(line) for line in f]
    
    for i,sample in enumerate(response_data):
        raw_response=["\n".join(sample["prompt"].split("\n")[-1:]) +response for response in sample["response"]]
        sample["raw_response"]=raw_response
        processed_response=sanitize_output(raw_response,lang)
        sample["response"]=processed_response 
        del sample["prompt"]
    with open(processed_input_directory, 'w') as f:
        for sample in response_data:
            f.write(json.dumps(sample) + '\n')
    print(f"Processed data saved to {processed_input_directory}")

if __name__ == "__main__":
    main()
    
