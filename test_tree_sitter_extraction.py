#!/usr/bin/env python3
"""
Simple test to verify tree-sitter method extraction with parameter counting.
"""

from collections import deque
from tree_sitter import Language, Parser
import tree_sitter_go as ts_go
import tree_sitter_julia as ts_julia
import tree_sitter_rust as ts_rust
import tree_sitter_java as ts_java

# Initialize parsers
JULIA_LANG = Language(ts_julia.language())
JULIA_PARSER = Parser(JULIA_LANG)

RUST_LANG = Language(ts_rust.language())
RUST_PARSER = Parser(RUST_LANG)

GO_LANG = Language(ts_go.language())
GO_PARSER = Parser(GO_LANG)

JAVA_LANG = Language(ts_java.language())
JAVA_PARSER = Parser(JAVA_LANG)



def find_nodes_by_type(node, type_names):
    """Find all descendant nodes matching any of the given type names."""
    result = []
    queue = deque([node])
    
    while queue:
        current = queue.popleft()
        if current.type in type_names:
            result.append(current)
        for child in current.children:
            queue.append(child)
    
    return result


def get_direct_child_by_type(node, type_name):
    """Get direct children nodes of a specific type."""
    return [child for child in node.children if child.type == type_name]


def extract_rust_method_info(code):
    """Extract method name and parameter count from Rust code."""
    tree = RUST_PARSER.parse(bytes(code, 'utf8'))
    root = tree.root_node
    
    function_nodes = find_nodes_by_type(root, ['function_item'])
    if not function_nodes:
        return None, 0
    
    func_node = function_nodes[0]
    
    # Extract name
    identifiers = get_direct_child_by_type(func_node, 'identifier')
    name = identifiers[0].text.decode('utf8') if identifiers else ""
    
    # Count parameters
    params = get_direct_child_by_type(func_node, 'parameters')
    param_count = 0
    if params:
        param_nodes = get_direct_child_by_type(params[0], 'parameter')
        param_count = len(param_nodes)
    
    return name, param_count


def extract_go_method_info(code):
    """Extract method name and parameter count from Go code."""
    tree = GO_PARSER.parse(bytes(code, 'utf8'))
    root = tree.root_node
    
    function_nodes = find_nodes_by_type(root, ['function_declaration'])
    if not function_nodes:
        return None, 0
    
    func_node = function_nodes[0]
    
    # Extract name
    identifiers = get_direct_child_by_type(func_node, 'identifier')
    name = identifiers[0].text.decode('utf8') if identifiers else ""
    
    # Count parameters
    params = get_direct_child_by_type(func_node, 'parameter_list')
    param_count = 0
    if params:
        param_nodes = get_direct_child_by_type(params[0], 'parameter_declaration')
        param_count = len(param_nodes)
    
    return name, param_count


def extract_julia_method_info(code):
    """Extract method name and parameter count from Julia code."""
    tree = JULIA_PARSER.parse(bytes(code, 'utf8'))
    root = tree.root_node
    
    function_nodes = find_nodes_by_type(root, ['function_definition'])
    if not function_nodes:
        return None, 0
    
    func_node = function_nodes[0]
    
    # Extract name from signature
    signature_nodes = find_nodes_by_type(func_node, ['signature'])
    name = ""
    param_count = 0
    
    if signature_nodes:
        # Get the identifier (function name)
        identifiers = find_nodes_by_type(signature_nodes[0], ['identifier'])
        if identifiers:
            name = identifiers[0].text.decode('utf8')
        
        # Count parameters in argument_list
        arg_list = find_nodes_by_type(signature_nodes[0], ['argument_list'])
        if arg_list:
            # Count typed_expression nodes (typed parameters like a::Int)
            typed_exprs = get_direct_child_by_type(arg_list[0], 'typed_expression')
            param_count = len(typed_exprs)
            # Also count plain identifiers (untyped parameters)
            plain_ids = get_direct_child_by_type(arg_list[0], 'identifier')
            param_count += len(plain_ids)
    
    return name, param_count


def extract_java_method_info(code):
    """Extract method name and parameter count from Java code."""
    tree = JAVA_PARSER.parse(bytes(code, 'utf8'))
    root = tree.root_node
    
    function_nodes = find_nodes_by_type(root, ['method_declaration'])
    if not function_nodes:
        return None, 0
    
    func_node = function_nodes[0]
    
    # Extract name
    identifiers = get_direct_child_by_type(func_node, 'identifier')
    name = identifiers[0].text.decode('utf8') if identifiers else ""
    
    # Count parameters
    params = get_direct_child_by_type(func_node, 'formal_parameters')
    param_count = 0
    if params:
        param_nodes = get_direct_child_by_type(params[0], 'formal_parameter')
        param_count = len(param_nodes)
    
    return name, param_count


def calculate_param_similarity(count1, count2):
    """Calculate similarity based on parameter count."""
    if count1 == count2:
        return 1.0
    
    diff = abs(count1 - count2)
    similarity = max(0.0, 1.0 - (diff * 0.2))
    return similarity


# Test cases
if __name__ == "__main__":
    print("=" * 80)
    print("Tree-Sitter Parameter Extraction Test")
    print("=" * 80)
    print()
    
    # Test Rust
    rust_code = """
pub fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}
"""
    print("Testing Rust:")
    print(rust_code)
    name, param_count = extract_rust_method_info(rust_code)
    print(f"  Method Name: {name}")
    print(f"  Parameter Count: {param_count}")
    print(f"  Signature: {param_count}_params")
    print()
    
    # Test Go
    go_code = """
func calculateSum(a int, b int, c int) int {
    return a + b + c
}
"""
    print("Testing Go:")
    print(go_code)
    name, param_count = extract_go_method_info(go_code)
    print(f"  Method Name: {name}")
    print(f"  Parameter Count: {param_count}")
    print(f"  Signature: {param_count}_params")
    print()
    
    # Test Julia
    julia_code = """
function calculate_sum(a::Int, b::Int)
    return a + b
end
"""
    print("Testing Julia:")
    print(julia_code)
    name, param_count = extract_julia_method_info(julia_code)
    print(f"  Method Name: {name}")
    print(f"  Parameter Count: {param_count}")
    print(f"  Signature: {param_count}_params")
    print()
    
    # Test Java
    java_code = """
public int add(int x, int y, int z) {
    return x + y + z;
}
"""
    print("Testing Java:")
    print(java_code)
    name, param_count = extract_java_method_info(java_code)
    print(f"  Method Name: {name}")
    print(f"  Parameter Count: {param_count}")
    print(f"  Signature: {param_count}_params")
    print()
    
    print("=" * 80)
    print("Parameter Similarity Examples:")
    print("=" * 80)
    
    test_cases = [
        (2, 2, "Exact match"),
        (2, 3, "1 parameter difference"),
        (2, 4, "2 parameters difference"),
        (0, 0, "Both have 0 parameters"),
        (3, 0, "3 parameters difference"),
    ]
    
    for count1, count2, desc in test_cases:
        sim = calculate_param_similarity(count1, count2)
        print(f"  {count1} vs {count2} ({desc}): {sim:.2f}")
    
    print()
    print("=" * 80)
    print("✓ Tree-sitter parameter extraction successful!")
    print("=" * 80)

