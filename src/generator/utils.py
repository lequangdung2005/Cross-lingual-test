from tree_sitter import Language, Parser
import tree_sitter_go as ts_go
import tree_sitter_julia as ts_julia
import tree_sitter_rust as ts_rust

Julia_lang=Language(ts_julia.language())
julia_parser = Parser(Julia_lang)

Rust_lang=Language(ts_rust.language())
rust_parser = Parser(Rust_lang)

Go_lang=Language(ts_go.language())
go_parser = Parser(Go_lang)

from collections import deque
def get_child_node_by_name(node,node_name):
    q=deque([node])
    result=[]
    while q:
        curr=q.popleft()
        if curr.type==node_name:
            result.append(curr)
            continue
        for child in curr.children:
            q.append(child)
    return result
def get_next_child_node_by_name(node,node_name):
    result=[]
    for child in node.children:
        if child.type==node_name:
            result.append(child)
    return result

def delete_test_from_filecontext(parser, content) -> None:

    code = content
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    # collect spans to delete
    spans_to_delete = []
    new_code = code

    mod_nodes=get_child_node_by_name(root,"mod_item")
    for mod_node in mod_nodes:
        identifiers=get_next_child_node_by_name(mod_node,"identifier")
        for identifier in identifiers:
            if "test" in identifier.text.decode("utf8") and (mod_node.start_byte, mod_node.end_byte) not in spans_to_delete:
                spans_to_delete.append((mod_node.start_byte, mod_node.end_byte))
                break
    for start, end in sorted(spans_to_delete, key=lambda x: -x[0]):
        new_code = code[:start] + code[end:]
    spans_to_delete = []

    tree = parser.parse(bytes(new_code, "utf8"))
    root = tree.root_node
    function_nodes = get_child_node_by_name(root, "function_item")
    for function in function_nodes:
        block_node = get_next_child_node_by_name(function, "block")[0]
        if "assert" in block_node.text.decode("utf8") and (function.start_byte, function.end_byte) not in spans_to_delete:
            spans_to_delete.append((function.start_byte, function.end_byte))

    for start, end in sorted(spans_to_delete, key=lambda x: -x[0]):
        new_code = new_code[:start] + new_code[end:]

    spans_to_delete = []
    tree = parser.parse(bytes(new_code, "utf8"))
    root = tree.root_node

    test_attributes = get_child_node_by_name(root, "attribute_item")
    for attribute in test_attributes:
        if attribute.text.decode()=="#[test]" or attribute.text.decode()=="#[cfg(test)]":
            spans_to_delete.append((attribute.start_byte, attribute.end_byte))
    for start, end in sorted(spans_to_delete, key=lambda x: -x[0]):
        new_code = new_code[:start] + new_code[end:]

    spans_to_delete = []
    comment_nodes=get_child_node_by_name(root,"line_comment")
    comment_nodes.extend(get_child_node_by_name(root,"block_comment"))
    for comment in comment_nodes:
        if "test" in comment.text.decode("utf8") or "assert" in comment.text.decode("utf8"):
            spans_to_delete.append((comment.start_byte, comment.end_byte))

    for start, end in sorted(spans_to_delete, key=lambda x: -x[0]):
        new_code = new_code[:start] + new_code[end:]

    return new_code


def find_descendants_by_type_non_strict(node, type_name_list, avoid_type):
    result=[]
    if not avoid_type:  
        avoid_type = []
    nodes=deque([node])
    while nodes:
        current=nodes.popleft()
        if current.type in avoid_type:
            continue
        if any(type_name in current.type for type_name in type_name_list):
            result.append(current)
            continue
        for child in current.children:
            if child.type not in avoid_type:
                nodes.append(child)
    return result

def get_import_context(lang,file_content):
    import_list_nodes=[]
    if lang=="julia":
        parser=julia_parser
        import_types=["import_statement","using_statement"]
    elif lang=="rust":
        parser=rust_parser
        import_types=["use_declaration"]
    elif lang=="go":
        parser=go_parser
        import_types=["import_declaration"]
    tree=parser.parse(bytes(file_content,'utf8'))
    import_list_nodes=find_descendants_by_type_non_strict(tree.root_node,import_types, avoid_type=["comment","mod_item"])
    import_list=[child.text.decode("utf8") for child in import_list_nodes]

    return "\n".join(import_list)