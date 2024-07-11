from tree_sitter import Language, Parser

# Load the Haskell grammar
Language.build_library(
    'build/my-languages.so',
    ['./src/tree-sitter-haskell']
)

HASKELL_LANGUAGE = Language('build/my-languages.so', 'haskell')
parser = Parser()
parser.set_language(HASKELL_LANGUAGE)


def extract_haskell_functions(haskell_code):
    tree = parser.parse(bytes(haskell_code, "utf8"))
    root_node = tree.root_node

    functions = []
    comments = []

    def extract_functions(node):
        if node.type == 'comment':
            comment_text = node.text.decode('utf8')
            if comment_text.startswith('{-@') and '::' in comment_text and '@-}' in comment_text:
                comment_parts = comment_text[3:-3].split('::')
                comment_func_name = comment_parts[0].strip()
                comment_content = comment_parts[1].strip()
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                comments.append({
                    'name': comment_func_name,
                    'content': comment_text,
                    'start_line': start_line,
                    'end_line': end_line
                })
        elif node.type in {'function', 'signature'}:
            func_name = None
            for child in node.children:
                if child.type == 'variable':
                    func_name = child.text.decode('utf8')
                    break

            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            code_snippet = '\n'.join(haskell_code.split('\n')[start_line - 1:end_line])

            if func_name:
                functions.append({
                    'type': node.type,
                    'name': func_name,
                    'start_line': start_line,
                    'end_line': end_line,
                    'code': code_snippet
                })
            else:
                functions.append({
                    'type': node.type,
                    'name': node.text.decode('utf8'),
                    'start_line': start_line,
                    'end_line': end_line,
                    'code': code_snippet
                })
        else:
            for child in node.children:
                extract_functions(child)

    extract_functions(root_node)
    return functions, comments


def group_functions(functions, comments):
    grouped_functions = []
    current_group = []

    for func in functions:
        # Find leading comment for the function if it exists
        if not current_group or func['name'] != current_group[-1]['name']:
            if current_group:
                grouped_functions.append(current_group)
            current_group = []
            leading_comment = next((comment for comment in comments if comment['name'] == func['name'] and comment['end_line'] < func['start_line']), None)
            if leading_comment:
                current_group.append({
                    'type': 'leading_comment',
                    'name': func['name'],
                    'start_line': leading_comment['start_line'],
                    'end_line': leading_comment['end_line'],
                    'code': leading_comment['content']
                })

        current_group.append(func)

    if current_group:
        grouped_functions.append(current_group)

    return grouped_functions


def extract_and_group_haskell_functions(haskell_code):
    functions, comments = extract_haskell_functions(haskell_code)
    grouped_functions = group_functions(functions, comments)

    grouped_functions_dict = {}
    for group in grouped_functions:
        group_code = "\n".join(func['code'] for func in group)
        function_name = group[0]['name']
        grouped_functions_dict[function_name] = group_code

    return grouped_functions_dict


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_haskell_functions.py <haskell_file_path>")
        sys.exit(1)

    haskell_file_path = sys.argv[1]

    # Read Haskell code from the file provided as a command-line argument
    with open(haskell_file_path, 'r') as file:
        haskell_code = file.read()

    grouped_functions_dict = extract_and_group_haskell_functions(haskell_code)

    for function_name, group_code in grouped_functions_dict.items():
        print(f"Function: {function_name}")
        print(f"Code:\n{group_code}\n")
