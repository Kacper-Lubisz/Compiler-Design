def recursive_descent(replacement_rules, tokens, cursor_index):
    current_token = tokens[cursor_index]

    used_rule, used_replacement = None, None

    for rule in replacement_rules:
        root, replacements = rule
	
        if used_rule is not None:
            break

        for rep in replacements:
            
            current_token_name, span = current_token
            if len(rep) == 0 or current_token_name == rep[0]:
                used_rule, used_replacement = rule, rep
                break
    
    
    new_cursor= cursor_index
    children = []
    for character in used_replacement:
        if chaaracter == character.lower():
	    new_cursor += 1
        else:
            cursor_change, tree = recursive_descent(replacement_rules, tokens, new_cursor)
            new_cursor += cursor_change
            children.append(tree)

    index_change = new_cursor - cursor_index
    tree = (used_rule[0], children)
    print(cursor_index, tree)
    
    return index_change, tree

if __name__ == '__main__':
    replacement_rules = [
        ("S", [
            ["s","F"],
            ["@", "L"],
            []
        ]),

        ("F", [
            ["a", "P"]
        ]),
        ("P", [
            ["_", "P"],
            []
        ]),

        ("L", [
            ["b", "B"]
        ]),
        ("B", [
            ["+", "B"],
            []
        ])
    ]

    string = "@a_____b++++a_____b+++++"
    token_list = []
    for char in string:
        token_list.append((char, char))

    print(token_list)

    change, tree = recursive_descent(replacement_rules, token_list, 0)
    print(tree)
