def recursive_descent(replacement_rules, terminals, tokens, cursor_index, looking_for):
    current_token = tokens[cursor_index]

    used_rule, used_replacement = None, None

    for rule in replacement_rules:
        root, replacements = rule

        if used_rule is not None:
            break

        if root is not looking_for:
            continue

        for rep in replacements:

            current_token_name, span = current_token
            if len(rep) == 0 or current_token_name == rep[0]:
                used_rule, used_replacement = rule, rep
                break

    if used_replacement is None:
        print("lmao can't find shit")

    print(looking_for, current_token)

    new_cursor = cursor_index
    children = []
    for character in used_replacement:

        if character in terminals:
            new_cursor += 1
            children.append(character)
            print(f"accepting character {character}")
        else:
            print(f"recurring at {new_cursor} to look for a {character}")
            cursor_change, sub_tree = recursive_descent(replacement_rules, terminals, tokens, new_cursor, character)
            new_cursor += cursor_change
            children.append(sub_tree)

    index_change = new_cursor - cursor_index

    return index_change, (used_rule[0], children)


def print_tree(tree, depth=0):
    if isinstance(tree, str):
        print("\t" * depth + tree)
    else:

        root, children = tree
        print("\t" * depth + root)
        for c in children:
            print_tree(c, depth + 1)


if __name__ == '__main__':
    replacement_rules = [
        ("OB", [
            ["&"],
            ["|"]
        ]),

        ("E", [
            ["(", "E", "OB", "E", ")"],
            ["!", "E"],
            ["f", "FB"],
            ["a"],
            ["b"],
            ["c"],
            ["d"]
        ]),

        ("FB", [
            ["(", "E", "FB+"],
        ]),

        ("FB+", [
            [")"],
            [",", "E", "FB+"],
            []
        ])

    ]
    terminals = {"&", "|", "(", ")", "!", "f", ",", "a", "b", "c", "d"}
    string = "((f(a, b, c, d) & f(a, b, c, d)) | f(a, b, c, d))".replace(" ", "")
    token_list = []
    for char in string:
        token_list.append((char, char))

    change, tree = recursive_descent(replacement_rules, terminals, token_list, 0, "E")
    print_tree(tree)
