def recursive_descent(replacement_rules, tokens, current_index):
    current_token = tokens[current_index]

    used_rule, used_replacement = None, None

    for rule in replacement_rules:
        root, replacements = rule

        for rep in replacements:
            if current_token == rep[0]:
                used_rule, used_replacement = rule, rep

    for character in used_replacement:
        recursive_descent()


if __name__ == '__main__':
    replacement_rules = [
        ("S", [
            ["F", "S"],
            ["L", "S"],
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

    string = "a_____b++++a_____b+++++"
    token_list = []
    for char in string:
        token_list.append((char, char))

    print(token_list)

    recursive_descent(replacement_rules, token_list, 0)
