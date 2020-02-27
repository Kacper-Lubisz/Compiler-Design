import sys
from typing import Sequence, MutableSet
from string import ascii_lowercase, ascii_uppercase, digits
import math


# from enum import Enum

def read_file(input_file):
    """
    This method reads in the file and performs some simple pre-processing and **simple** parsing on the input file.
    It returns the parsed FOL definition and the unchanged formula string
    :param input_file: The path to the input file
    :return: The parsed FOL and formula
    """

    with open(input_file, "r") as file:
        content = file.read().replace("\t", " ").replace("\n", " ").strip()
        # file with all white space replaced with ' ' characters

    without_whitespace = ""
    for char in content:
        if char != ' ' or without_whitespace[-1] != ' ':
            without_whitespace += char
    # replace all spans of whitespace with a singular space
    content = without_whitespace

    search_for = [
        "variables",
        "constants",
        "predicates",
        "equality",
        "connectives",
        "quantifiers",
        "formula"
    ]
    found_start_indices = {}
    end_indices = []

    for search_item in search_for:  # find the start and end index of each section
        start_index = content.find(search_item)

        if start_index == -1:
            print("Couldn't find the string '" + search_item + "' in the input file")
            exit(1)

        end_indices.append(start_index)
        start_index += len(search_item)

        # ignore whitespace between identifier and colon
        # e.g. 'variables   :', the white space is ignored
        while content[start_index] != ":":
            start_index += 1
        start_index += 1

        found_start_indices[search_item] = start_index

    search_results = {}

    for search_item in search_for:
        start_index = found_start_indices[search_item]

        possible_ends = list(filter(lambda index: index >= start_index, end_indices))
        if len(possible_ends) == 0:
            end_index = len(content)
        else:
            end_index = min(possible_ends)

        search_results[search_item] = content[start_index:end_index]

    # at this point the strings between the "search_for"s are in the search_results map
    # these also have all white space replaced with spaces and have leading and trailing whitespace removed

    variables = list(filter(lambda item: len(item) != 0, search_results["variables"].split(" ")))
    constants = list(filter(lambda item: len(item) != 0, search_results["constants"].split(" ")))
    equality = search_results["equality"].strip()
    connectives = list(filter(lambda item: len(item) != 0, search_results["connectives"].split(" ")))
    quantifiers = list(filter(lambda item: len(item) != 0, search_results["quantifiers"].split(" ")))
    formula = search_results["formula"]

    parsed_predicates = []
    for predicate in search_results["predicates"].replace(" ", "").split("]"):  # remove all white space

        if len(predicate) == 0 or predicate == " ":
            continue

        split = predicate.split("[")

        if len(split) != 2:
            print("Missing name or arity of predicate in the input file", file=sys.stderr)
            exit(1)
        name, arity = split
        try:
            arity = int(arity)
        except ValueError:
            print("One predicate's arity is not an integer", file=sys.stderr)
            exit(1)

        if arity < 1:
            print("One predicate's arity is not positive", file=sys.stderr)
            exit(1)

        parsed_predicates.append((name, arity))
    predicates = parsed_predicates

    logic = (variables, constants, predicates, equality, connectives, quantifiers)

    print(logic)

    return logic, formula


class SyntaxError(Exception):

    def __init__(self, message, location_string, token):
        self.message = message
        self.location_string = location_string
        self.token = token

    def get_message(self, context_string):
        string = f"Syntax Error " \
                 f"in {self.location_string} {self.token.line}:{self.token.line_start} " \
                 f"(character {self.token.absolute_start})\n"

        string += self.message + "\n"

        line_start = self.token.absolute_start
        line_end = self.token.absolute_start

        while line_start != 0 and context_string[line_start] != '\n':
            line_start -= 1

        if line_start != 0:
            previous_line_start = line_start - 1
            while previous_line_start != 0 and context_string[previous_line_start] != '\n':
                previous_line_start -= 1
        else:
            previous_line_start = None

        while line_end != len(context_string) - 1 and context_string[line_end] != '\n':
            line_end += 1

        if line_end != len(context_string) - 1:
            following_line_end = line_end + 1
            while following_line_end != len(context_string) - 1 and context_string[following_line_end] != '\n':
                following_line_end += 1
        else:
            following_line_end = None

        line_number = self.token.line
        max_line_number_length = max(
            len(str(line_number)),
            0 if previous_line_start is None else len(str(line_number - 1)),
            0 if following_line_end is None else len(str(line_number + 1)),
        )

        if previous_line_start is not None:
            justified_previous_number = str(line_number - 1).rjust(max_line_number_length, ' ')
            string += f"{justified_previous_number} | {context_string[previous_line_start + 1:line_start]}\n"

        justified_line_number = str(line_number).rjust(max_line_number_length, ' ')

        string += f"{justified_line_number} | {context_string[line_start + 1:line_end]}\n"

        print(self.token)
        print(self.token.line_start)

        tab_size = 4
        offset_within_line = 0  # in spaces
        for char in context_string[line_start + 1:line_start + 1 + self.token.line_start]:  # this loop expands tabs
            offset_within_line += 1
            if char == '\t':
                offset_within_line = math.ceil(offset_within_line / tab_size) * tab_size

        span_length = 0
        # this loop expands tabs
        for char in context_string[self.token.absolute_start:self.token.absolute_start + len(self.token.span)]:
            span_length += 1
            if char == '\t':
                span_length = math.ceil(span_length / tab_size) * tab_size

        string += " " * (len(justified_line_number) + 3 + offset_within_line) + "^" * span_length + "\n"

        if following_line_end is not None:
            justified_following_number = str(line_number + 1).rjust(max_line_number_length, ' ')
            string += f"{justified_following_number} | {context_string[line_end + 1:following_line_end]}\n"

        return string


class Token:

    def __init__(self, type, id, span, line, absolute_start, line_start):
        self.type = type
        self.id = id
        self.span = span
        self.line = line
        self.absolute_start = absolute_start
        self.line_start = line_start

    def __str__(self) -> str:
        return f"Token(type: '{self.type}', id: {self.id}, span: '{self.span}', " \
               f"line: {self.line}, absolute_start: {self.absolute_start}, line_start: {self.line_start})"

    def __repr__(self) -> str:
        return self.__str__()


class FSM:

    def __init__(self, states, start_state, transitions):
        self.states = states
        self.transitions = transitions
        self.current_state = self.start_state = start_state

        self.span = ""

    def reset(self):
        self.span = ""
        self.current_state = self.start_state

    def feed(self, character):

        transitions_out = self.transitions[self.current_state]

        if character not in transitions_out and None not in transitions_out:
            self.current_state = None
            self.span += character
            return None

        if character not in transitions_out:
            next_state_name = transitions_out[None]
        else:
            next_state_name = transitions_out[character]

        self.current_state = next_state_name
        evaluate = self.states[next_state_name]

        if evaluate is None:
            self.span += character
            return None

        back_step, token_name = evaluate

        if not back_step:
            self.span += character

        return back_step, token_name, self.span

    def reached_end(self):
        return self.current_state is None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"FSMInstance(state: '{self.current_state}', snap:'{self.span}')"


def tokenise(string: str, machines: Sequence[FSM]):
    token_id = 0
    cursor = 0
    line_start = 0
    line = 0

    tokens = []

    for machine in machines:
        machine.reset()

    length = len(string)
    while cursor != length:
        current_character = string[cursor]

        for current_machine in machines:

            result = current_machine.feed(current_character)

            if result is not None:
                back_step, token_name, span = result

                if back_step:
                    cursor -= 1

                tokens.append(Token(
                    token_name,
                    token_id,
                    span,
                    line,
                    cursor - len(span) + 1,
                    cursor - line_start - len(span) + 1
                ))
                token_id += 1

                for m in machines:  # reset all the machines
                    m.reset()
                break

            if current_machine.reached_end():
                current_machine.reset()

        cursor += 1
        if current_character == '\n':
            line += 1
            line_start = cursor

    # add the EOF
    tokens.append(Token("EOF", token_id, "", line, cursor, cursor - line_start))

    return tokens


def build_repeated_symbol_token_machine(matching_characters, token_name):
    # building a machine which accepts the regex (matching_characters)*

    start = 0
    matching = 1
    accept = 2

    transitions = {
        start: dict(),
        matching: {None: accept}
    }
    for char in matching_characters:
        transitions[start][char] = matching
        transitions[matching][char] = matching

    return FSM({accept: (True, token_name), start: None, matching: None}, start, transitions)


def build_simple_match_token_machine(match, token_name):
    start_state = current = ""

    matched = 0

    states = {
        match: None,
        matched: (True, token_name)  # this state is the accepting state
    }

    transitions = {
        match: {None: matched}
    }

    for char in match:
        next_state = current + char

        states[current] = None
        transitions[current] = {char: next_state}

        current = next_state

    return FSM(states, start_state, transitions)


def build_match_one_of_token_machine(matches, token_name):
    matched = 0

    states = {matched: (True, token_name)}  # this state is the accepting state
    transitions = dict()
    for match in matches:
        states[match] = None
        transitions[match] = {None: matched}

    start_state = ""
    for match in matches:
        current = start_state
        for char in match:
            next_state = current + char

            states[current] = None
            transitions[current] = {char: next_state}

            current = next_state

    return FSM(states, start_state, transitions)


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

            current_token_name, span = current_token.type, current_token.span
            if len(rep) == 0 or current_token_name == rep[0]:
                used_rule, used_replacement = rule, rep
                break

    raise SyntaxError(f"unexpected symbol sequence '{tokens[2].span}'", "Formula", tokens[2])

    if used_replacement is None:
        raise SyntaxError(f"unexpected symbol sequence '{current_token.span}'", "Formula", current_token)

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


def semantic_analysis(tree):
    if isinstance(tree, str):
        pass
        # print("\t" * depth + tree)
    else:
        pass
        # root, children = tree
        # print("\t" * depth + root)
        # for c in children:
        #     print_tree(c, depth + 1)

    return tree


def main(input_file):
    FOL, formula = read_file(input_file)

    # ordered by priority, these machines run in parallel and simulate one massive machine
    # for example I could put the operators into the identifier machine but to save time I separated them
    token_machines = [
        build_simple_match_token_machine("(", "OPEN_BRACKET"),
        build_simple_match_token_machine(")", "CLOSE_BRACKET"),
        build_simple_match_token_machine(",", "COMMA"),
        build_match_one_of_token_machine(["AND", "OR", "+"], "BI_IN_OP"),
        build_match_one_of_token_machine(["NOT"], "UN_PRE_OP"),
        build_repeated_symbol_token_machine(ascii_lowercase + ascii_uppercase + digits + "_", "IDENTIFIER")
    ]

    # TODO write a function to combine all these machines into one

    replacement_rules = [
        ("BIN_OP", [
            ["AND"],
            ["OR"],
            ["+"]
        ]),

        ("EXPRESSION", [
            ["OPEN_BRACKET", "EXPRESSION", "BIN_OP", "EXPRESSION", "CLOSE_BRACKET"],
            ["UN_PRE_OP", "EXPRESSION"],
            ["IDENTIFIER", "PRED_BODY"],
            ["IDENTIFIER"],
        ]),

        ("PRED_BODY", [
            ["OPEN_BRACKET", "EXPRESSION", "PRED_BODY+"],
        ]),

        ("PRED_BODY+", [
            ["CLOSE_BRACKET"],
            ["COMMA", "EXPRESSION", "PRED_BODY+"],
            []
        ])

    ]
    terminals = {"IDENTIFIER", "OPEN_BRACKET", "CLOSE_BRACKET", "COMMA"}
    string = '"(\n	(\n		function_name(aVariable + 2, bigvariable, constant, D) \n		AND\n		NOT function_2name(aVariable, bigvariable, constant, D)\n	) \n	OR\n	anotherone(aVariable, bigvariable, constant, D) \n)"'

    # token_list = tokenise(formula, token_machines)
    token_list = tokenise(string, token_machines)

    try:
        _, tree = recursive_descent(replacement_rules, terminals, token_list, 0, "E")
        print_tree(tree)

    except SyntaxError as syntax_error:
        print(syntax_error.get_message(string), file=sys.stderr)
        exit(1)

    operation_tree = semantic_analysis(tree)

    print_tree(operation_tree)


def print_tree(tree, depth=0):
    if isinstance(tree, str):
        print("\t" * depth + tree)
    else:
        root, children = tree
        print("\t" * depth + root)
        for c in children:
            print_tree(c, depth + 1)


if __name__ == '__main__':
    input_file = "./example_whitespace.txt"
    main(input_file)
