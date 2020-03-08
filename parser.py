import math
import sys
from abc import abstractmethod
from string import ascii_lowercase, ascii_uppercase, digits
from typing import List, Tuple, Dict, Union, TextIO, Any, Set
import networkx as nx
import matplotlib.pyplot as plt


class InvalidInputException(Exception):
    """
    This exception is thrown when the input file is invalid
    """
    pass


def read_file(input_file: str) -> Tuple[
    Tuple[
        List[str],
        List[str],
        Dict[str, int],
        str,
        List[str],
        List[str]
    ],
    str
]:
    """
    This method reads in the file and performs some simple pre-processing and **simple** parsing on the input file.
    It returns the parsed FOL definition and the unchanged formula string
    :param input_file: The path to the input file
    :return: The parsed FOL and formula
    """

    with open(input_file, "r") as file:
        content = file.read()

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
            raise Exception("Couldn't find the string '" + search_item + "' in the input file")

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

    def split_at_whitespace(string: str) -> List[str]:
        replaced_whitespace = string.replace("\n", " ").replace("\t", " ").split(" ")
        return list(filter(lambda item: len(item) != 0, replaced_whitespace))

    variables = split_at_whitespace(search_results["variables"])
    constants = split_at_whitespace(search_results["constants"])
    equality = search_results["equality"].strip()
    connectives = split_at_whitespace(search_results["connectives"])
    quantifiers = split_at_whitespace(search_results["quantifiers"])
    formula = search_results["formula"]

    parsed_predicates = dict()
    whitespace_free_predicates = search_results["predicates"].replace(" ", "").replace("\n", " ").replace("\t", " ")
    for predicate in whitespace_free_predicates.split("]"):  # remove all white space

        if len(predicate) == 0 or predicate == " ":
            continue

        split = predicate.split("[")

        if len(split) != 2:
            raise Exception("Missing name or arity of predicate in the input file")

        name, arity = split
        try:
            arity = int(arity)
        except ValueError:
            raise Exception("One predicate's arity is not an integer")

        if arity < 1:
            raise Exception("One predicate's arity is not positive")

        parsed_predicates[name] = arity
    predicates = parsed_predicates

    logic = (variables, constants, predicates, equality, connectives, quantifiers)

    return logic, formula


class Token:
    """
    This class is used to store the tokens in the parsing process
    """

    def __init__(
            self,
            token_type: str,
            token_id: int,
            span: str,
            line: int,
            absolute_start: int,
            line_start: int,
            location: str
    ):
        self.token_type = token_type
        self.token_id = token_id
        self.span = span
        self.line = line
        self.absolute_start = absolute_start
        self.line_start = line_start
        self.location = location

    def __str__(self) -> str:
        return f"Token(type: '{self.token_type}', id: {self.token_id}, span: '{self.span}', " \
               f"line: {self.line}, absolute_start: {self.absolute_start}, line_start: {self.line_start})"

    def __repr__(self) -> str:
        return self.__str__()

    def get_location_string(self) -> str:
        """
        This method generates the string which describes where the token can be found
        :return: The location string
        """
        return f"{self.location} {self.line}:{self.line_start} (character {self.absolute_start})"


class Operator:
    """
    This class encodes the operation tree which is used for the semantic analysis part of the parser
    """

    def __init__(
            self,
            token: Token,
            operator_type: str,
            operands
    ):
        self.token = token
        self.operator_type = operator_type
        self.operands = operands

    def __str__(self) -> str:
        return f"Operator({self.token.span})"

    def __repr__(self) -> str:
        return self.__str__()


class TokenError(Exception):
    """
    This class represents all the Exceptions which can be raised during the parsing process
    """

    def __init__(
            self,
            title: str,
            message: str,
            token: Token
    ):
        self.title = title
        self.message = message
        self.token = token

    def get_token_location_string(self, context_string: str) -> str:
        """
        This method builds a string which visually shows where in the context_string the token exists
        :param context_string: The string in which the token exists
        :return: The message
        """

        string = ""

        line_start = self.token.absolute_start if self.token.absolute_start < len(context_string) else len(
            context_string) - 1
        line_end = line_start

        while line_start != 0 and context_string[line_start] != '\n':
            line_start -= 1

        if line_start != 0:
            previous_line_start = line_start - 1
            while previous_line_start != 0 and context_string[previous_line_start] != '\n':
                previous_line_start -= 1
            previous_line_start += 1
            line_start += 1
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
            string += f"{justified_previous_number} |{context_string[previous_line_start:line_start - 1]}\n"

        justified_line_number = str(line_number).rjust(max_line_number_length, ' ')

        string += f"{justified_line_number} |{context_string[line_start:line_end]}\n"

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
        if span_length == 0:
            span_length = 1

        string += " " * (len(justified_line_number) + 2 + offset_within_line) + "^" * span_length + "\n"

        if following_line_end is not None:
            justified_following_number = str(line_number + 1).rjust(max_line_number_length, ' ')
            string += f"{justified_following_number} |{context_string[line_end + 1:following_line_end]}\n"

        return string

    @abstractmethod
    def get_message(self, context_string: str) -> str:
        """
        This method builds the user message representing the TokenError
        :param context_string:
        :return:
        """
        context_string += '\n'
        return f"{self.title} in {self.token.get_location_string()}\n" \
               f"{self.message}\n" \
               f"{self.get_token_location_string(context_string)}\n"


class FSM:
    """
    This class models an FSM.
    """

    def __init__(
            self,
            states: Dict[Any, Tuple[bool, str]],
            start_state: Any,
            transitions: Dict[Any, Dict[Any, Any]]
    ):
        self.states = states
        self.transitions = transitions
        self.current_state = self.start_state = start_state

        self.span = ""

    def reset(self) -> None:
        """
        Resets the FSM to the start state
        """
        self.span = ""
        self.current_state = self.start_state

    def feed(self, character: Any) -> Union[None, Tuple[bool, str, str]]:
        """
        Transitions the FSM into a new state, returns the properties associated with that state and the span which lead
        to that state.  If there is no such transition possible then the state is set to None and the machine has
        reached it's end.

        :param character: The value used to determine which transition should occur
        :return: The properties of the new state, null otherwise
        """

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
        """
        :return: if the machine is in a dead end state
        """
        return self.current_state is None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"FSMInstance(state: '{self.current_state}', span:'{self.span}')"


def to_tokens(string: str, machines: List[FSM]) -> List[Token]:
    """
    This method performs the tokenizing part of the compilation process
    :param string: The string to tokenize
    :param machines: The list of FSMs which accept tokens in the string
    :return: A list of tokens produced by the FSMs
    """
    location_description = "Formula"
    string += "\0"

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
                    cursor - line_start - len(span) + 1,
                    location_description
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
    tokens.append(Token("EOF", token_id, "EOF", line, cursor, cursor - line_start, location_description))

    return tokens


def build_repeated_symbol_token_machine(
        matching_characters: List[Any],
        token_name,
        back_step_characters: List[Union[str, None]]
):
    """
    This function builds a FMS which accepts any sequence of the specified matching characters, followed by the
    back_step_character.
    :param matching_characters: A list of characters that can be matched
    :param token_name: The token type of the tokens this machine will produce
    :param back_step_characters: The set of characters which can occur as the character after a match which cause the
     match to be accepted, None meaning any character
    :return: The FSM which accepts the specified string
    """

    start = 0
    matching = 1
    accept = 2

    states = {
        accept: (True, token_name),
        start: None,
        matching: None
    }
    transitions = {
        start: dict(),
        matching: dict()
    }

    for char in back_step_characters:
        transitions[matching][char] = accept

    for char in matching_characters:
        transitions[start][char] = matching
        transitions[matching][char] = matching

    return FSM(states, start, transitions)


def build_simple_match_token_machine(match: str, token_name: str, back_step_characters: List[Union[str, None]]) -> FSM:
    """
    This method builds a FMS which accept the specified match, followed by any specified back_step character
    :param match: The matching string
    :param token_name: The token type of the tokens this machine will produce
    :param back_step_characters: The set of characters which can occur as the character after a match which cause the
     match to be accepted, None meaning any character
    :return: The FSM which accepts the specified string
    """
    start_state = current = ""

    matched = 0

    states = {
        match: None,
        matched: (True, token_name)  # this state is the accepting state
    }

    transitions = {
        match: dict()
    }
    for char in back_step_characters:
        transitions[match][char] = matched

    for char in match:
        next_state = current + char

        states[current] = None
        transitions[current] = {char: next_state}

        current = next_state

    return FSM(states, start_state, transitions)


def build_match_one_of_token_machine(matches: List[str], token_name: str, back_step_characters: List[str]) -> FSM:
    """
    This method builds a FMS which accept any of the specified matches
    :param matches: The list of strings that this machine will match
    :param token_name: The token type of the tokens this machine will produce
    :param back_step_characters: The set of characters which can occur as the character after a match which cause the
     match to be accepted
    :return: The FSM which accepts the specified strings
    """
    matched = 0

    states = {matched: (True, token_name)}  # this state is the accepting state
    transitions = dict()
    for match in matches:
        states[match] = None
        transitions[match] = dict()
        for char in back_step_characters:
            transitions[match][char] = matched

    start_state = ""
    for match in matches:
        current = start_state

        for char in match:
            next_state = current + char

            states[current] = None
            if current in transitions:
                transitions[current][char] = next_state
            else:
                transitions[current] = {char: next_state}

            current = next_state

    return FSM(states, start_state, transitions)


def recursive_descent(
        replacement_rules: List[Tuple[str, List[Tuple[List[str], List[Tuple[int, str]]]]]],
        tokens: List[Token],
        cursor_index: int,
        looking_for: str,
        expected_translator: Dict[str, str]
) -> Tuple[
    int,
    Union[
        Token,
        Tuple[str, List[Any]]
    ]
]:
    """
    Performs recursive descent to build the derivation tree of the string if possible, if not possible then a
    TokenError is raised
    :raise: TokenError if a syntax error exists
    :param replacement_rules: The replacement rules
    :param tokens: The sequence of tokens to be parsed
    :param cursor_index: The index pointing to where parsing should start from
    :param looking_for: The terminal/none-terminal which is to being looked for
    :param expected_translator: A dictionary translating between what is expected and what an error should read
    :return: The syntax tree
    """

    current_token = tokens[cursor_index]

    if current_token.token_type == looking_for:
        # print("\t" * depth + f"found {looking_for}")

        return 1, current_token
    else:

        used_rule, new_sequence = None, None

        for rule in replacement_rules:
            root, replacements = rule

            if used_rule is not None:
                break

            if root is not looking_for:
                continue

            for replacement in replacements:

                if used_rule is not None:
                    break

                sequence, look_ahead = replacement

                if len(look_ahead) == 0:
                    used_rule, new_sequence = rule, sequence
                    break
                else:
                    matched = True
                    for look_ahead in look_ahead:
                        distance, expected = look_ahead

                        if cursor_index + distance >= len(tokens) \
                                or tokens[cursor_index + distance].token_type != expected:
                            matched = False
                            break

                    if matched:
                        used_rule, new_sequence = rule, sequence

        if new_sequence is None:
            expected = expected_translator[looking_for]
            raise TokenError(
                "Syntax Error",
                f"Expected to find a '{expected}' ({looking_for}), instead found '{current_token.span}'",
                current_token
            )

        new_cursor = cursor_index
        children = []
        for looking_for in new_sequence:
            cursor_change, sub_tree = recursive_descent(
                replacement_rules,
                tokens,
                new_cursor,
                looking_for,
                expected_translator
            )
            new_cursor += cursor_change
            children.append(sub_tree)

        index_change = new_cursor - cursor_index

        return index_change, (used_rule[0], children)


def semantic_analysis(
        operation_tree: Union[Token, Operator],
        symbol_scope: Set[str],
        fol: Tuple[List[str], List[str], Dict[str, int], str, List[str], List[str]]) -> List[TokenError]:
    """
    This method analyses the semantics of the formula for errors and returns a list of them
    :param operation_tree:
    :param symbol_scope:
    :param fol:
    :return:
    """

    variables, constants, predicates, equality, connectives, quantifiers = fol

    issues = []
    if isinstance(operation_tree, Operator):

        if operation_tree.operator_type == "QUANTIFIER":
            variable_token, over_operator = operation_tree.operands
            variable_id = variable_token.span

            if variable_id in symbol_scope:
                issues.append(TokenError(
                    "Semantic Error",
                    f"The quantified variable '{variable_id}' can't be bound again",
                    variable_token
                ))

            if variable_id not in variables:
                issues.append(TokenError(
                    "Semantic Error",
                    f"The variable '{variable_id}' is quantified over without declaration in the logic definition",
                    variable_token
                ))

            if variable_id not in symbol_scope:
                symbol_scope.add(variable_id)
                issues += semantic_analysis(over_operator, symbol_scope, fol)
                symbol_scope.remove(variable_id)
            else:
                issues += semantic_analysis(over_operator, symbol_scope, fol)

        elif operation_tree.operator_type == "PREDICATE":

            for argument in operation_tree.operands:
                issues += semantic_analysis(argument, symbol_scope, fol)

            predicate_identifier = operation_tree.token.span
            if predicate_identifier not in predicates:
                issues.append(TokenError(
                    "Semantic Error",
                    f"The predicate '{predicate_identifier}' is used without declaration in the logic definition",
                    operation_tree.token
                ))
            else:  # is declared
                if len(operation_tree.operands) != predicates[predicate_identifier]:
                    issues.append(TokenError(
                        "Semantic Error",
                        f"The predicate '{predicate_identifier}' of arity {predicates[predicate_identifier]} is used"
                        f" with {len(operation_tree.operands)} input",
                        operation_tree.token
                    ))

        elif operation_tree.operator_type == "BIN_OP":
            for operand in operation_tree.operands:
                issues += semantic_analysis(operand, symbol_scope, fol)
        elif operation_tree.operator_type == "UN_OP":  #
            for operand in operation_tree.operands:
                issues += semantic_analysis(operand, symbol_scope, fol)
        else:
            raise Exception("Unexpected Operator encountered")

    else:

        token = operation_tree
        token_id = token.span  # identifier

        in_scope = token_id in symbol_scope
        is_constant = token_id in constants
        is_variable = token_id in variables

        if not is_variable and not is_constant:
            issues.append(TokenError(
                "Semantic Error",
                f"The identifier '{token_id}' doesn't refer to any variable or constant",
                token
            ))
        elif is_variable and not in_scope:
            issues.append(TokenError(
                "Semantic Error",
                f"The variable '{token_id}' is used before it is bound",
                token
            ))

    return issues


def build_operation_tree(syntax_tree: Union[Token, Tuple[str, List[Any]]]) -> Union[Token, Operator, List[Token], None]:
    """
    This method parses the derivation tree to convert it to an operation tree.
    :param syntax_tree: The syntax tree to parse (Any in the type definition means the enter tree type recurred)
    :return: The operation tree equivalent
    """
    if isinstance(syntax_tree, Token):  # terminal
        token = syntax_tree

        removable_tokens = {
            "OPEN_BRACKET",
            "CLOSE_BRACKET",
            "COMMA",
        }

        if token.token_type in removable_tokens:
            return None
        else:
            return token

    else:

        root, children = syntax_tree
        children = [build_operation_tree(child) for child in children]
        children = [child for child in children if child is not None]

        if root == "FORMULA":
            if len(children) == 1:
                # rule (["IDENTIFIER"], [(0, "IDENTIFIER")])
                return children[0]

            elif len(children) == 3:
                left, middle, right = children

                if middle.token_type == "BI_IN_OP":
                    return Operator(middle, "BIN_OP", [left, right])
                else:
                    return Operator(left, "QUANTIFIER", [middle, right])

            else:  # len(children) == 2:
                left, right = children
                if left.token_type == "UN_PRE_OP":
                    return Operator(left, "UN_OP", [right])

                else:
                    return Operator(left, "PREDICATE", right)

        elif root == "BIN_COMB":
            left, middle, right = children

            return Operator(middle, "BIN_OP", [left, right])

        elif root == "PRED_BODY":
            left, right = children
            return [left] + right
        else:  # root == "PRED_BODY+"
            if len(children) == 0:
                return []
            else:
                left, right = children
                return [left] + right


def print_replacement_rules(
        rules: List[
            Tuple[
                str,
                List[
                    Tuple[
                        List[str],
                        List[Tuple[int, str]]
                    ]
                ]
            ]
        ]
) -> None:
    """
    This method prints out the replacement rules in Backus–Naur form
    :param rules: The replacement rules

    The following describes the rule type definition
    List[
        Tuple[
            str, # replace from
            List[  # replace tos
                Tuple[
                    List[str], # new sequence
                    List[Tuple[int, str]] # look ahead rules (look forward by int and see str)
                ]
            ]
        ]
    ]
    """

    for root, replacements in rules:
        print("<" + root + ">", "::=", end=" ")
        for index, (new_sequence, _) in enumerate(replacements):
            if index != 0:
                print(" | ", end="")
            for symbol in new_sequence:
                print("<" + symbol + ">", end="")

        print()


def draw_operation_tree(tree: Union[Operator, Token], tree_file_path: str) -> None:
    """
    This method creates an image of the operation tree passed to it and writes it to a file
    :param tree: The tree to visualise
    :param tree_file_path: The path of the tree image file which will be created
    """
    graph = nx.Graph()

    labels = dict()
    positions = dict()

    def build_operation_graph(
            node: Union[Operator, Token],
            start_x: int,
            depth: int
    ) -> Tuple[int, float]:
        """
        Simple post order traversal which builds the NetworkX operation tree graph.
        :param node: The node of the operation tree which is to be added
        :param start_x: The x coordinate lower bound for the tree which will be produced
        :param depth: The depth in the tree
        :return: The upper bound on the nodes in the produced tree and the x coordinate assigned to node
        """

        if isinstance(node, Operator):
            labels[node] = node.token.span

            sum_x = 0
            end_x = start_x

            for child in node.operands:
                end_x, x = build_operation_graph(child, end_x, depth + 1)

                sum_x += x

                graph.add_edge(node, child)

            x = sum_x / len(node.operands)
            positions[node] = (x, -depth)

            return end_x, x

        else:
            end_x = start_x + 1
            x = (start_x + end_x) / 2

            labels[node] = node.span
            positions[node] = (x, -depth)
            graph.add_node(node)

            return end_x, x

    build_operation_graph(tree, 0, 0)

    nx.draw(graph, positions)
    nx.draw_networkx_labels(graph, positions, labels)
    plt.savefig(tree_file_path)


def main(input_file: str, tree_path: str, log_path: str) -> None:
    log_file: TextIO = open(log_path, "w")
    print(f"Arguments:\n"
          f"Input File: {input_file}\n"
          f"Tree Output File: {tree_path}\n"
          f"Log File: {log_path}\n", file=log_file)

    print("        Reading input file", file=log_file)
    try:
        logic, formula = read_file(input_file)
        print("SUCCESS Reading input file", file=log_file)

    except InvalidInputException as invalid_input:
        print(f"FAIL    Reading input file, invalid input\n{invalid_input}", file=log_file)
        return

    except OSError as read_error:
        print(f"FAIL    Reading input file, file couldn't be read\n{read_error}", file=log_file)
        return

    variables, constants, predicates, equality, connectives, quantifiers = logic

    # ordered by priority, these machines run in parallel and simulate one massive machine
    # for example I could put the operators into the identifier machine but to save time I separated them

    white_space = [" ", "\t", "\n"]

    token_machines = [
        build_simple_match_token_machine("(", "OPEN_BRACKET", [None]),
        build_simple_match_token_machine(")", "CLOSE_BRACKET", [None]),
        build_simple_match_token_machine(",", "COMMA", [None]),
        build_match_one_of_token_machine(quantifiers, "BI_PRE_OP", white_space),
        build_match_one_of_token_machine([equality], "BI_ID_IN_OP", white_space),
        build_match_one_of_token_machine(connectives[:-1], "BI_IN_OP", white_space),
        build_match_one_of_token_machine([connectives[-1]], "UN_PRE_OP", white_space),
        build_repeated_symbol_token_machine(ascii_lowercase + ascii_uppercase + digits + "_", "IDENTIFIER", [None])
    ]  # TODO write a function to combine all these machines into one

    replacement_rules = [  # grammar definition
        ("FORMULA", [
            # in the following line [(0, "OPEN_BRACKET")] defines the lookahead needed to match this replacement
            (["OPEN_BRACKET", "BIN_COMB", "CLOSE_BRACKET"], [(0, "OPEN_BRACKET")]),
            (["UN_PRE_OP", "FORMULA"], [(0, "UN_PRE_OP")]),
            (["IDENTIFIER", "OPEN_BRACKET", "PRED_BODY"], [(1, "OPEN_BRACKET")]),
            (["BI_PRE_OP", "IDENTIFIER", "FORMULA"], [(1, "IDENTIFIER")]),  # quantifier
            (["IDENTIFIER"], [(0, "IDENTIFIER")]),
        ]),

        ("BIN_COMB", [
            (["IDENTIFIER", "BI_ID_IN_OP", "IDENTIFIER"], [(1, "BI_ID_IN_OP")]),
            (["FORMULA", "BI_IN_OP", "FORMULA"], []),
        ]),

        ("PRED_BODY", [
            (["FORMULA", "PRED_BODY+"], []),
        ]),

        ("PRED_BODY+", [
            (["CLOSE_BRACKET"], [(0, "CLOSE_BRACKET")]),
            (["COMMA", "FORMULA", "PRED_BODY+"], [(0, "COMMA")]),
        ])
    ]
    expected_translator = {
        "OPEN_BRACKET": "(",
        "CLOSE_BRACKET": ")",
        "COMMA": ",",
        "BI_PRE_OP": "Binary Prefix Operator (Quantifier)",
        "BI_ID_IN_OP": "Binary Infix Operator on Variables/Constants",  # on identifiers
        "BI_IN_OP": "Binary Infix Operator on Formulas",
        "UN_PRE_OP": "Unary Prefix Operator",
        "IDENTIFIER": "Variable or Constant",
    }
    print_replacement_rules(replacement_rules)

    print("        Tokenizing formula", file=log_file)
    token_list = to_tokens(formula, token_machines)
    print("SUCCESS Tokenizing formula", file=log_file)

    try:
        print("        Parsing formula", file=log_file)
        tokens_read, syntax_tree = recursive_descent(replacement_rules, token_list, 0, "FORMULA", expected_translator)

        if tokens_read + 1 < len(token_list):  # +1 to account for EOF
            raise TokenError("Syntax Error", "Formula continued unexpectedly", token_list[tokens_read])

        print("SUCCESS Parsing formula", file=log_file)

        print("        Building Operation Tree", file=log_file)
        operation_tree = build_operation_tree(syntax_tree)
        print("SUCCESS Building Operation Tree", file=log_file)

        print("        Analysing Semantics", file=log_file)
        issues = semantic_analysis(operation_tree, set(), logic)
        print("SUCCESS Analysing Semantics", file=log_file)

        if len(issues) != 0:
            print(f"        {len(issues)} issues found\n", file=log_file)
            for issues in issues:
                print(issues.get_message(formula), file=log_file)
            print(file=log_file)
        else:
            print("        No issues found", file=log_file)

        print("        Drawing Tree", file=log_file)
        draw_operation_tree(operation_tree, tree_path)
        print("SUCCESS Drawing Tree", file=log_file)

    except TokenError as syntax_error:
        print("FAIL    Parsing formula\n", file=log_file)
        print(syntax_error.get_message(formula), file=log_file)
        return


def print_operation_tree_to_stdout(tree: Union[Token, Operator], depth: int = 0) -> None:
    """
    This method prints an operation tree to the console
    :param tree: The tree to be printed
    :param depth: The current depth
    """

    if isinstance(tree, Operator):
        print("\t" * depth + f"{tree.token.span}")

        for c in tree.operands:
            print_operation_tree_to_stdout(c, depth + 1)

    else:
        print("\t" * depth + str(tree.span))


def print_syntax_tree_to_stdout(tree: Union[Token, Tuple], depth: int = 0) -> None:
    """
    This method prints a syntax tree to the console
    :param tree: The tree to be printed
    :param depth: The current depth
    """
    if isinstance(tree, Token):
        print("\t" * depth + str(tree))
    else:
        root, children = tree
        print("\t" * depth + root)
        for c in children:
            print_syntax_tree_to_stdout(c, depth + 1)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        input_path = sys.argv[1]
        tree_image_path = sys.argv[2]
        log_file_path = sys.argv[3]

        try:
            main(input_path, tree_image_path, log_file_path)

        except Exception as e:
            print(f"Unrecoverable failure", file=sys.stderr)
            print(e)
            raise e
    else:
        print("Invalid command line arguments, see README.md", file=sys.stderr)
