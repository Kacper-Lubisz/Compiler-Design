# from enum import Enum

from typing import Sequence, MutableSet


class Token:

    def __init__(self, type, id, span, line, start_character, end_character):
        self.type = type
        self.id = id
        self.span = span
        self.line = line
        self.start_character = start_character
        self.end_character = end_character

    def __str__(self) -> str:
        return f"(type:{self.type}, id:{self.id}, span:{self.span}, line:{self.line}, start_character:{self.start_character}, end_character:{self.end_character})"

    def __repr__(self):
        return self.__str__()


class FSMDefinition:
    def __init__(self, states: MutableSet[str], start_state, transitions):
        self.states = states
        self.start_state = start_state
        self.transitions = transitions


class FSMInstance:

    def __init__(self, definition):
        self.states = definition.states
        self.transitions = definition.transitions

        self.current_state = definition.start_state

        self.span = ""

    def feed(self, character):
        transitions_out = self.transitions[self.current_state]

        if character not in transitions_out and None not in transitions_out:
            self.current_state = None
            return None

        self.span += character

        next_state_name = transitions_out[character]
        self.current_state = next_state = self.states[next_state_name]

        name, evaluate = next_state

        if evaluate is None:
            return None

        back_step, token_name = evaluate

        return back_step, token_name, self.span

    def reached_end(self):
        return self.current_state is None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.current_state}"


def lexer(string: str, machine_definitions: Sequence[FSMDefinition]):
    token_id = 0
    cursor = 0
    line = 0

    tokens = []

    state_machines = []
    for m_index in range(len(machine_definitions)):
        state_machines.append(FSMInstance(machine_definitions[m_index]))

    length = len(string)
    while cursor != length:
        current_character = string[cursor]

        for m_index in range(len(state_machines)):
            current_machine = state_machines[m_index]

            result = current_machine.feed(current_character)

            if result is not None:
                print("result \t>\t", result)
                state_machines[m_index] = FSMInstance(machine_definitions[m_index])

            if current_machine.reached_end():
                state_machines[m_index] = FSMInstance(machine_definitions[m_index])

        cursor += 1

    # add the EOF
    tokens.append(Token("EOF", token_id, "", line, cursor, cursor))

    return tokens


def build_simple_token_machine(match, token_name):
    states = set()
    start_state = current = ""

    transitions = dict()

    for char in match:
        next_state = current + char

        states.add((current, None))
        transitions[current] = dict()
        transitions[current][char] = next_state

        current = next_state

    states.add((match, None))
    evaluate = (True, token_name)
    states.add(("match", evaluate))  # this state represents the accepting state

    transitions[match] = dict()
    transitions[match][None] = "match"

    return FSMDefinition(states, start_state, transitions)


if __name__ == '__main__':
    test_string = "abs if 2asd hello ( identifier ) ~ and"

    token_machines = [
        build_simple_token_machine("(", "OPEN_BRACKET"),
        build_simple_token_machine(")", "CLOSE_BRACKET"),
        build_simple_token_machine("if", "IF"),
        build_simple_token_machine("~", "TILDA"),
    ]
    rules = ()

    results = lexer(test_string, token_machines)

    print(results)
