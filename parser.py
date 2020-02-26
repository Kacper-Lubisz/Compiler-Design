import sys


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


def main(input_file):
    read_file(input_file)


if __name__ == '__main__':
    input_file = "./example_whitespace.txt"
    main(input_file)
