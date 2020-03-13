#!/usr/bin/env bash
python parser.py examples/example.txt tree_example.png log_example.txt
python parser.py examples/example1.txt tree_example1.png log_example1.txt
python parser.py examples/example_whitespace.txt tree_example_whitespace.png log_example_whitespace.txt
python parser.py examples/example_lexical.txt tree_example_lexical.png log_example_lexical.txt
python parser.py examples/example_invalid_syntax1.txt tree_example_invalid_syntax1.png log_example_invalid_syntax1.txt
python parser.py examples/example_invalid_syntax2.txt tree_example_invalid_syntax2.png log_example_invalid_syntax2.txt
python parser.py examples/example_invalid_syntax3.txt tree_example_invalid_syntax3.png log_example_invalid_syntax3.txt
python parser.py examples/example_invalid_semantic.txt tree_example_invalid_semantic.png log_example_invalid_semantic.txt
