#Instructions

## Note

I know that the assignment is quite specific in its wording to ensure that lost errors in the tree exist at the grammar
level (such as the number of arguments).
My implementation has its grammar defined more generically such that this error is raised at a semantic pass.
I hope this isn't an issue. 

## Examples

Example inputs for the parser are provided in `examples/`

The file `examples.sh` (`examples.bat` also provided for windows) automatically runs `parser.py` on the following 
examples:

+ `example.txt` - Provided as part of the assignment
+ `example1.txt` - Provided as part of the assignment
+ `example_whitespace.txt` - shows the robustness to whitespace in the input file
+ `example_invalid_syntax1.txt` - demonstrates an invalid syntax (error message)
+ `example_invalid_syntax2.txt` - demonstrates an invalid syntax (error message)
+ `example_invalid_syntax3.txt` - demonstrates an invalid syntax (error message)
+ `example_invalid_semantic.txt` - demonstrates a formula with invalid semantics and the errors this produces
+ `example_lexical.txt` - demonstrates how operators and identifiers are parsed correctly at the lexical stage
    - quantifier `A`, operator `AND`, identifier `ANDY` 

## Usage

Example usage:

```shell script
$ python3 parser.py input.txt tree.png log.txt
``` 

This parses `input.txt`, writes the grammar into `stdout`, produces `tree.png` and `log.txt`,
which contain the image of the parse tree (if possible) and parse log respectfully.


## Dependencies

The script `parser.py` is compatible with python 3.7 (should be forward compatible).
NetworkX and Matplotlib libraries are required.

Install the current release of `networkx` with `pip`:
```shell script
$ pip install networkx
```

Install the current release of `matplotlib` with `pip`:
```shell script
$ pip install matplotlib
```

A Python 3.7.4 interpreter with the necessary dependencies is available on AppAnywhere on university machines.