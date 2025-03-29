from __future__ import annotations

import contextlib
import os
from typing import Any


class hushed:
    """
    A context manager that suppresses stdout and stderr for the code block.

    Example:
    -------
    with hushed():
        # Code here will not print anything to stdout or stderr
        print("This won't be seen.")
    """

    def __enter__(self):
        self._devnull = open(os.devnull, "w")  # Open /dev/null for writing
        self._stdout_redirector = contextlib.redirect_stdout(self._devnull)
        self._stderr_redirector = contextlib.redirect_stderr(self._devnull)
        
        # Enter redirection contexts
        self._stdout_redirector.__enter__()
        self._stderr_redirector.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        # Exit redirection contexts
        self._stderr_redirector.__exit__(exc_type, exc_value, traceback)
        self._stdout_redirector.__exit__(exc_type, exc_value, traceback)
        self._devnull.close()  # Ensure devnull file is closed properly


def to_tuple(not_tuple: list | Any) -> tuple:
    """
    Recursively converts a list (or nested lists) to a tuple. 

    This function is designed to handle lists that may contain nested lists, 
    and it will recursively convert all levels of lists into tuples.

    Parameters
    ----------
    not_tuple
        The input to be converted. If the input is a list, it will be 
        recursively converted to a tuple. Otherwise, it will be returned as-is.

    Returns
    -------
    A tuple equivalent of the input list, or the original element if it is not a
    list.

    Example
    -------
    >>> totuple([1, 2, [3, 4, [5, 6]], 7])
    (1, 2, (3, 4, (5, 6)), 7)

    >>> totuple('string')
    'string'
    """
    try:
        return tuple(to_tuple(i) for i in not_tuple)
    except TypeError:
        return not_tuple


def uniform_repr(
    object_name: str,
    *positional_args: Any,
    max_width: int = 60,
    stringify: bool = True,
    indent_size: int = 2,
    **keyword_args: Any,
) -> str:
    """
    Generates a uniform string representation of an object, supporting both
    positional and keyword arguments.
    """

    def format_value(value: Any) -> str:
        """
        Converts a value to a string, optionally wrapping strings in quotes.
        """
        if isinstance(value, str) and stringify:
            return f'"{value}"'
        return str(value)

    # Format positional and keyword arguments
    components = [format_value(arg) for arg in positional_args]
    components += [
        f"{key}={format_value(value)}" 
        for key, value in keyword_args.items()
    ]

    # Construct a single-line representation
    single_line_repr = f"{object_name}({', '.join(components)})"
    if len(single_line_repr) < max_width and "\n" not in single_line_repr:
        return single_line_repr

    # If exceeding max width, format as a multi-line representation.
    def indent(text: str) -> str:
        """Indents text with a specified number of spaces."""
        indentation = " " * indent_size
        return "\n".join(f"{indentation}{line}" for line in text.split("\n"))

    # Build multi-line representation
    multi_line_repr = f"{object_name}(\n"
    multi_line_repr += ",\n".join(indent(component) for component in components)
    multi_line_repr += "\n)"

    return multi_line_repr


def min_count(x: list[int]) -> tuple[int, int]:
    """
    Count the number of times the minimum value appears in the list `X`.
    """
    x = list(x)

    if not x:
        raise ValueError("The list X cannot be empty")
        
    min_value = min(x)
    return min_value, x.count(min_value)