# Title: String to numeric
# Author: javier http://stackoverflow.com/users/11649/javier
# Source: http://stackoverflow.com/a/379966
# License: MIT


def to_numeric(s):
    """Convert string to int or float

    Args:
        s (str): string to be converted to numeric

    Returns:
        int or float
    """
    try:
        return int(s)
    except ValueError:
        return float(s)
