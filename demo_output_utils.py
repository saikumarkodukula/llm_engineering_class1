"""
Console formatting helpers for classroom demos.
"""


def print_banner(title: str) -> None:
    """Print a full-width title block for a major demo stage."""
    line = "=" * 88
    print(f"\n{line}")
    print(title.upper().center(88))
    print(line)


def print_section(title: str) -> None:
    """Print a smaller section heading inside a demo."""
    line = "-" * 88
    print(f"\n{line}")
    print(title)
    print(line)


def print_key_value(label: str, value: str) -> None:
    """Print one labeled value so status details are easy to scan on screen."""
    print(f"{label:<24}: {value}")
