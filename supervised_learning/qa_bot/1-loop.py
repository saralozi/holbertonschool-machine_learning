#!/usr/bin/env python3
"""Interactive question-answering loop."""


def main():
    """Runs a simple Q&A loop until the user exits."""
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        user_input = input("Q: ")

        if user_input.lower() in exit_words:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == "__main__":
    main()
