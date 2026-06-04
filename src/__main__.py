def main() -> None:
    """Run the command-line entry point for the project."""
    from src import parsing, process_data
    import sys

    config = parsing(sys.argv)
    if config is None:
        return
    process_data(config)


if __name__ == "__main__":
    main()
