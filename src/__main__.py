def main() -> None:
    from src import parsing, process_data
    import sys

    config = parsing(sys.argv)
    if config is None:
        return
    process_data(config)


if __name__ == "__main__":
    main()
