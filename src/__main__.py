def main() -> None:
    try:
        from src import parsing, process_data
        import sys

        config = parsing(sys.argv)
        if config is None:
            return
        process_data(config)
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()
