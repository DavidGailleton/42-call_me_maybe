def main() -> None:
    from src import parsing, process_data
    import sys

    print("PARSING : \n")
    config = parsing(sys.argv)
    if config is None:
        return
    print("\n\nPROCESS_DATA : \n")
    process_data(config)


if __name__ == "__main__":
    main()
