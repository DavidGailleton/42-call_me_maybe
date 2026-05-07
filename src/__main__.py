def main() -> None:
    from src import parsing
    import sys

    try:
        parsing(sys.argv)
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()
