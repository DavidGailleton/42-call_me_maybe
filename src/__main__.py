def main() -> None:
    from src import parsing

    try:
        parsing()
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()
