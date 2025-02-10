from dotenv import load_dotenv
from examples.semantic_search import semantic_search_example

load_dotenv(override=True)

def main() -> None:
    semantic_search_example()
    return

if __name__ == "__main__":
    main()
