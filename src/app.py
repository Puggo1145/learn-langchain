from dotenv import load_dotenv
from examples import (
    # semantic_search_example,
    # classification_example,
    # extract_person_information_example,
    # extract_people_information_example,
    # chatbot_with_context_example,
    agent_example,
    # rag_one_example,
    rag_two_example
)

load_dotenv(override=True)


def main() -> None:
    # semantic_search_example()
    # classification_example()
    # agent_example()
    rag_two_example()
    return


if __name__ == "__main__":
    main()
