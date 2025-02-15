from dotenv import load_dotenv
import asyncio
from examples import (
    # semantic_search_example,
    # classification_example,
    # extract_person_information_example,
    # extract_people_information_example,
    # chatbot_with_context_example,
    agent_example,
    # rag_one_example,
    # rag_two_example,
    summary_example
)

load_dotenv(override=True)


async def main() -> None:
    # semantic_search_example()
    # classification_example()
    # agent_example()
    await summary_example()
    return


if __name__ == "__main__":
    # 使用 asyncio.run() 来运行异步主函数
    asyncio.run(main())
