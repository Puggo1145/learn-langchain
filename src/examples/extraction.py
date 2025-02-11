from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
import os
import getpass
from langchain_openai import ChatOpenAI

# types
from typing import Optional, List, cast

# llm
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# schema
class Person(BaseModel):
    """Information about a person"""

    name: Optional[str] = Field(
        default=None,
        description="The name of the person",
    )
    hair_color: Optional[str] = Field(
        default=None,
        description="The color of the person's hair if known",
        json_schema_extra={"rules": ["Must be a common color name"]},
    )
    height_in_meters: Optional[str] = Field(
        default=None,
        description="Height of the person measured in meters",
        json_schema_extra={"rules": ["Must be a numerical value"]},
    )


class People(BaseModel):
    """Extracted data about people."""

    people: List[Person]


# extractor
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)


EXAMPLE_TEXT_PERSON = "Alan Smith is 6 feet tall and has blond hair."


def extract_person_information_example(example_text=EXAMPLE_TEXT_PERSON):
    # llm supporting structured output
    structured_llm_for_person = llm.with_structured_output(schema=Person)

    prompt = prompt_template.invoke({"text": example_text})
    response = structured_llm_for_person.invoke(prompt)
    casted_response = cast(Person, response)

    print(casted_response.model_dump())


EXAMPLE_TEXT_PEOPLE = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me, and Andrew is 180cm tall"


def extract_people_information_example(example_text=EXAMPLE_TEXT_PEOPLE):
    # llm supporting structured output
    structured_llm_for_people = llm.with_structured_output(schema=People)

    prompt = prompt_template.invoke({"text": example_text})
    response = structured_llm_for_people.invoke(prompt)
    casted_response = cast(Person, response)

    print(casted_response.model_dump())
