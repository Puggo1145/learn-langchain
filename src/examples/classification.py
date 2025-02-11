import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# types
from typing import cast

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = getpass.getpass("Enter your API key for OpenAI: ")


# schema
class Classification(BaseModel):
    sentiment: str = Field(
        description="The sentiment of the text",
        json_schema_extra={"options": ["happy", "neutral", "sad"]},
    )
    aggressiveness: int = Field(
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        json_schema_extra={"options": [1, 2, 3, 4, 5]},
    )
    language: str = Field(
        description="The language the text is written in",
        json_schema_extra={"options": ["Chinese", "spanish", "english", "french", "german", "italian"]},
    )
    # reason: str = Field(
    #     description="provide a simple explaination of your answer. The explaination should be less than 100 words",
    # )


# function
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the Classification function.

Passage:
{input}
"""
)

llm = ChatOpenAI(
    temperature=0, 
    model="gpt-4o-mini",
).with_structured_output(
    Classification
)

EXAMPLE_TEXT = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
def classification_example(example_text=EXAMPLE_TEXT):
    prompt = tagging_prompt.invoke({"input": example_text})
    llm_response = llm.invoke(prompt)
    transformed_response = cast(Classification, llm_response)

    print(transformed_response.model_dump())
