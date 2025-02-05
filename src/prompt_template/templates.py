from langchain.prompts import ChatPromptTemplate
from parser import format_instructions

# email prompt
email_template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style} and the {language} language.
text: ```{text}```
"""
email_prompt_template = ChatPromptTemplate.from_template(email_template_string)

def get_custom_email_message(
    *, # 要求后面的参数必须强制使用关键字传递
    style: str,
    language: str,
    text: str,
):
    return email_prompt_template.format_messages(
        style=style,
        language=language,
        text=text,
    )

# JSON prompt
review_template = """\
For the following text, extract the following information:
1. gift
2. delivery_days
3. price_value

{format_instructions}

NOTE: DO NOT INCLUDE MARKDOWN

text: {text}
"""
review_prompt_template = ChatPromptTemplate.from_template(review_template)

def get_review_prompt_template(
    *,
    text: str
):
    return review_prompt_template.format_messages(
        text=text,
        format_instructions=format_instructions
    )