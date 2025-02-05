from langchain.output_parsers import ResponseSchema, StructuredOutputParser

response_schemas = [
    ResponseSchema(
        name="gift",
        description="Was the item purchased as a gift for someone else?"
    ),
    ResponseSchema(
        name="delivery_days",
        description="How many days did it take for the product to deliver?"
    ),
    ResponseSchema(
        name="price_value",
        description="Extract any sentences about the value or price"
    )
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
