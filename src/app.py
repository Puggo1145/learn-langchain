# from prompt_template import get_custom_email_message
from prompt_template import get_review_prompt_template
from ai.models import get_completion
from parser import output_parser

# message = get_custom_email_message(
#     style="soft and professional",
#     language="Chinese",
#     text="I wanna make a well-known AI product that is loved by many people"
# )
message = get_review_prompt_template(
    text="This leaf blower is fantastic! I bought it as a gift for my brother, and he absolutely loves it. It took just 3 days to arrive, which was impressive. The price was reasonable for the quality, and I believe it offers great value for money."
)
response = get_completion(
    model="qwen-plus",
    prompt=str(message[0].content),
    debug=True
)

if response is not None:
    output_dict = output_parser.parse(response)
    print(output_dict.get('gift'))
    print(output_dict.get('delivery_days'))
    print(output_dict.get('price_value'))
else:
    raise ValueError("LLM does not respond")