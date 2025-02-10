import os, sys
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from prompt_template.templates import get_review_prompt_template
from ai.models import get_completion
from parser.review_parser import output_parser

message = get_review_prompt_template(
    text="This leaf blower is fantastic! I bought it as a gift for my brother, and he absolutely loves it. It took just 3 days to arrive, which was impressive. The price was reasonable for the quality, and I believe it offers great value for money."
)
response = get_completion(
    prompt=str(message[0].content),
    debug=True
)

if response is not None:
    output_dict = output_parser.parse(str(response.content))
    print(output_dict.get('gift'))
    print(output_dict.get('delivery_days'))
    print(output_dict.get('price_value'))
else:
    raise ValueError("LLM does not respond")