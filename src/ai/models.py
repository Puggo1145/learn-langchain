import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from colorama import Fore, Style
from typing import Iterable, Optional
from openai.types.chat import ChatCompletionUserMessageParam

_ = load_dotenv(find_dotenv()) # read .env file

BASE_MODEL = "deepseek-v3"

client = OpenAI(
    api_key=os.environ['ALI_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def completion_debug_mode(
    model: str,
    prompt: str,
):
    print(
        f"""{Fore.BLUE}You are going to call{Style.RESET_ALL} {model} {Fore.BLUE}with the following prompt:{Style.RESET_ALL}
        
{prompt}
{Fore.BLUE}Are you sure you want to continue? [yes/no]{Style.RESET_ALL} """
    )
    
    user_input = input().lower().strip()
    if user_input not in ['y', 'yes']:
        print('Operation Canceld.')
        return False
    
def get_completion(
    *,
    prompt: str, 
    model: str = BASE_MODEL,
    debug: Optional[bool] = False,
) -> str | None:
    if debug:
        agree_to_continue = completion_debug_mode(model, prompt)
        if agree_to_continue is False:
            return None
    
    messages: Iterable[ChatCompletionUserMessageParam] = [
        { "role": "user", "content": prompt  }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content