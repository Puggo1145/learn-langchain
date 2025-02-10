import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from colorama import Fore, Style
from typing import Optional

_ = load_dotenv(find_dotenv()) # read .env file

BASE_MODEL = "qwen-plus"

model = ChatOpenAI(
    api_key=os.environ['ALI_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model=BASE_MODEL,
)

def completion_debug_mode(
    prompt: str,
):
    print(
        f"""{Fore.BLUE}You are going to call with the following prompt:{Style.RESET_ALL}
        
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
    debug: Optional[bool] = False,
) -> BaseMessage | None:
    if debug:
        agree_to_continue = completion_debug_mode(prompt)
        if agree_to_continue is False:
            return None
    
    messages = [
        HumanMessage(prompt)
    ]
    res = model.invoke(messages)

    return res