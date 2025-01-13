import json
import pprint
import traceback
from urllib import response

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from rich import print as ppprint
from langchain_core.output_parsers import JsonOutputParser


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
json_parser = JsonOutputParser()
format_instructions = json_parser.get_format_instructions()

def generate_hw01(question):
    pass
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
