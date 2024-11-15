from src.translator import translate_content
from dotenv import load_dotenv
from openai import AzureOpenAI
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint="https://4project.openai.azure.com/"  
)

def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message."

def test_llm_normal_response():
    is_english, translated_content = translate_content("This is an English message")
    assert is_english == True
    assert translated_content == ""

def test_llm_gibberish_response():
    is_english, translated_content = translate_content("xyz&^%$#")
    assert is_english == True
    assert translated_content == ""
