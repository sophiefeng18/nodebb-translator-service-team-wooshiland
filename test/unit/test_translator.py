from src.translator import translate_content


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

### From ChatGPT
from mock import patch
from openai import AzureOpenAI
import os

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key= os.getenv('API_KEY'),
    api_version="2024-02-15-preview",
    azure_endpoint="https://4project.openai.azure.com/"
)

@patch.object(client.chat.completions, 'create')
def test_unexpected_language(mocker):
  # we mock the model's response to return a random message
  mocker.return_value.choices[0].message.content = "I don't understand your request"

  # TODO assert the expected behavior -- not checking if translation is actually correct
  assert translate_content("Hier ist dein erstes Beispiel.") == (False, "I don't understand your request")

@patch.object(client.chat.completions, 'create')
def test_unexpected_int(mocker):
  # we mock the model's response to return non string
  mocker.return_value.choices[0].message.content = 42

  # TODO assert the expected behavior -- not checking if translation is actually correct
  assert translate_content("Hier ist dein erstes Beispiel.") == (True, "")

@patch.object(client.chat.completions, 'create')
def test_unexpected_bool(mocker):
  # we mock the model's response to return non string
  mocker.return_value.choices[0].message.content = False

  # TODO assert the expected behavior -- not checking if translation is actually correct
  assert translate_content("Hier ist dein erstes Beispiel.") == (True, "")

@patch.object(client.chat.completions, 'create')
def test_unexpected_float(mocker):
  # we mock the model's response to return non string
  mocker.return_value.choices[0].message.content = 4.2

  # TODO assert the expected behavior -- not checking if translation is actually correct
  assert translate_content("Hier ist dein erstes Beispiel.") == (True, "")

@patch.object(client.chat.completions, 'create')
def test_unexpected_char(mocker):
  # we mock the model's response to return non string
  mocker.return_value.choices[0].message.content = 'c'

  # TODO assert the expected behavior -- not checking if translation is actually correct
  assert translate_content("Hier ist dein erstes Beispiel.") == (False, "c")
