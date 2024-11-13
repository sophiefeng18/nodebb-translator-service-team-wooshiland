from src.translator import translate_content


def test_chinese():
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english == False
    assert translated_content == "This is a Chinese message"

def test_llm_normal_response():
    is_english, translated_content = translate_content("This is an English message")
    assert is_english == True
    assert translated_content == "This is an English message"

def test_french():
    is_english, translated_content = translate_content("This is a French message")
    assert is_english == False
    assert translated_content == "This is a French message"

def test_llm_gibberish_response():
    is_english, translated_content = translate_content("xyz&^%$#")
    assert is_english == True
    assert translated_content == "xyz&^%$#"

### From ChatGPT
from mock import patch

@patch.object(client.chat.completions, 'create')
def test_unexpected_language(mocker):
  # we mock the model's response to return a random message
  mocker.return_value.choices[0].message.content = "I don't understand your request"

  # TODO assert the expected behavior -- not checking if translation is actually correct
  assert translate_content("Hier ist dein erstes Beispiel.") == (True, "I don't understand your request")

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
  assert translate_content("Hier ist dein erstes Beispiel.") == (True, "c")
