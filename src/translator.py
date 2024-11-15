from openai import AzureOpenAI
import os

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key= os.getenv('API_KEY'),
    api_version="2024-02-15-preview",
    azure_endpoint="https://4project.openai.azure.com/"
)
import openai
def get_translation(post: str) -> str:
    context = "Translate the following text into English, only respond with the exact translation, only put punctuation exactly as it is in the origional text, and no other words:"
    prompt = f"{context}\n\n{post}"
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    translated_text = response.choices[0].message.content
    return translated_text.strip()


def get_language(post: str) -> str:
    context = "Identify the primary language of the following text, only provide the langauge, as it is known in english, and no other text (if you are not able to identify a clear language only reposnd with the word English):"
    prompt = f"{context}\n\n{post}"
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    language = response.choices[0].message.content
    return language.strip()

#ChatGPT assisted

def translate_content(post: str) -> tuple[bool, str]:
  try:

    language = get_language(post)

    if not isinstance(language, str):
      raise ValueError("Language detection should return a string.")

    english = language == 'English'

    if english:
      translation = ""
    else:
      translation = get_translation(post)

    if not isinstance(translation, str):
          raise ValueError("Translation should be a string.")


    # Return the result as a tuple of (bool, str)
    result = (english, translation)
    return result

  except Exception as e:
    # If there was an error (invalid format or anything else), handle gracefully
    # Log the error for debugging (optional, depends on your logging system)
    print(f"Error in query_llm_robust: {e}")

    # Return a default safe response (assume English and no translation)
    return (True, "")  # Default: assume it's in English with no translation needed













# def old_translate_content(content: str) -> tuple[bool, str]:
#     if content == "这是一条中文消息":
#         return False, "This is a Chinese message"
#     if content == "Ceci est un message en français":
#         return False, "This is a French message"
#     if content == "Esta es un mensaje en español":
#         return False, "This is a Spanish message"
#     if content == "Esta é uma mensagem em português":
#         return False, "This is a Portuguese message"
#     if content  == "これは日本語のメッセージです":
#         return False, "This is a Japanese message"
#     if content == "이것은 한국어 메시지입니다":
#         return False, "This is a Korean message"
#     if content == "Dies ist eine Nachricht auf Deutsch":
#         return False, "This is a German message"
#     if content == "Questo è un messaggio in italiano":
#         return False, "This is an Italian message"
#     if content == "Это сообщение на русском":
#         return False, "This is a Russian message"
#     if content == "هذه رسالة باللغة العربية":
#         return False, "This is an Arabic message"
#     if content == "यह हिंदी में संदेश है":
#         return False, "This is a Hindi message"
#     if content == "นี่คือข้อความภาษาไทย":
#         return False, "This is a Thai message"
#     if content == "Bu bir Türkçe mesajdır":
#         return False, "This is a Turkish message"
#     if content == "Đây là một tin nhắn bằng tiếng Việt":
#         return False, "This is a Vietnamese message"
#     if content == "Esto es un mensaje en catalán":
#         return False, "This is a Catalan message"
#     if content == "This is an English message":
#         return True, "This is an English message"
#     return True, content
