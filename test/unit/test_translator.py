from src.translator import translate_content
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

### Cannot hardcode tests for an LLM that can respond differntly 

#ChatGPT generated
complete_eval_set = [
    {
        "post": "Hier ist dein erstes Beispiel.",
        "expected_answer": (False, "This is your first example.")
    },
    # English posts
    {
        "post": "Hello, how are you today?",
        "expected_answer": (True, "")
    },
    {
        "post": "I love going for walks in the park.",
        "expected_answer": (True, "")
    },
    {
        "post": "The weather is nice and sunny.",
        "expected_answer": (True, "")
    },
    {
        "post": "Can you help me with this task?",
        "expected_answer": (True, "")
    },
    {
        "post": "What time is it?",
        "expected_answer": (True, "")
    },
    {
        "post": "I enjoy reading books in my free time.",
        "expected_answer": (True, "")
    },
    {
        "post": "This is a beautiful painting.",
        "expected_answer": (True, "")
    },
    {
        "post": "How was your weekend?",
        "expected_answer": (True, "")
    },
    {
        "post": "The concert was amazing!",
        "expected_answer": (True, "")
    },
    {
        "post": "Are you coming to the party tonight?",
        "expected_answer": (True, "")
    },
    {
        "post": "I just finished my homework.",
        "expected_answer": (True, "")
    },
    {
        "post": "It's great to see you again.",
        "expected_answer": (True, "")
    },
    {
        "post": "I need a cup of coffee.",
        "expected_answer": (True, "")
    },
    {
        "post": "My favorite color is blue.",
        "expected_answer": (True, "")
    },

    # Non-English posts
    {
        "post": "Hola, ¿cómo estás?",
        "expected_answer": (False, "Hello, how are you?")
    },
    {
        "post": "Ich liebe es, im Park spazieren zu gehen.",
        "expected_answer": (False, "I love to take walks in the park.")
    },
    {
        "post": "どういたしまして。",
        "expected_answer": (False, "You're welcome.")
    },
    {
        "post": "¿Qué hora es?",
        "expected_answer": (False, "What time is it?")
    },
    {
        "post": "J'aime lire des livres.",
        "expected_answer": (False, "I like reading books.")
    },
    {
        "post": "Μου αρέσει να παίζω ποδόσφαιρο.",
        "expected_answer": (False, "I like playing soccer.")
    },
    {
        "post": "Mon chien est très gentil.",
        "expected_answer": (False, "My dog is very friendly.")
    },
    {
        "post": "Você pode me ajudar?",
        "expected_answer": (False, "Can you help me?")
    },
    {
        "post": "Ceci est un test.",
        "expected_answer": (False, "This is a test.")
    },
    {
        "post": "나는 한국어를 배워요.",
        "expected_answer": (False, "I am learning Korean.")
    },
    {
        "post": "هذه فكرة رائعة.",
        "expected_answer": (False, "This is a great idea.")
    },
    {
        "post": "這是一個美麗的城市。",
        "expected_answer": (False, "This is a beautiful city.")
    },
    {
        "post": "Zdravo, kako si?",
        "expected_answer": (False, "Hello, how are you?")
    },
    {
        "post": "Selamat pagi, apa kabar?",
        "expected_answer": (False, "Good morning, how are you?")
    },

    # Unintelligible or malformed posts
    {
        "post": "aslfkdj wqe flkjasd flkj q!@#",
        "expected_answer": (True, "")
    },
    {
        "post": "???@%@!???!  123  234  !@##",
        "expected_answer": (True, "")
    },
    {
        "post": "!!!!!!!???????????",
        "expected_answer": (True, "")
    },
    {
        "post": "xyz&^%$#",
        "expected_answer": (True, "")
    },
    {
        "post": "   *&^@!~`",
        "expected_answer": (True, "")
    }
]
from typing import Callable

def evaluate(query_fn: Callable[[str], str], eval_fn: Callable[[str, str], float], dataset) -> float:
  '''
  TODO: Computes an aggregate score of the chosen evaluation metric across the given dataset. Calls the query_fn function to generate
  LLM outputs for each of the posts in the evaluation dataset, and calls eval_single_response to calculate the metric.
  '''
  # ----------------- YOUR CODE HERE ------------------ #
  #chatgpt
  scores = []

  # Iterate through the dataset and calculate the metric for each example
  for entry in dataset:
      # Extract the post and the expected answer (tuple of bool and translation)
      input_text = entry["post"]
      expected_output = entry["expected_answer"]

      # Get the LLM's response for the input_text
      llm_response = query_fn(input_text)

      # Calculate the evaluation score using the eval_fn
      score = eval_fn(expected_output, llm_response)

      # Store the score
      scores.append(score)

  # Calculate the average score
  avg_score = sum(scores) / len(scores)

  return avg_score
    
def eval_single_response_complete(expected_answer: tuple[bool, str], llm_response: tuple[bool, str]) -> float:
  '''TODO: Compares an LLM response to the expected answer from the evaluation dataset using one of the text comparison metrics.'''
   # Unpack the expected and LLM response tuples
  expected_english, expected_translation = expected_answer
  predicted_english, predicted_translation = llm_response

  # Evaluate the boolean part (English detection)
  english_accuracy = 1.0 if expected_english == predicted_english else 0.0

  # Evaluate the translation part using cosine similarity
  embeddings_expected = model.encode(expected_translation, convert_to_tensor=True)
  embeddings_predicted = model.encode(predicted_translation, convert_to_tensor=True)
  translation_similarity = util.pytorch_cos_sim(embeddings_expected, embeddings_predicted).item()

  # Combine the two scores (could use a weighted average)
  final_score = 0.5 * english_accuracy + 0.5 * translation_similarity

  return final_score
    
def run_eval():
    eval_score = evaluate(translate_content, eval_single_response_complete, complete_eval_set)
    assert eval_score >= .9 #allow some room for variabtion