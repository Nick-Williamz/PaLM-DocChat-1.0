import google.generativeai as palm
import textwrap
import numpy as np
import pandas as pd
import os

DOCUMENT_DIR = "./documents/"
HISTORY_FILENAME = 'history.txt'

API_KEY = os.getenv('API_KEY')
palm.configure(api_key=API_KEY)

models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
text_model = models[0]


texts = []
for filename in os.listdir(DOCUMENT_DIR):
    if filename.endswith(".txt"):
      with open(os.path.join(DOCUMENT_DIR, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())

df = pd.DataFrame(texts)
df.columns = ['Text']
df['Embeddings'] = [palm.generate_embeddings(model=text_model, text=text)['embedding'] for text in df['Text']]
df['Embeddings'] = df['Embeddings'].apply(np.array)

# Read history if it exists
history = {}
try:
    with open(HISTORY_FILENAME, 'r') as history_file:
        lines = history_file.readlines()
        for line in lines:
            query, answer = line.strip().split('|', 1)
            history[query] = answer
except FileNotFoundError:
    pass

def find_best_passage(query, dataframe):
     query_embedding = palm.generate_embeddings(model=text_model, text=query)
     dot_products = np.array([np.dot(embed, query_embedding['embedding']) for embed in dataframe['Embeddings']])
     idx = np.argmax(dot_products)

     return dataframe.iloc[idx]['Text']

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""\
  You are a helpful and informative bot that answers questions using text from the Python documentation included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information and any necessary Python code. \
  Remember, you are talking to someone who understands basic Python, so feel free to include technical details as necessary. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER (feel free to include Python code):
  """).format(query=query, relevant_passage=escaped)

  return prompt

text_models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
text_model2 = text_models[0]

while True:
    query = input("Enter your question (or 'quit' to stop, 'clear' to clear history): ")
    if query.lower() == 'quit':
        break
    elif query.lower() == 'clear':
        history = {}
        if os.path.exists(HISTORY_FILENAME):
            os.remove(HISTORY_FILENAME)
        print("History cleared.")
        continue

    passage = find_best_passage(query, df)
    prompt = make_prompt(query, passage)

    temperature = 0.5
    answer = palm.generate_text(
    prompt=prompt,
    model=text_model2,
    candidate_count=3,
    temperature=temperature,
    max_output_tokens=1000
  )
    answer = answer.candidates[0]['output']
    history[query] = answer
    with open(HISTORY_FILENAME, 'a') as history_file:
        history_file.write(f"Question: {query}\nAnswer: {answer}\n\n")
        print(f"\nAnswer to '{query}': {answer}\n")

