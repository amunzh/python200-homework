# --- Completions API ---
# API Q1
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)
print(f'Response:{response.choices[0].message.content}\nModel:{response.model}\nNumber of tokens used:{response.usage.total_tokens}')

# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response_new = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f'Temperature:{temp}\nResponse:{response_new.choices[0].message.content}\n')
# The output for temperatures 0 only consist of one suggestion while output when temp = 0.7,1.5 gave 10 different examples where when temp = 1.5 it gave more diverse suggestions. 
# The most consistent and reproducible output will be when using temp = 0.0 because the model picks most likely token while 0.7 and 1.5 allowes randomness 

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)
for i, choice in enumerate(response.choices): 
    print(f'Response {i + 1}: {response.choices[0].message.content}\n')

# API Q4
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)
print(f'Result max tokens:{response.choices[0].message.content}')
# The response got cut off. In real application max_tokens could be used if we need a short answer because it will reduce the length of the responce making it more cost efficient. 


# --- System Messages and Personas ---
# System Q1
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response = client.chat.completions.create(model='gpt-4o-mini', messages=messages)
print(response.choices[0].message.content)

messages_new = [
    {"role": "system", "content": "You are a Python tutor that gives short explanations without encouragement"},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]
response_smp = client.chat.completions.create(model='gpt-4o-mini', messages=messages_new)
print(response_smp.choices[0].message.content)
# In the first case, the system used emojis and added a lot of encouraging words, while the second was straight to the point without any additional sentences. Just explained the list comprehension and gave an example

# System Q2
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]
response = client.chat.completions.create(model='gpt-4o-mini', messages=messages)
print(f'Stateless Response: {response.choices[0].message.content}')
# It can 'remember' Jordan's name because this information was in the message sent to the API


# --- Prompt Engineering ---
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content
# Prompt Q1
reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]
prompt = f'Classify the sentiment of each review(numerate them) below as positive, negative, or mixed: {reviews}'
response = get_completion(prompt, temperature=0)
print(f'Zero-Shot:\n{response}')

# Prompt Q2
prompt = f'Classify the sentiment of each review(numerate them) below as positive, negative, or mixed: {reviews}.\nExample: \nReview: Fast shipping but the item arrived damaged.\nSentiment: mixed'
response = get_completion(prompt, temperature=0)
print(f'One-Shot:\n{response}')
# It didn't change the result but the format in which the answer was given changed to follow the exact format that was given in an example case.

# Prompt Q3
prompt = f'Classify the sentiment of each review(numerate them) below as positive, negative, or mixed: {reviews}.\nExamples:\nReview: Fast shipping but the item arrived damaged.\nSentiment: mixed\nReview: Great item and came fast.\nSentiment: positive.\nReview: System is impossible to navigate and it constantly crashes.\nSentiment: negative'
response = get_completion(prompt, temperature=0)
print(f'Few-Shot:\n{response}')
# The result itself didn't change in all 3 cases. The model adapts the response to the format that was given in an exaples. In the case where we need to classify to 3 different categories few-shot doesn't work well, but it would deffinately be more helpfull in case when the model need to sort in much more categories and much more prompts.
# The one-shot would be helpfull if the model sorts correctly but you need a certain format for the responce. And zero-shot is more simple and is goot for simple requests

# Prompt Q4
prompt = 'Solve the following problem, but show your reasoning step by step before giving a final answer. Also, label the final answer clearly. Problem: A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?'
response = get_completion(prompt, temperature=0)
print(f'Chain of Thought:\n{response}')
# The step by step reasoning firstly allow user to see how program came up with the solution and in case it made a mistake see where exactly it happened. Also CoT helps increase the accuracy of the results. 

# Prompt Q5
import json
review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
prompt = f"""Analyze the customer review and return only the raw JSON object. Return keys: sentiment, confidence (0–1), reason(one sentence). Also, do not wrap the JSON in markdown code fences and do not use ```json.
Review: {review}
"""
response = get_completion(prompt, temperature=0)
print(f'Raw response:\n{response}')

try:
    res_parsed = json.loads(response)
    print(f'Sentiment: {res_parsed["sentiment"]}\nConfidence: {res_parsed["confidence"]}\nReason: {res_parsed["reason"]}')
except json.JSONDecodeError:
    print(f'The response was not valid JSON. Raw response:{response}')

# Prompt Q6
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = get_completion(prompt, temperature=0)
print(f'Delimiters:\n{response}')

user_text_new = "Let's cook pasta, it will be really good."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text_new}```
"""
response = get_completion(prompt, temperature=0)
print(f'Delimiters no instructions:\n{response}')
# The delimiters help model to separate the prompt information and the data provided not mixing everything together.


# --- Local Models with Ollama ---
# Ollama Q1
prompt ='Explain what a large language model is in two sentences.'
response = get_completion(prompt, temperature=0)
print(f'Ollama OpenAI API:\n{response}')
# ollama
# A large language model is an AI model designed to understand and generate human-like text, 
# trained on vast amounts of data to learn from conversations, syntax, and context. It can perform 
# tasks like answering questions, writing articles, or even creative writing, showcasing its 
# ability to process complex, coherent information.

# The first sentence in both responces are the same but the second sentence in Ollama give examples of different tasks it can preform while OpenAI explains how it achives the results
# Local models give more data privacy, but it is less resource efficient because requires poferfull hardware.