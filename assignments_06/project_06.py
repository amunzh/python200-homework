from dotenv import load_dotenv
import os
import string
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from pathlib import Path

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path('groundwork_docs')
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

#Part 2
docs = SimpleDirectoryReader('groundwork_docs').load_data()
print(f'Number of documents loaded: {len(docs)}\n')
print('Loaded files:')
for doc in docs:
    print(doc.metadata.get('file_name', 'Unknown file name'))

#Part 3
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)
print('Index built successfully. Ready to answer questions.')

#Part 4
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    top_node = response.source_nodes[0]
    print(f"Document name: {top_node.node.metadata.get('file_name', 'Unknown file')}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text Snippet: {top_node.node.get_content()[:200]}...")
    print("-" * 30)
# The answers to all 5 questions were very dry and just answered a question. No answers surprised me, but in some cases, the answers should have provided further information. Like for the last question, the model should've also provided options for ordering catering.
# Part 5

questions = ["Is there a drive through?",]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Document name: {node_with_score.node.metadata.get('file_name', 'Unknown file')}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)
# I have asked a question about a drive-through because it wasn't mentioned in any of the documents.
# The model doesn't have enough information about it, and it clearly stated that in the answer(Answer was: 'There is no mention of a drive-through in the provided context information.' )
# The answer was still confident and dry. It clearly stated that it cannot provide any kind of information about drive-through, and in this case, we can trust AI-generated responses because, instead of giving wrong information or providing a vague general answer, it clearly states that it cannot help with that. 
# Because this program would be used by customers, the AI tone should be a little bit softer but still preserve correct information. For example, instead of unfriendly stating 'There is no mention of a drive-through in the provided context information.' It can instead say, 'Unfortunately, I do not have enough information about the drive-through. Would you like to ask anything else?' Would sound more friendly

#Part 6
# 1. It only took a couple of lines of code to build everything in this project. It saves cost and time of building everything from scratch; instead, it helps focus on the query and answers itself 
# 2. It can be useful in store settings. For example, documents could contain items and prices, working hours, return policy, delivery, and so on. Creating a model that could answer customers' questions about products or policy would help relieve the customer service center. 
# 3. Some content described in the documents could not be fully understood by the model. In this case, it could lead to misinformation.