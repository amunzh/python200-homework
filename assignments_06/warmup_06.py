from dotenv import load_dotenv
import os
import string
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---
# Concepts Q1
# Scenario A: RAG would be best because it can retrieve a large amount of information that constantly changes, and each team member can write a prompt how they prefer. 
# Scenario A: fine-tuning is best here because the startup requires a lot of information to be stored and teach the model the specific brand voice they want to use 
# Scenario C: prompt engineering would be best, she's only using it once for only one document(everything else for this task would be cost-inefficient)

# Concepts Q2
# Because when an AI appears confident in its answer, the person using it will really double-check the information. If AI clearly states that "I am not sure"(or something similar), the user will have doubt about the answer and will double-check it somewhere else. 
# For example if the person asks AI on what they can/cannot eat or which vitamins/medications they should take and model answers without any warning/labeling that it is uncertain information can lead to big problems with health 
# We used to separate different texts by tone, for example, in posts online or in personal texts, we use a more informal tone, which indicates more emotional information, while a more formal tone indicates professionalism (thought more trustworthy in terms of information). So if the AI speaks in formal terms, it seems more respectful and trustworthy, even though the information there could be wrong

# Concepts Q3
#steps = [
#    "Extract text from source documents", taking text from the documents stored
#    "Split text into chunks", preparing text by spliting them into different chunks
#    "Convert text chunks into embeddings", creating embeds for each chunk
#    "Receive the user's query", taking user's query
#    "Embed the user's query",  converts query to models embeds
#    "Retrieve the most relevant chunks", takes most relevent chunks stored
#    "Inject retrieved chunks into the prompt", takes this chunks and puts them into the prompt
#    "Generate a response from the LLM", generating model's response based on the prompt and users chunks
#]


# --- Keyword RAG ---
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1
query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}
simple_keyword_retrieval(query, documents)
# The loyalty.txt document was selected because only the word 'your' matched in any of those documents, and the hiring.txt was the first one to match.
# Keyword Q2
query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents)
# No document was selected; the words selected didn't have any overlap with the documents available. The best in this case would be Semantic RAG

# Keyword Q3
# None because there isn't any overlaping words in the prompt and the documents
query = "How do I sign up for rewards?"
simple_keyword_retrieval(query, documents)
# my prediction was correct because this simple function doesn't capture the relation between words line rewards and loyalty program


# --- Semantic RAG Concepts ---
# Semantic Q1
# 1. Vector embedding converts information stored as words/symbols/images to numeric vectors, which captures the meaning of words. 
# 2. Chunk #1 is more relevant to the query because the cosine score is higher. The score tells how close the meaning of the query and the text are 
# 3. It happens because the information is stored as vectors(numbers), not as words. And if by meaning 2 text are related, the cosine similarity score will be higher even if there are no exact matching words

# Semantic Q2
#| Feature                    | Keyword RAG                       | Semantic RAG                                |
#|----------------------------|-----------------------------------|---------------------------------------------|
#| What is compared?          | Exact word overlap                | Embendings of the query and document chunks |
#| What is retrieved?         | Full document                     | Chunks of the document                      |
#| Can it handle synonyms?    | No                                | Yes                                         |
#| Storage format             | Plain text dictionary             | Numeric vectors                             |
#| Relevance score            | Number of overlapping keywords    | Cosine similarity score                     |


# --- LlamaIndex ---

# LlamaIndex Q1
from llama_index.readers.file import PyMuPDFReader
docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
# The chunks retrieved (especially with a higher similarity score) are related to the question in both cases 
# The model's response is very confident. The tone seems strict and professional. 
# The unexpected part is that the overview documents for both questions have high similarity scores. Yes, it does include information that could be considered benefits and security policy, but without running the model, I wouldn't say they have a high relation.

# LlamaIndex Q2
question = ["What employee benefits does BrightLeaf offer?"]
print(f"\nQ: {question}")
print('--- Similarity Top k = 1 ---')
query_engine2 = index.as_query_engine(similarity_top_k=1)
for q in question:
    response = query_engine2.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)

print('--- Similarity Top k = 5 ---')
query_engine3 = index.as_query_engine(similarity_top_k=5)
for q in question:
    response = query_engine3.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)
# The responses didn't change much, important information related to the question is present in both cases. With more retrive content there's more general information that is unnecessary.

# LlamaIndex Q3
question_new = ["What does the company care about most?"]
for q in question_new:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print("-" * 30)
# I asked a question that was related to the company, but it wasn't stored in any documents. So the answer was really vague without any specifications.
# If the model doesn't have enough information to answer the question, it can immediately ask the user to rephrase the question or state that it doesn't have enough information to answer the query

# LlamaIndex Q4
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Unrelated question
q ="How to make apple pie?"
response = query_engine.query(q)
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Unrelated Faithfulness Evaluation: " + str(faithfulness_result.score))
relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Unrelated Relevancy Result: " + str(relevancy_result.score))

# 1. A faithfulness score of 1.0 means that information provided by the model was retrieved from the documents. While 0.0 indicates that the answer could not be supported by the documents stored in RAG 
# 2. The relevancy score indicates if the model answered the question the user asked, while the faithfulness score shows if the information was retrieved from the documents, no matter if it answered(relevancy) the query or not 
# 3. The scores changed because I had asked an unrelated question, which caused a change in both scores.
# 4. "LLM-as-a-judge" uses other LLMs to compute the evaluation matrix. It is used in RAG evaluation because the answer is usually open-ended, and using other LLMs as a 'judge' to check faithfulness, accuracy, and so on of the result is more useful.