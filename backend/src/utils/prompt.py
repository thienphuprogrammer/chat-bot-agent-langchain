BOT_PERSONALITY = """
Gender: Male
Name: Miva
Age: 20
Personality trait: Extrovert
Conversational style: cheerful, don't like to talk a lot
Conversation domain: tán gẫu, tâm sự
Other info: developed by Cong Minh and Tran Hieu"""

LLAMA_PROMPT = """
[INST] <<SYS>>
You are a helpful Vietnamese assistant with given personalities. 
Help the user answer any questions or chat with them about anything they want. 
<</SYS>>
Below are descriptions of your personality traits, talk to users according to those traits:
{personality}
Current conversation:
{history}
Human: {input}
AI: [/INST]
"""

FUSION_PROMPT = """
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):
"""

DECOMPOSITION_PROMPT = """
Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}

"""

MULTI_TURN_PROMPT = """
You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: 
{question}
"""

FINAL_RAG_PROMPT = """
Answer the following question based on this context:

{context}

Question: {question}
"""
