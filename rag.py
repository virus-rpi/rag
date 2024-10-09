import glob
import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import clickhouse_connect
from transformers import AutoModel

print("Initializing models...")

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username='default',
    password='12345'
)

model = genai.GenerativeModel('gemini-1.5-flash')
chat_session = model.start_chat(history=[])

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

keywords = ["weather"]


def get_passage_embedding(text: str) -> np.array:
    return embedding_model.encode(text, task='retrieval.passage').tolist()

def get_query_embedding(text: str) -> np.array:
    return embedding_model.encode(text, task='retrieval.query').tolist()

def pdf_embeddings(path: str) -> np.array:
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    text = "\n".join([doc.page_content for doc in pages])

    docs = text_splitter.create_documents([text])
    for i, d in enumerate(docs):
        d.metadata = {"doc_id": i}

    return docs

def txt_embeddings(path: str) -> np.array:
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()

    docs = text_splitter.create_documents([text])

    for i, d in enumerate(docs):
        d.metadata = {"doc_id": i}

    return docs

def keywords_embeddings(words: list[str]) -> np.array:
    docs = text_splitter.create_documents(words)

    for i, d in enumerate(docs):
        d.metadata = {"doc_id": i}

    return docs


def load_files(batch_size: int) -> None:
    print("Loading files...")
    docs = []

    for path in glob.glob("./files/*.pdf"):
        docs.extend(pdf_embeddings(path))
    for path in glob.glob("./files/*.txt"):
        docs.extend(txt_embeddings(path))

    print("Generating embeddings...")
    content_list = [doc.page_content for doc in docs]
    embeddings = [get_passage_embedding(content) for content in content_list]

    dataframe = pd.DataFrame({
        'page_content': content_list,
        'embeddings': embeddings
    })

    print("Preparing DB...")
    client.command("""
        DROP TABLE IF EXISTS default.docs
    """)

    client.command("""
        CREATE TABLE default.docs (
            id Int64,
            page_content String,
            embeddings Array(Float32),
            CONSTRAINT check_data_length CHECK length(embeddings) = 1024
        ) ENGINE = MergeTree()
        ORDER BY id
    """)

    num_batches = len(dataframe) // batch_size
    print(f"Adding {num_batches} batches of size {batch_size}")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_data = dataframe[start_idx:end_idx]
        client.insert("default.docs", batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
        print(f"Batch {i+1}/{num_batches} inserted.")

    client.command("""
    ALTER TABLE default.docs
        ADD VECTOR INDEX vector_index embeddings
        TYPE SCANN
    """)

    embeddings = [get_passage_embedding(keyword) for keyword in keywords]
    keywords_df = pd.DataFrame({
        "keywords": keywords,
        "embeddings": embeddings
    })

    client.command("""
            DROP TABLE IF EXISTS default.keywords
        """)

    client.command("""
        CREATE TABLE default.keywords (
            id Int64,
            keywords String,
            embeddings Array(Float32),
            CONSTRAINT check_data_length CHECK length(embeddings) = 1024
        ) ENGINE = MergeTree()
        ORDER BY id
    """)

    client.insert("default.keywords", keywords_df.to_records(index=False).tolist(), column_names=keywords_df.columns.tolist())
    print("Keywords added.")

    client.command("""
    ALTER TABLE default.keywords
        ADD VECTOR INDEX vector_index embeddings
        TYPE SCANN
    """)


def get_weather_data(prompt: str) -> str:
    url_meta_prompt = f"""
    Answer only with a valid request url for an api.open-meteo.com api endpoint. 
    The url should fetch all necessary data to answer the following prompt: "{prompt}"
    Use the parameter "&timezone=auto" to avoid time zone errors. The position should be given to the api as &longitude=... and &latitude=... .
    ONLY CREATE A VALID Open-Meteo API URL! ANSWER ONLY WITH THE URL!
    """
    temp_session = model.start_chat(history=chat_session.history)
    answer = temp_session.send_message(url_meta_prompt).text.strip().lower().replace("\n", "").replace("```", "")
    return str(requests.get(answer).json())

def get_relevant_docs(query_embeddings):
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist 
        FROM default.docs 
        ORDER BY dist 
        LIMIT 3
    """)
    return [row['page_content'] for row in results.named_results()]

def get_relevant_keywords(query_embeddings, threshold=1.6):
    results = client.query(f"""
        SELECT keywords,
        distance(embeddings, {query_embeddings}) as dist 
        FROM default.keywords  
        ORDER BY dist 
        LIMIT 3
    """)
    relevant_keywords = [
        row['keywords'] for row in results.named_results()
        if row['dist'] < threshold
    ]
    return relevant_keywords

def get_relevant_api_data(user_query, query_embeddings) -> str:
    relevant_keywords = get_relevant_keywords(query_embeddings)
    data = ""
    if "weather" in relevant_keywords:
        data += get_weather_data(user_query)
    return data

def make_rag_prompt(query, relevant_passage, relevant_data):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n"
        f"DATA: '{relevant_data}'\n"
        '\n'
    )
    return prompt

def generate_answer(query):
    if not query or query == "":
        return
    query_embeddings = get_query_embedding(query)
    relevant_text = get_relevant_docs(query_embeddings)
    relevant_api_data = get_relevant_api_data(query, query_embeddings)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=text, relevant_data=relevant_api_data)
    return chat_session.send_message(prompt).text

if __name__ == "__main__":
    if input("Reindex information? (y/n): ") == "y": load_files(5)
    initial_prompt = """
    You are a helpful and informative chatbot that answers questions using information from the passage and/or data included below the question.
    Respond in a complete sentence and make sure that your response is easy to understand for everyone.
    Maintain a friendly and conversational tone. If the passage is really irrelevant, feel free to ignore it.
    Do not talk about the provided passage or data and only use it to answer the question!\n\n
    """
    print(generate_answer(initial_prompt + input("> ")))
    while True:
        print(generate_answer(input("> ")))
