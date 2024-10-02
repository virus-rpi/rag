import glob

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import clickhouse_connect

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


def get_embeddings(text: str) -> np.array:
    embedding = genai.embed_content(model='models/embedding-001',
                                    content=text,
                                    task_type="retrieval_document")
    return embedding['embedding']

def pdf_embeddings(path: str) -> np.array:
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    text = "\n".join([doc.page_content for doc in pages])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    for i, d in enumerate(docs):
        d.metadata = {"doc_id": i}

    return docs

def txt_embeddings(path: str) -> np.array:
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.create_documents([text])

    for i, d in enumerate(docs):
        d.metadata = {"doc_id": i}

    return docs

def load_files(batch_size: int) -> None:
    docs = []

    for path in glob.glob("./files/*.pdf"):
        docs.extend(pdf_embeddings(path))
    for path in glob.glob("./files/*.txt"):
        docs.extend(txt_embeddings(path))

    content_list = [doc.page_content for doc in docs]
    embeddings = [get_embeddings(content) for content in content_list]

    dataframe = pd.DataFrame({
        'page_content': content_list,
        'embeddings': embeddings
    })

    client.command("""
        DROP TABLE IF EXISTS default.test
    """)

    client.command("""
        CREATE TABLE default.test (
            id Int64,
            page_content String,
            embeddings Array(Float32),
            CONSTRAINT check_data_length CHECK length(embeddings) = 768
        ) ENGINE = MergeTree()
        ORDER BY id
    """)

    num_batches = len(dataframe) // batch_size
    print(f"Adding {num_batches} batches of size {batch_size}")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_data = dataframe[start_idx:end_idx]
        client.insert("default.test", batch_data.to_records(index=False).tolist(), column_names=batch_data.columns.tolist())
        print(f"Batch {i+1}/{num_batches} inserted.")

    client.command("""
    ALTER TABLE default.test
        ADD VECTOR INDEX vector_index embeddings
        TYPE SCANN
    """)

def get_relevant_docs(user_query):
    query_embeddings = get_embeddings(user_query)
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist FROM default.test ORDER BY dist LIMIT 3
    """)
    relevant_docs = []
    for row in results.named_results():
        relevant_docs.append(row['page_content'])
    return relevant_docs

def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)
    prompt = (
        f"QUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
    )
    return prompt

def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=text)
    return chat_session.send_message(prompt).text

if __name__ == "__main__":
    if input("Reindex information? (y/n): ") == "y": load_files(5)
    initial_prompt = """
    You are a helpful and informative chatbot that answers questions using text from the reference passage included below the question.
    Respond in a complete sentence and make sure that your response is easy to understand for everyone.
    Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n
    """
    print(generate_answer(initial_prompt + input("> ")))
    while True:
        print(generate_answer(input("> ")))
