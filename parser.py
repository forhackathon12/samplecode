import json
import ast
import ollama

from chunker import chunk_file, chunk_text
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from openai import OpenAI

import psycopg2
from pgvector.psycopg2 import register_vector


SYSTEM_PROMPT = """
You are a Principal Solution Architect and Technical Writer.

Analyze the supplied source code and generate engineering documentation.

Rules:
1. Only use information visible in the code.
2. Do not invent functionality.
3. If information is unknown, return "Not Available".
4. Focus on architecture, dependencies, business logic, risks, and operations.
5. Return valid JSON only.

Generate:

{
  "application_summary": "",
  "components": [],
  "functions": [],
  "classes": [],
  "dependencies": [],
  "data_flow": [],
  "business_rules": [],
  "exceptions": [],
  "operational_runbook": {
      "startup_steps": [],
      "shutdown_steps": [],
      "monitoring_points": [],
      "troubleshooting_steps": []
  },
  "architecture_description": "",
  "mermaid_diagram": ""
}

Return only valid JSON.
"""


def ingestion_pipeline(file_path):

    # Create PGVector connection
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="documind",
        user="documind_user",
        password="documind@123"
    )
    cur = conn.cursor()

    # Register pgvector type in psycopg2
    register_vector(conn)

    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    chunks = []
    lines = source.splitlines()

    data_to_insert = []

    tree = ast.parse(source)

    metadata = {
        "functions": [],
        "classes": [],
        "imports": []
    }

    for node in ast.walk(tree):
        # start_line = node.lineno - 1
        # end_line = node.end_lineno
        # chunk_text = "\n".join(lines[start_line:end_line])
        # chunks.append(chunk_text)

        if isinstance(node, ast.FunctionDef):
            metadata["functions"].append(node.name)

            #chunking
            start_line = node.lineno - 1
            end_line = node.end_lineno
            chunk_text = "\n".join(lines[start_line:end_line])
            chunks.append(chunk_text)

        elif isinstance(node, ast.ClassDef):
            metadata["classes"].append(node.name)
            #chunking
            start_line = node.lineno - 1
            end_line = node.end_lineno
            chunk_text = "\n".join(lines[start_line:end_line])
            chunks.append(chunk_text)

        elif isinstance(node, ast.Import):
            for imp in node.names:
                metadata["imports"].append(imp.name)


    # chunking - AST
    print("\n\n ##### AST BASED CHUNKING ##### \n")
    for i, chunk in enumerate(chunks):
        print(f"CHUNK {i+1} = ")
        print(chunk)

        response = ollama.embeddings(
            model='mxbai-embed-large',
            prompt=chunk
        )
        print(f"\n\n CHUNK {i+1} EMBEDDING = ")
        print(response['embedding'])

        data_to_insert.append((chunk, response['embedding']))   
    
    # Bulk Insert the data directly as a Python list
    cur.executemany(
        "INSERT INTO code (content, embedding) VALUES (%s, %s);",
        data_to_insert
    )

    conn.commit()
    cur.close()
    conn.close()
    
    return source, metadata

#  End-to-End RAG Pipeline
def rag_pipeline(question):

    print("\nGenerating query embedding...")
    query_embedding = generate_query_embedding(question)

    print("\nRetrieving chunks...")
    retrieved_chunks = retrieve_chunks(
        query_embedding,
        top_k=5
    )

    show_retrieved_chunks(retrieved_chunks)

    # Optional
    retrieved_chunks = rerank_chunks(
        question,
        retrieved_chunks
    )

    print("\nBuilding context...")
    context = build_context(retrieved_chunks)

    print("\nGenerating answer...")
    result = generate_response_with_citations(
        question,
        retrieved_chunks
    )

    return result

def generate_query_embedding(question: str):
    """
    Generate embedding for user question
    """

    response = ollama.embeddings(
        model="mxbai-embed-large",
        prompt=question
    )

    return response["embedding"]


def retrieve_chunks(question_embedding, top_k=5):

    conn = psycopg2.connect(
        host="localhost",
        database="documind",
        user="documind_user",
        password="documind@123"
    )

    register_vector(conn)

    cur = conn.cursor()

    query = """
    SELECT
        id,
        content,
        embedding <=> %s::vector AS distance
    FROM code
    ORDER BY distance
    LIMIT %s;
    """

    cur.execute(query, (question_embedding, top_k))

    results = cur.fetchall()

    cur.close()
    conn.close()

    return results


# Display Top-K Retrieved Chunks
def show_retrieved_chunks(results):

    print("\n===== TOP K CHUNKS =====\n")

    for row in results:
        chunk_id = row[0]
        content = row[1]
        score = row[2]

        print(f"Chunk ID : {chunk_id}")
        print(f"Similarity Score : {score}")
        print(content)
        print("-" * 80)


# Optional Re-ranking
def rerank_chunks(question, retrieved_chunks):

    prompt = f"""
    User Question:
    {question}

    Rank the following chunks from most relevant to least relevant.

    Return only the chunk numbers.

    Chunks:

    """

    for idx, row in enumerate(retrieved_chunks):
        prompt += f"\nChunk {idx+1}:\n{row[1]}\n"

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role":"user","content":prompt}
        ]
    )

    print(response["message"]["content"])

    return retrieved_chunks

# Context Builder
def build_context(retrieved_chunks):

    contexts = []

    for row in retrieved_chunks:
        chunk_id = row[0]
        content = row[1]

        contexts.append(
            f"""
            [CITATION:{chunk_id}]
            {content}
            """
        )

    return "\n\n".join(contexts)


# Build RAG Prompt
def build_rag_prompt(question, context):

    prompt = f"""
    You are a Principal Solution Architect.

    Answer ONLY using the supplied context.

    If the answer is not available in the context,
    reply with:

    "I could not find the answer in the retrieved documents."

    Context:
    {context}

    Question:
    {question}

    Provide:
    1. Detailed answer
    2. Architectural explanation
    3. Relevant citations
    """
    return prompt


# Generate Final Response
def generate_rag_response(question, context):

    prompt = build_rag_prompt(question, context)

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response["message"]["content"]


# Citation Extraction

def generate_response_with_citations(question, chunks):

    context = build_context(chunks)

    answer = generate_rag_response(
        question,
        context
    )

    citations = []

    for row in chunks:
        citations.append({
            "chunk_id": row[0]
        })

    return {
        "answer": answer,
        "citations": citations
    }


# Example Usage
if __name__ == "__main__":

    source, metadata = ingestion_pipeline("./input/discount.py")

    question = """
    Explain how the discounts are applied and provide the 
    flow of the code.
    """

    result = rag_pipeline(question)

    print("\n========== ANSWER ==========\n")
    print(result["answer"])

    print("\n========== CITATIONS ==========\n")
    print(result["citations"])

    print("Documentation generated")