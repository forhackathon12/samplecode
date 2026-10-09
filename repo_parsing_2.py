import os
import ast
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama
from pathlib import Path

"""
To convert your single-file ingestion pipeline into a robust repository ingestion tool, you need to handle three main changes:

Directory Traversal: Recursively discover Python files while excluding non-code directories (like .git, __pycache__, or .venv).

Context-Aware Metadata: Attach file paths and node types to each chunk so vectors retain their context when queried.

Optimized Insertions: Batch database operations and AST parsing to prevent memory bottlenecks on large codebases.

Refactored Repository Ingestion Pipeline
Here is the complete updated function using standard Python libraries (pathlib) alongside your existing database and AST logic:

"""

def ingest_repository(repo_path, batch_size=50):
    """
    Recursively processes a Python repository, generates AST-based chunks,
    creates embeddings via Ollama, and inserts them into PostgreSQL/pgvector.
    """
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="documind",
        user="documind_user",
        password="documind@123"
    )
    cur = conn.cursor()
    register_vector(conn)

    # Directories to skip during traversal
    IGNORED_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "node_modules", ".pytest_cache"}

    repo_dir = Path(repo_path)
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise ValueError(f"Invalid repository path: {repo_path}")

    data_to_insert = []
    total_chunks = 0

    print(f"Starting ingestion for repository: {repo_dir.resolve()}\n")

    # 1. Walk through repository to find Python files
    for root, dirs, files in os.walk(repo_dir):
        # In-place directory filtering
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = Path(root) / file
            relative_path = str(file_path.relative_to(repo_dir))

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Skipping file {relative_path} due to read error: {e}")
                continue

            try:
                tree = ast.parse(source)
            except SyntaxError:
                print(f"Skipping invalid Python syntax in: {relative_path}")
                continue

            lines = source.splitlines()

            # 2. Extract AST Nodes (Functions and Classes)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    chunk_text = "\n".join(lines[start_line:end_line])

                    if not chunk_text.strip():
                        continue

                    # 3. Generate Embedding via Ollama
                    try:
                        response = ollama.embeddings(
                            model='mxbai-embed-large',
                            prompt=chunk_text
                        )
                        embedding = response['embedding']
                    except Exception as e:
                        print(f"Error generating embedding for {relative_path}:{node.name}: {e}")
                        continue

                    data_to_insert.append((relative_path, node.name, chunk_text, embedding))
                    total_chunks += 1

                    # 4. Batch DB Inserts to manage memory
                    if len(data_to_insert) >= batch_size:
                        cur.executemany(
                            "INSERT INTO code (file_path, node_name, content, embedding) VALUES (%s, %s, %s, %s);",
                            data_to_insert
                        )
                        conn.commit()
                        print(f"Flushed {len(data_to_insert)} chunks to database...")
                        data_to_insert.clear()

    # Insert remaining records
    if data_to_insert:
        cur.executemany(
            "INSERT INTO code (file_path, node_name, content, embedding) VALUES (%s, %s, %s, %s);",
            data_to_insert
        )
        conn.commit()
        print(f"Flushed final {len(data_to_insert)} chunks to database.")

    cur.close()
    conn.close()

    print(f"\nIngestion Complete! Processed {total_chunks} code chunks across repository.")
    return total_chunks


"""
Key Improvements Made
Database Schema Update Recommendation:
To make search queries practical across a full repository, update your PostgreSQL table schema to store metadata like file paths and function names:

SQL
CREATE TABLE IF NOT EXISTS code (
    id SERIAL PRIMARY KEY,
    file_path TEXT,
    node_name TEXT,
    content TEXT,
    embedding vector(1024)
);
Graceful Syntax & Encoding Error Handling:
Repositories often contain third-party scripts, invalid syntax variants, or unexpected encodings. The try/except blocks prevent a single broken .py file from crashing the entire process.

Memory Batching:
Instead of appending every chunk into an in-memory list before doing a single bulk write, the script flushes records to PostgreSQL in configurable batches (batch_size=50).

Async & Class Handlers:
Added ast.AsyncFunctionDef alongside ast.FunctionDef to capture modern async code blocks.
"""