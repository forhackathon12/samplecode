import os
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# Initialize Tree-sitter Python parser
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Define Tree-sitter query to extract classes, functions/methods, and imports
PY_QUERY = PY_LANGUAGE.query("""
    (class_definition) @class
    (function_definition) @function
    (import_statement) @import
    (import_from_statement) @import_from
""")

def extract_chunks_and_metadata_treesitter(source_bytes, source_code):
    tree = parser.parse(source_bytes)
    captures = PY_QUERY.captures(tree.root_node)

    chunks = []
    metadata = {
        "functions": [],
        "classes": [],
        "imports": []
    }

    lines = source_code.splitlines()

    for node, capture_name in captures:
        if capture_name == "function":
            # Extract function name node
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                metadata["functions"].append(func_name)

            # Chunking using Tree-sitter start/end line coordinates
            start_line = node.start_point[0]
            end_line = node.end_point[0] + 1
            chunk_text = "\n".join(lines[start_line:end_line])
            chunks.append(chunk_text)

        elif capture_name == "class":
            # Extract class name node
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                metadata["classes"].append(class_name)

            # Chunking
            start_line = node.start_point[0]
            end_line = node.end_point[0] + 1
            chunk_text = "\n".join(lines[start_line:end_line])
            chunks.append(chunk_text)

        elif capture_name in ("import", "import_from"):
            import_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
            metadata["imports"].append(import_text.strip())

    return chunks, metadata


def ingest_repository(repo_path):
    conn = psycopg2.connect(
        host="localhost",
        database="documind",
        user="documind_user",
        password="documind@123"
    )
    cur = conn.cursor()
    register_vector(conn)

    ignored_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', 'node_modules', '.pytest_cache'}
    repo_metadata = {}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith('.')]

        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)

            try:
                with open(file_path, "rb") as f:
                    source_bytes = f.read()
                source_code = source_bytes.decode("utf-8")
            except Exception as e:
                print(f"Error reading file {rel_path}: {e}")
                continue

            # Parse using Tree-sitter
            try:
                chunks, file_metadata = extract_chunks_and_metadata_treesitter(source_bytes, source_code)
            except Exception as e:
                print(f"Error parsing {rel_path} with Tree-sitter: {e}")
                continue

            repo_metadata[rel_path] = file_metadata

            if not chunks:
                continue

            # Generate embeddings and prepare insert list
            data_to_insert = []
            print(f"\n##### Processing File: {rel_path} ({len(chunks)} chunks extracted via Tree-sitter) #####")

            for chunk in chunks:
                try:
                    response = ollama.embeddings(
                        model='mxbai-embed-large',
                        prompt=chunk
                    )
                    data_to_insert.append((rel_path, chunk, response['embedding']))
                except Exception as e:
                    print(f"Error generating embedding for chunk in {rel_path}: {e}")

            # Bulk insert chunks
            if data_to_insert:
                cur.executemany(
                    "INSERT INTO code (file_path, content, embedding) VALUES (%s, %s, %s);",
                    data_to_insert
                )
                conn.commit()

    cur.close()
    conn.close()

    return repo_metadata