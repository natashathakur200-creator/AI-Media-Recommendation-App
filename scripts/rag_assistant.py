import os
import json
import glob
import argparse
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


DOCS_DIR = "data/docs"
INDEX_PATH = "data/index.json"

EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.6-flash"

DEFAULT_TOP_K = 5


def get_client() -> genai.Client:
    """Create and return the Gemini client."""

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env"
        )

    return genai.Client(api_key=api_key)


def load_documents() -> list[dict[str, Any]]:
    """
    Load text documents from data/docs.
    """

    files = glob.glob(
        os.path.join(DOCS_DIR, "*.txt")
    )

    if not files:
        raise RuntimeError(
            f"No .txt documents found in {DOCS_DIR}."
        )

    documents = []

    for file_path in files:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read().strip()

        if not text:
            continue

        documents.append(
            {
                "source": os.path.basename(file_path),
                "text": text,
            }
        )

    if not documents:
        raise RuntimeError(
            "No readable text documents found."
        )

    return documents


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks.
    """

    if chunk_size <= overlap:
        raise ValueError(
            "chunk_size must be greater than overlap."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def extract_metadata(
    source_name: str,
    text: str,
) -> dict[str, Any]:
    """
    Extract basic metadata from the research snapshot.

    Metadata is read from explicit fields in the document.
    """

    metadata = {
        "source": source_name,
        "source_type": None,
        "publisher": None,
        "source_url": None,
        "geographic_scope": None,
        "industry": None,
        "last_updated_on_source": None,
    }

    lines = text.splitlines()

    for line in lines:

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        if key == "Source Type":
            metadata["source_type"] = value

        elif key == "Publisher":
            metadata["publisher"] = value

        elif key == "Source URL":
            metadata["source_url"] = value

        elif key == "Geographic Scope":
            metadata["geographic_scope"] = value

        elif key == "Industry":
            metadata["industry"] = value

        elif key == "Last Updated on Source Page":
            metadata["last_updated_on_source"] = value

    return metadata


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str,
) -> list[list[float]]:
    """
    Generate Gemini embeddings.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def build_index() -> None:
    """
    Build the local RAG index.
    """

    print("===== BUILDING RAG INDEX =====")

    print("Loading documents...")

    documents = load_documents()

    all_chunks = []

    for document in documents:

        source_name = document["source"]
        text = document["text"]

        metadata = extract_metadata(
            source_name,
            text,
        )

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "source": source_name,
                    "chunk_id": index,
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    print(
        f"Created {len(all_chunks)} chunks."
    )

    client = get_client()

    texts = [
        item["text"]
        for item in all_chunks
    ]

    print("Creating embeddings...")

    embeddings = embed_texts(
        client,
        texts,
        task_type="RETRIEVAL_DOCUMENT",
    )

    index_data = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": [],
    }

    for item, embedding in zip(
        all_chunks,
        embeddings,
    ):

        index_data["chunks"].append(
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "embedding": embedding,
            }
        )

    os.makedirs(
        os.path.dirname(INDEX_PATH),
        exist_ok=True,
    )

    with open(
        INDEX_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            index_data,
            file,
            ensure_ascii=False,
        )

    print(
        f"Index saved to {INDEX_PATH}"
    )

    print(
        "===== RAG INDEX COMPLETE ====="
    )


def load_index() -> dict[str, Any]:
    """
    Load the local RAG index.
    """

    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(
            f"Index not found: {INDEX_PATH}. "
            "Run with --build-index first."
        )

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def retrieve_top_k(
    client: genai.Client,
    index_data: dict[str, Any],
    query: str,
    k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks.
    """

    query_embedding = embed_texts(
        client,
        [query],
        task_type="RETRIEVAL_QUERY",
    )[0]

    query_vector = np.array(
        query_embedding,
        dtype=np.float32,
    )

    results = []

    for item in index_data["chunks"]:

        document_vector = np.array(
            item["embedding"],
            dtype=np.float32,
        )

        denominator = (
            np.linalg.norm(query_vector)
            * np.linalg.norm(document_vector)
        )

        if denominator == 0:
            similarity = 0.0

        else:
            similarity = float(
                np.dot(
                    query_vector,
                    document_vector,
                )
                / denominator
            )

        result = {
            **item,
            "score": similarity,
        }

        results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:k]


def answer_with_context(
    client: genai.Client,
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Ask Gemini to answer using retrieved evidence.
    """

    context_parts = []

    for item in retrieved_chunks:

        metadata = item.get(
            "metadata",
            {},
        )

        context_parts.append(
            f"""
SOURCE:
{item['source']}

CHUNK:
{item['chunk_id']}

INDUSTRY:
{metadata.get('industry')}

GEOGRAPHIC SCOPE:
{metadata.get('geographic_scope')}

SOURCE TYPE:
{metadata.get('source_type')}

PUBLISHER:
{metadata.get('publisher')}

SOURCE URL:
{metadata.get('source_url')}

LAST UPDATED ON SOURCE:
{metadata.get('last_updated_on_source')}

CONTENT:
{item['text']}
""".strip()
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    system_prompt = """
You are an evidence-grounded AI research assistant
for an AI Media Recommendation and Planning System.

Answer the user's question using ONLY the provided
retrieved context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information,
   clearly say that the available documents are insufficient.
4. Distinguish source information from reasoning.
5. Do not present estimates as verified facts.
6. Do not guarantee ROI, revenue, leads, or campaign results.
7. Mention the relevant source and chunk when making
   factual claims.
8. Respect the geographic scope of the evidence.
9. Do not assume that an old research snapshot is current.
10. If the source contains a limitation, preserve that limitation.

Retrieved context:

""" + context

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
        ),
    )

    return response.text.strip()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the RAG index.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )

    args = parser.parse_args()

    if args.build_index:

        build_index()

        return

    client = get_client()

    index_data = load_index()

    print(
        "===== DOCUMENT-AWARE AI ASSISTANT ====="
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print(
        "The assistant answers using retrieved document context."
    )

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in {
            "exit",
            "quit",
        }:

            print("Goodbye.")

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        try:

            retrieved = retrieve_top_k(
                client,
                index_data,
                question,
                args.k,
            )

            print(
                "\n===== RETRIEVED CHUNKS ====="
            )

            for item in retrieved:

                metadata = item.get(
                    "metadata",
                    {},
                )

                print(
                    f"- score={item['score']:.3f} "
                    f"| source={item['source']} "
                    f"| chunk={item['chunk_id']} "
                    f"| industry={metadata.get('industry')} "
                    f"| geography={metadata.get('geographic_scope')}"
                )

            answer = answer_with_context(
                client,
                question,
                retrieved,
            )

            print(
                "\n===== ASSISTANT ====="
            )

            print(answer)

        except Exception as error:

            print(
                "\nError:"
            )

            print(str(error))


if __name__ == "__main__":
    main()