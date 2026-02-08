"""
Tworzy dataset testowy dla ewaluacji RAG w LangSmith.

Użycie:
  python eval_dataset.py                    # tworzy dataset "Docker RAG Eval"
  python eval_dataset.py --dataset my-eval  # własna nazwa
"""

import argparse

from dotenv import load_dotenv

load_dotenv(override=True)  # ładuje .env przed Client (LANGSMITH_API_KEY)

from langsmith import Client

DATASET_NAME = "Docker RAG Eval"
DESCRIPTION = "Zestaw pytań o dokumentację Docker – ewaluacja pipeline RAG."

# Pytania testowe + opcjonalne oczekiwane słowa kluczowe (dla evaluatorów)
EXAMPLES = [
    {"query": "How can I persist data in Docker containers?", "expected_keywords": ["volume", "bind", "mount", "data"]},
    {"query": "How to build a Docker image from Dockerfile?", "expected_keywords": ["docker build", "Dockerfile"]},
    {"query": "What is Docker Compose?", "expected_keywords": ["compose", "multi-container", "yml", "yaml"]},
    {"query": "How to run a container in detached mode?", "expected_keywords": ["-d", "detach", "docker run"]},
    {"query": "How to remove unused Docker images?", "expected_keywords": ["docker image prune", "prune", "remove"]},
    {"query": "How to expose a port when running a container?", "expected_keywords": ["-p", "port", "EXPOSE"]},
    {"query": "How to connect containers with Docker network?", "expected_keywords": ["network", "docker network", "connect"]},
    {"query": "What is a Docker volume?", "expected_keywords": ["volume", "persistent", "storage"]},
]


def create_dataset(client: Client, dataset_name: str = DATASET_NAME) -> str:
    """Tworzy dataset w LangSmith. Zwraca nazwę datasetu. Jeśli istnieje – pomija."""
    datasets = list(client.list_datasets(dataset_name=dataset_name))
    if datasets:
        print(f"Dataset '{dataset_name}' już istnieje")
        return dataset_name
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=DESCRIPTION,
    )
    for ex in EXAMPLES:
        client.create_example(
            inputs={"query": ex["query"]},
            outputs={"expected_keywords": ex.get("expected_keywords", [])},
            dataset_id=dataset.id,
        )
    print(f"Dataset '{dataset_name}' utworzony: {len(EXAMPLES)} przykładów")
    return dataset_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default=DATASET_NAME, help="Nazwa datasetu")
    args = parser.parse_args()
    client = Client()
    create_dataset(client, args.dataset)


if __name__ == "__main__":
    main()
