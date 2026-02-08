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
DESCRIPTION = "Zestaw pytań o dokumentację Docker – ewaluacja pipeline RAG (keywords + opcjonalnie expected_answer dla LLM-as-judge)."

# Pytania testowe + opcjonalne oczekiwane słowa kluczowe i pełna odpowiedź (dla LLM-as-judge)
# expected_answer – opcjonalna wzorcowa odpowiedź; jeśli ustawiona, evaluator qa_correctness ją wykorzysta
EXAMPLES = [
    {"query": "How can I persist data in Docker containers?", "expected_keywords": ["volume", "bind", "mount", "data"], "expected_answer": "Use Docker volumes or bind mounts to persist data. Volumes are managed by Docker; bind mounts map a host directory."},
    {"query": "How to build a Docker image from Dockerfile?", "expected_keywords": ["docker build", "Dockerfile"], "expected_answer": "Use docker build to build an image from a Dockerfile. Example: docker build -t myimage ."},
    {"query": "What is Docker Compose?", "expected_keywords": ["compose", "multi-container", "yml", "yaml"], "expected_answer": "Docker Compose is a tool for defining and running multi-container Docker applications using a YAML file."},
    {"query": "How to run a container in detached mode?", "expected_keywords": ["-d", "detach", "docker run"], "expected_answer": "Use docker run -d to run a container in detached (background) mode."},
    {"query": "How to remove unused Docker images?", "expected_keywords": ["docker image prune", "prune", "remove"], "expected_answer": "Use docker image prune -a to remove all unused images, or docker image prune for dangling images only."},
    {"query": "How to expose a port when running a container?", "expected_keywords": ["-p", "port", "EXPOSE"], "expected_answer": "Use docker run -p host_port:container_port to publish a port. In Dockerfile, use EXPOSE to document the port."},
    {"query": "How to connect containers with Docker network?", "expected_keywords": ["network", "docker network", "connect"], "expected_answer": "Create a network with docker network create, then use docker run --network or docker network connect to attach containers."},
    {"query": "What is a Docker volume?", "expected_keywords": ["volume", "persistent", "storage"], "expected_answer": "A Docker volume is persistent storage managed by Docker, used to persist data outside container lifecycle."},
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
        outputs = {"expected_keywords": ex.get("expected_keywords", [])}
        if ex.get("expected_answer"):
            outputs["expected_answer"] = ex["expected_answer"]
        client.create_example(
            inputs={"query": ex["query"]},
            outputs=outputs,
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
