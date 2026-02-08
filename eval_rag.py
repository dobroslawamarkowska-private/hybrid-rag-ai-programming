"""
Ewaluacja RAG workflow przez LangSmith Client.

Uruchamia pipeline na datasetcie testowym i ocenia wyniki.
Wymaga: LANGCHAIN_API_KEY (LangSmith), utworzony dataset (eval_dataset.py).

Użycie:
  python eval_rag.py                           # dataset "Docker RAG Eval"
  python eval_rag.py --dataset my-eval         # własna nazwa
  python eval_rag.py --blocking false          # nie czekaj na zakończenie
"""

import argparse

from dotenv import load_dotenv

load_dotenv(override=True)  # przed importem LangChain/LangSmith

from langsmith import Client
from langsmith.evaluation import EvaluationResult

# Import po load_dotenv
from workflow import ask


def predict(inputs: dict) -> dict:
    """Target dla client.evaluate – wywołuje RAG i zwraca output."""
    query = inputs.get("query", "")
    answer = ask(query)
    return {"answer": answer}


def answer_not_empty(run, example) -> EvaluationResult:
    """Evaluator: odpowiedź nie jest pusta."""
    answer = run.outputs.get("answer", "") or ""
    score = 1.0 if answer.strip() else 0.0
    return EvaluationResult(key="answer_not_empty", score=score, comment="OK" if score else "Empty answer")


def expected_keywords_present(run, example) -> EvaluationResult:
    """Evaluator: odpowiedź zawiera oczekiwane słowa kluczowe (co najmniej jedno)."""
    answer = (run.outputs.get("answer", "") or "").lower()
    expected = example.outputs.get("expected_keywords") or []
    if not expected:
        return EvaluationResult(key="expected_keywords", score=1.0, comment="No keywords to check")
    found = sum(1 for kw in expected if kw.lower() in answer)
    score = min(1.0, found / max(1, len(expected)))
    return EvaluationResult(
        key="expected_keywords",
        score=score,
        comment=f"Found {found}/{len(expected)} keywords",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="Docker RAG Eval", help="Nazwa datasetu")
    parser.add_argument("--prefix", "-p", default="RAG Eval", help="Prefix nazwy eksperymentu")
    parser.add_argument("--blocking", type=lambda x: x.lower() == "true", default=True, help="Czekaj na zakończenie")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Max równoległych wywołań")
    args = parser.parse_args()

    client = Client()

    evaluators = [answer_not_empty, expected_keywords_present]

    print(f"Uruchamianie ewaluacji na datasetcie '{args.dataset}'...")
    results = client.evaluate(
        predict,
        data=args.dataset,
        evaluators=evaluators,
        experiment_prefix=args.prefix,
        description="Ewaluacja RAG workflow – Docker docs",
        metadata={"workflow": "orchestrator-workers"},
        max_concurrency=args.max_concurrency,
        blocking=args.blocking,
    )

    if args.blocking:
        print(f"\nEksperyment: {results.experiment_name}")
        print("Wyniki w LangSmith:", "https://smith.langchain.com")

    return results


if __name__ == "__main__":
    main()
