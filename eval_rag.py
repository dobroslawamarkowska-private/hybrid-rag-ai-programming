"""
Ewaluacja RAG workflow przez LangSmith Client.

Uruchamia pipeline na datasetcie testowym i ocenia wyniki.
Wymaga: LANGCHAIN_API_KEY (LangSmith), utworzony dataset (eval_dataset.py).

Użycie:
  python eval_rag.py                           # dataset "Docker RAG Eval"
  python eval_rag.py --dataset my-eval         # własna nazwa
  python eval_rag.py --llm-judge               # włącza LLM-as-judge (qa_correctness)
  python eval_rag.py --blocking false          # nie czekaj na zakończenie
"""

import argparse

from dotenv import load_dotenv

load_dotenv(override=True)  # przed importem LangChain/LangSmith

from langsmith import Client
from langsmith.evaluation import EvaluationResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import GRADER_LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from workflow import ask


class EvalScore(BaseModel):
    """Structured output dla LLM-as-judge."""
    score: float = Field(description="Ocena 0.0–1.0: jak dobrze odpowiedź pokrywa oczekiwaną (1.0 = w pełni poprawna).")
    comment: str = Field(default="", description="Krótkie uzasadnienie oceny.")


QA_CORRECTNESS_PROMPT = """Oceń, na ile odpowiedź asystenta jest poprawna względem oczekiwanej.

Pytanie użytkownika: {question}

Oczekiwana (wzorcowa) odpowiedź: {expected}

Odpowiedź asystenta: {actual}

Podaj score od 0.0 do 1.0 (1.0 = odpowiedź w pełni poprawna/pokrywa oczekiwaną, 0.0 = całkowicie błędna lub nie na temat)."""


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


def _get_eval_llm():
    """LLM do ewaluacji LLM-as-judge."""
    return ChatOpenAI(
        model=GRADER_LLM_MODEL,
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def qa_correctness(run, example) -> EvaluationResult:
    """
    Evaluator LLM-as-judge: ocenia zgodność odpowiedzi z oczekiwaną (0.0–1.0).
    Działa tylko dla przykładów z expected_answer. Bez niego – pomija (score 1.0).
    """
    expected = (example.outputs or {}).get("expected_answer", "").strip()
    actual = (run.outputs.get("answer") or "").strip()
    question = (example.inputs or {}).get("query", "")

    if not expected:
        return EvaluationResult(key="qa_correctness", score=1.0, comment="Skipped (no expected_answer)")

    if not question or (not expected and not actual):
        return EvaluationResult(key="qa_correctness", score=0.0, comment="Missing inputs")

    prompt = QA_CORRECTNESS_PROMPT.format(question=question, expected=expected, actual=actual)
    llm = _get_eval_llm()
    grader = llm.with_structured_output(EvalScore)
    result = grader.invoke([{"role": "user", "content": prompt}])
    score = max(0.0, min(1.0, float(result.score)))
    return EvaluationResult(key="qa_correctness", score=score, comment=result.comment or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="Docker RAG Eval", help="Nazwa datasetu")
    parser.add_argument("--prefix", "-p", default="RAG Eval", help="Prefix nazwy eksperymentu")
    parser.add_argument("--llm-judge", action="store_true", help="Włącz LLM-as-judge (qa_correctness) dla przykładów z expected_answer")
    parser.add_argument("--blocking", type=lambda x: x.lower() == "true", default=True, help="Czekaj na zakończenie")
    parser.add_argument("--max-concurrency", type=int, default=2, help="Max równoległych wywołań")
    args = parser.parse_args()

    client = Client()

    evaluators = [answer_not_empty, expected_keywords_present]
    if args.llm_judge:
        evaluators.append(qa_correctness)
        print("LLM-as-judge (qa_correctness): włączony")

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
