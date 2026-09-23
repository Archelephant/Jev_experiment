"""Evaluate a document with the TypeSafe adapter over GigaChat.

Run inside the protected perimeter with the GigaChat credentials and TLS
certificates configured via environment variables (see README.md):

    export GIGACHAT_CREDENTIALS='<client_id:secret>'
    export GIGACHAT_CA_BUNDLE_FILE=/path/to/root-ca.crt
    export GIGACHAT_CERT_FILE=/path/to/client.crt
    export GIGACHAT_KEY_FILE=/path/to/client.key
    export GIGACHAT_MODEL='GigaChat-2'

    python demo_gigachat.py --mode structured
    python demo_gigachat.py --mode prompted
"""

from __future__ import annotations

import argparse
import asyncio
import json

from gigachat_provider import GigaChatAsyncProvider, GigaChatSyncProvider
from system_one_adapter import AsyncSystemOneAdapterClient, SystemOneAdapterClient
from typesafe_sdk import Choice, Noul, RetryPolicy, Score

STATE = {
    "contract": (
        "Договор оказания услуг между ООО «СервисПро» и ООО «Клиент».\n"
        "1. Исполнитель обрабатывает персональные данные Заказчика в соответствии "
        "с ФЗ-152 «О персональных данных» исключительно в целях исполнения договора.\n"
        "2. Все споры разрешаются в Арбитражном суде города Москвы.\n"
        "3. Применимое право — законодательство Российской Федерации.\n"
        "4. Срок действия договора — 12 месяцев с возможностью пролонгации."
    ),
}

QUESTIONS = {
    "has_gdpr_clause": Noul(
        type="noul",
        instructions="Содержит ли договор пункт об обработке персональных данных?",
        criteria={"true": "Договор упоминает персональные данные", "false": "Договор не упоминает их"},
    ),
    "clarity": Score(
        type="score",
        instructions="Оцените ясность и полноту договора.",
        criteria=["Неясно и неполно", "Частично ясно", "Ясно и полно"],
    ),
    "jurisdiction": Choice(
        type="choice",
        instructions="Какой суд указан для разрешения споров?",
        criteria={
            "Москва": "Арбитражный суд города Москвы",
            "Санкт-Петербург": "Арбитражный суд Санкт-Петербурга",
            "Не указан": "Суд не упоминается",
        },
    ),
}


def _print_answer(label: str, answer: object) -> None:
    print(f"\n[{label}]")
    print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))


def _print_response(response: object) -> None:
    usage = response.usage
    print("\n=== Model ===", response.model)
    print("=== Usage ===", usage.model_dump(mode="json", exclude_none=True))
    print("=== Retries ===", usage.n_retries, "transient,", usage.n_retries_malformed_structure, "malformed")
    for question_id, answer in response.answers.items():
        _print_answer(question_id, answer)


def run_sync(*, structured: bool) -> None:
    provider = GigaChatSyncProvider()
    client = SystemOneAdapterClient(
        model=provider,
        structured_outputs=structured,
        llm_answer_mode="discrete",
        retry=RetryPolicy(max_retries=2),
    )
    with client:
        response = client.system_one(STATE, QUESTIONS)
    _print_response(response)


async def run_async(*, structured: bool) -> None:
    provider = GigaChatAsyncProvider()
    client = AsyncSystemOneAdapterClient(
        model=provider,
        structured_outputs=structured,
        llm_answer_mode="discrete",
        retry=RetryPolicy(max_retries=2),
    )
    async with client:
        response = await client.system_one(STATE, QUESTIONS)
    _print_response(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("structured", "prompted"),
        default="structured",
        help="structured: native GigaChat json_schema output; prompted: schema in instructions",
    )
    parser.add_argument("--async", dest="use_async", action="store_true", help="use the async provider")
    args = parser.parse_args()
    structured = args.mode == "structured"
    if args.use_async:
        asyncio.run(run_async(structured=structured))
    else:
        run_sync(structured=structured)


if __name__ == "__main__":
    main()