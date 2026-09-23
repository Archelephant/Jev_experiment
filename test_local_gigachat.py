"""Локальный тест: доступ к GigaChat по credentials из .env.

Отличие от целевого (защищённого) контура: здесь используется OAuth
``client_credentials`` из ``GIGACHAT_CREDENTIALS`` и публичное api.giga.chat
(без mTLS-сертификатов). Логика TLS повторяет пример asr.lourie.info:
если файл из ``CA_BUNDLE_PATH`` существует - верификация сертификата через него,
иначе ``verify_ssl_certs=False`` (только для локального теста).

Запуск:
    python test_local_gigachat.py                 # prompted-режим, sync
    python test_local_gigachat.py --mode structured
    python test_local_gigachat.py --async
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv

from demo_gigachat import QUESTIONS, STATE
from gigachat_provider import GigaChatAsyncProvider, GigaChatSyncProvider
from system_one_adapter import AsyncSystemOneAdapterClient, SystemOneAdapterClient
from typesafe_sdk import RetryPolicy

DEFAULT_MODEL = "GigaChat-2"


def get_secret(name: str) -> Optional[str]:
    """Вернуть секрет из env или из файла, путь к которому лежит в ``<name>_FILE``.

    Повторяет логику `get_secret` из asr.lourie.info/main.py.
    """
    file_path = os.environ.get(f"{name}_FILE")
    if file_path and os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as handle:
            return handle.read().strip()
    return os.environ.get(name)


def configure_env() -> dict:
    """Загрузить .env и выставить GIGACHAT_* переменные для локального доступа."""
    load_dotenv()

    credentials = get_secret("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise SystemExit("GIGACHAT_CREDENTIALS не найден в .env.")

    model = get_secret("GIGACHAT_MODEL") or DEFAULT_MODEL
    os.environ["GIGACHAT_CREDENTIALS"] = credentials
    os.environ["GIGACHAT_MODEL"] = model

    ca_bundle = get_secret("CA_BUNDLE_PATH")
    if ca_bundle and os.path.exists(ca_bundle):
        os.environ["GIGACHAT_CA_BUNDLE_FILE"] = ca_bundle
        os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "true"
        ca_status = ca_bundle
        verify = True
    else:
        os.environ.pop("GIGACHAT_CA_BUNDLE_FILE", None)
        os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "false"
        ca_status = None
        verify = False

    return {
        "model": model,
        "base_url": os.environ.get("GIGACHAT_BASE_URL") or "https://api.giga.chat/v1",
        "ca_bundle_file": ca_status,
        "verify_ssl_certs": verify,
        "scope": os.environ.get("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS",
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


def check_auth(provider) -> float:
    """Запросить реальный access token, чтобы проверить credentials и сеть."""
    start = time.perf_counter()
    token = provider._client.get_token()
    elapsed = time.perf_counter() - start
    expires = getattr(token, "exp", None)
    print(f"\n[AUTH] token получен за {elapsed:.1f}s, expires_at={expires}")
    return elapsed


async def check_auth_async(provider) -> float:
    """Асинхронный вариант `check_auth`."""
    start = time.perf_counter()
    token = await provider._client.aget_token()
    elapsed = time.perf_counter() - start
    expires = getattr(token, "exp", None)
    print(f"\n[AUTH] token получен за {elapsed:.1f}s, expires_at={expires}")
    return elapsed


def run_sync(*, structured: bool) -> None:
    provider = GigaChatSyncProvider()
    check_auth(provider)
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
    await check_auth_async(provider)
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
        choices=("prompted", "structured"),
        default="prompted",
        help="prompted: схема в инструкциях (надёжнее); structured: native json_schema GigaChat (Beta)",
    )
    parser.add_argument("--async", dest="use_async", action="store_true", help="проверить и асинхронный провайдер")
    args = parser.parse_args()

    config = configure_env()
    print("Конфигурация локального теста:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    if not config["verify_ssl_certs"]:
        print("  WARNING: проверка сертификатов отключена (нет CA_BUNDLE_PATH). "
              "Только для локального теста, в контуре используйте сертификаты.")

    structured = args.mode == "structured"
    if args.use_async:
        asyncio.run(run_async(structured=structured))
    else:
        run_sync(structured=structured)


if __name__ == "__main__":
    main()