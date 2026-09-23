# TypeSafe System One адаптер поверх GigaChat

Реализация провайдера GigaChat для официального адаптера TypeSafe
(`system-one-adapter-python`). Адаптер не обращается к API TypeSafe: он сам
готовит вопросы, схему ответа и system prompt, а модель (здесь — GigaChat)
выполняет вызов. Работает в защищённом контуре: весь трафик идёт на контурные
эндпоинты GigaChat по TLS с сертификатами.

## Поток работы

1. `SystemOneAdapterClient` собирает вопросы (`Noul` / `Score` / `Choice`) в
   динамическую Pydantic-модель и системный промпт (с защитой от prompt
   injection: содержимое документа считается непроверяемыми данными).
2. Провайдер `GigaChatSyncProvider` / `GigaChatAsyncProvider` выполняет один
   вызов `chat.create()` (native-structure) или обычный `chat.create()`
   (prompted), парсит текст ответа и возвращает токены `input`/`output`.
3. Клиент распаковывает ответ в `SystemOneResponse` (choice/score/noul с
   вероятностями) и при malformed-ответе делает корректирующие повторы.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка окружения (сертификаты, креды, модель)

```bash
# Учётные данные OAuth 2.0: <client_id>:<secret> в base64 (client_credentials)
export GIGACHAT_CREDENTIALS='BASE64(client_id:secret)'

# TLS серверной стороны: свой УЦ внутри контура
export GIGACHAT_CA_BUNDLE_FILE=/path/to/root-ca.crt

# mTLS: клиентский сертификат и ключ
export GIGACHAT_CERT_FILE=/path/to/client.crt
export GIGACHAT_KEY_FILE=/path/to/client.key
export GIGACHAT_KEY_FILE_PASSWORD=''            # если ключ зашифрован паролем

# При необходимости — адреса внутри контура
export GIGACHAT_BASE_URL='https://<gateway>/v1'
export GIGACHAT_AUTH_URL='https://<oauth>/oauth'

export GIGACHAT_MODEL='GigaChat-2'
export GIGACHAT_VERIFY_SSL_CERTS='true'
export GIGACHAT_TIMEOUT='60'
```

Полный список переменных — `gigachat.settings.Settings` (префикс `GIGACHAT_`):
`BASE_URL`, `AUTH_URL`, `CREDENTIALS`, `SCOPE`, `ACCESS_TOKEN`, `MODEL`,
`PROFANITY_CHECK`, `USER`, `PASSWORD`, `TIMEOUT`, `VERIFY_SSL_CERTS`,
`CA_BUNDLE_FILE`, `CERT_FILE`, `KEY_FILE`, `KEY_FILE_PASSWORD`.

Альтернативно всё можно передать параметрами конструктора провайдера.

## Запуск

```bash
python demo_gigachat.py --mode structured   # native json_schema (Beta в GigaChat)
python demo_gigachat.py --mode prompted     # схема в промпте, разбор на клиенте
python demo_gigachat.py --mode prompted --async   # асинхронный провайдер
```

### Локальный тест по credentials (вне защищённого контура)

Проверяет доступ к GigaChat по OAuth `client_credentials` из `.env`
(`GIGACHAT_CREDENTIALS`), получает токен и прогоняет полную оценку через
адаптер. Логика TLS как в asr.lourie.info: если файл `CA_BUNDLE_PATH`
существует — верификация через него, иначе `verify_ssl_certs=False`.

```bash
pip install -r requirements.txt   # включает python-dotenv
python test_local_gigachat.py                 # prompted, sync
python test_local_gigachat.py --mode structured
python test_local_gigachat.py --async
```

В защищённом контуре вместо этого используйте сертификаты
(`GIGACHAT_CERT_FILE`/`GIGACHAT_KEY_FILE`/`GIGACHAT_CA_BUNDLE_FILE`), как в
`gigachat_provider.py` — локальный тест to целевой контур не применим.

## Использование в своём коде

```python
from gigachat_provider import GigaChatSyncProvider
from system_one_adapter import SystemOneAdapterClient
from typesafe_sdk import RetryPolicy

provider = GigaChatSyncProvider()          # настройки из GIGACHAT_*
client = SystemOneAdapterClient(
    model=provider,
    structured_outputs=True,               # или False для prompted-режима
    llm_answer_mode="probabilities",       # или "discrete"
    retry=RetryPolicy(max_retries=2),
)

with client:
    response = client.system_one({"text": "..."}, {
        "q1": Noul(type="noul", instructions="...", criteria={"true": "...", "false": "..."}),
        "q2": Score(type="score", instructions="...", criteria=["плохо", "хорошо"]),
        "q3": Choice(type="choice", instructions="...", criteria={"A": "...", "B": "..."}),
    })

print(response.answers["q1"])   # NoulAnswer(noul=..., confidence=...)
```

## Важные детали

- При `structured_outputs=True` используется `response_format =
  {"type": "json_schema", "schema": <схема>, "strict": true}` — в GigaChat
  помечено как Beta и может не работать для всех версий моделей. При ошибке
  API ошибка транслируется как `TypeSafeAPIStatusError` (не ретраится, если
  статус не в списке повторов).
- При `structured_outputs=False` схема вставляется в system prompt, и клиент
  сам разбирает JSON и делает corrective-retries (`n_retry_malformed_structure`).
- Официальный timeout/connection errors GigaChat отображаются в
  `TypeSafeAPITimeoutError` / `TypeSafeAPIConnectionError`.
- Провайдер подходит и как самостоятельный объект для
  `client.system_one(..., model=provider_instance)`, и как
  `client = SystemOneAdapterClient(model=provider_instance)`.

## Ограничения

- Это НЕ реализация Jev/OpenJev: движок GigaChat — отдельная LLM, а не новая
  архитектура TypeSafe. Адаптер лишь воспроизводит клиентский протокол System One.
- GigaChat structured output (json_schema) — Beta: для надёжности в проде
  рекомендован `--mode prompted` с небольшим числом corrective-retries.