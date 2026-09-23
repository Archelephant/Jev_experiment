"""TypeSafe System One providers backed by GigaChat.

This module implements the ``SyncProvider`` / ``AsyncProvider`` protocol from
``system_one_adapter.providers.base`` so a GigaChat deployment (optionally behind
mTLS certificates) can be used as the model for the TypeSafe-compatible adapter.

Which model answers every call:
    The model name is resolved from, in priority order: the ``model_name``
    constructor argument, the ``GIGACHAT_MODEL`` environment variable, and the
    ``model`` parameter on the GigaChat client.

Transport and certificates:
    Everything is driven by the standard ``GIGACHAT_*`` environment variables handled
    by ``gigachat.settings.Settings``: ``GIGACHAT_CREDENTIALS``,
    ``GIGACHAT_ACCESS_TOKEN``, ``GIGACHAT_BASE_URL``, ``GIGACHAT_AUTH_URL``,
    ``GIGACHAT_CA_BUNDLE_FILE``, ``GIGACHAT_CERT_FILE``, ``GIGACHAT_KEY_FILE``,
    ``GIGACHAT_VERIFY_SSL_CERTS``, and ``GIGACHAT_TIMEOUT``. Constructor arguments
    override the environment when both are provided.
"""

from __future__ import annotations

from typing import Any

import httpx
from gigachat import GigaChatAsyncClient, GigaChatSyncClient
from gigachat.exceptions import (
    AuthenticationError,
    BadRequestError,
    ForbiddenError,
    GigaChatException,
    NotFoundError,
    RateLimitError,
    RequestEntityTooLargeError,
    ResponseError,
    ServerError,
    UnprocessableEntityError,
)
from gigachat.models.chat_completions import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatModelOptions,
    ChatResponseFormat,
)
from gigachat.settings import Settings
from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPITimeoutError, TypeSafeError
from typesafe_sdk._core.errors import api_error

from system_one_adapter.providers.base import (
    Message,
    ProviderResult,
    record_request,
    record_response,
    render_messages,
    translating,
)

__all__ = ["GigaChatAsyncProvider", "GigaChatSyncProvider"]

_STATUS_ERRORS = (
    AuthenticationError,
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    RequestEntityTooLargeError,
    RateLimitError,
    UnprocessableEntityError,
    ServerError,
    ResponseError,
)


def _resolve_model_name(model_name: str | None) -> str:
    if model_name and model_name.strip():
        return model_name
    configured = Settings().model
    if configured and configured.strip():
        return configured
    raise ValueError("A GigaChat model is required: pass `model_name` or set GIGACHAT_MODEL.")


def _translate_error(error: Exception) -> TypeSafeError:
    """Map a GigaChat or httpx exception to a `typesafe_sdk` error."""
    import httpx2

    if isinstance(error, TypeSafeError):
        return error
    if isinstance(error, ResponseError):
        headers = dict(error.headers or {})
        body: Any = error.content
        if isinstance(body, bytes):
            body = body.decode("utf-8", errors="replace")
        return api_error(error.status_code, body, headers)
    if isinstance(error, GigaChatException):
        return TypeSafeError(str(error))
    if isinstance(error, httpx.TimeoutException):
        return TypeSafeAPITimeoutError(httpx2.Timeout(None))
    if isinstance(error, httpx.TransportError):
        return TypeSafeAPIConnectionError(str(error))
    return TypeSafeError(str(error))


def _result(response: ChatCompletionResponse) -> ProviderResult:
    """Extract the assistant text and token counts from a chat completion."""
    finish_reason = response.finish_reason
    record_response(response, finish_reason=finish_reason)
    if finish_reason not in (None, "stop"):
        raise TypeSafeError(f"GigaChat completion did not finish cleanly: {finish_reason!r}.")
    text = "".join(
        part.text or ""
        for message in response.messages
        if message.role == "assistant" and message.content
        for part in message.content
    )
    usage = response.usage
    return ProviderResult(
        text=text,
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
    )


def _request_payload(
    messages: list[Message],
    model_name: str,
    schema: dict[str, Any],
    *,
    structured: bool,
    strict: bool,
) -> dict[str, Any]:
    rendered = render_messages(messages)
    request: dict[str, Any] = {
        "model": model_name,
        "messages": [ChatMessage(role=item["role"], content=item["content"]) for item in rendered],
    }
    if structured:
        request["model_options"] = ChatModelOptions(
            response_format=ChatResponseFormat(type="json_schema", schema=schema, strict=strict)
        )
    return request


class GigaChatSyncProvider:
    """Call GigaChat synchronously as a TypeSafe System One provider."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        strict: bool = True,
        credentials: str | None = None,
        access_token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
        auth_url: str | None = None,
        verify_ssl_certs: bool | None = None,
        ca_bundle_file: str | None = None,
        cert_file: str | None = None,
        key_file: str | None = None,
        key_file_password: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the provider and its synchronous GigaChat client.

        Args:
            model_name: Model to request (fallback: ``GIGACHAT_MODEL`` env var).
            strict: Request strict schema adherence for native structured output.
            credentials: ``client_id:secret`` used to obtain an OAuth token.
            access_token: Ready-to-use JWE token instead of ``credentials``.
            user: Username for HTTP basic authentication.
            password: Password for HTTP basic authentication.
            base_url: GigaChat API endpoint (fallback: ``GIGACHAT_BASE_URL``).
            auth_url: OAuth token endpoint (fallback: ``GIGACHAT_AUTH_URL``).
            verify_ssl_certs: Whether to verify the server TLS certificate.
            ca_bundle_file: Path to a CA bundle used to verify the server.
            cert_file: Path to the client TLS certificate (mTLS).
            key_file: Path to the client TLS private key (mTLS).
            key_file_password: Password for an encrypted client private key.
            timeout: Request timeout in seconds (fallback: ``GIGACHAT_TIMEOUT``).
        """
        self.model_name = _resolve_model_name(model_name)
        self._strict = strict
        self._client = GigaChatSyncClient(
            model=self.model_name,
            credentials=credentials,
            access_token=access_token,
            user=user,
            password=password,
            base_url=base_url,
            auth_url=auth_url,
            verify_ssl_certs=verify_ssl_certs,
            ca_bundle_file=ca_bundle_file,
            cert_file=cert_file,
            key_file=key_file,
            key_file_password=key_file_password,
            timeout=timeout,
        )

    def translate_error(self, error: Exception) -> TypeSafeError:
        """Map an exception raised by the GigaChat SDK to a TypeSafe error."""
        return _translate_error(error)

    def close(self) -> None:
        """Release the underlying httpx connection pool."""
        self._client.close()

    def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one synchronous GigaChat request.

        Args:
            messages: Conversation in provider-neutral form.
            schema: JSON schema for the model's answer.
            structured: Use native ``json_schema`` response format when ``True``;
                otherwise the client prompts the JSON schema in the instructions
                and parses the answer client-side.

        Returns:
            The response text and input and output token counts.

        Raises:
            TypeSafeError: The GigaChat request fails or the completion does not
                finish cleanly.
        """
        with translating(self.translate_error):
            request = _request_payload(
                messages,
                self.model_name,
                schema,
                structured=structured,
                strict=self._strict,
            )
            record_request(request, api="gigachat.chat.create")
            response = self._client.chat.create(ChatCompletionRequest(**request))
        return _result(response)


class GigaChatAsyncProvider:
    """Call GigaChat asynchronously as a TypeSafe System One provider."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        strict: bool = True,
        credentials: str | None = None,
        access_token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        base_url: str | None = None,
        auth_url: str | None = None,
        verify_ssl_certs: bool | None = None,
        ca_bundle_file: str | None = None,
        cert_file: str | None = None,
        key_file: str | None = None,
        key_file_password: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the provider and its asynchronous GigaChat client.

        Args mirror `GigaChatSyncProvider`; only the transport is asynchronous.
        """
        self.model_name = _resolve_model_name(model_name)
        self._strict = strict
        self._client = GigaChatAsyncClient(
            model=self.model_name,
            credentials=credentials,
            access_token=access_token,
            user=user,
            password=password,
            base_url=base_url,
            auth_url=auth_url,
            verify_ssl_certs=verify_ssl_certs,
            ca_bundle_file=ca_bundle_file,
            cert_file=cert_file,
            key_file=key_file,
            key_file_password=key_file_password,
            timeout=timeout,
        )

    def translate_error(self, error: Exception) -> TypeSafeError:
        """Map an exception raised by the GigaChat SDK to a TypeSafe error."""
        return _translate_error(error)

    async def aclose(self) -> None:
        """Release the underlying httpx connection pool."""
        await self._client.aclose()

    async def request(
        self,
        messages: list[Message],
        *,
        schema: dict[str, Any],
        structured: bool,
    ) -> ProviderResult:
        """Perform one asynchronous GigaChat request.

        Args and return value match `GigaChatSyncProvider.request`.
        """
        with translating(self.translate_error):
            request = _request_payload(
                messages,
                self.model_name,
                schema,
                structured=structured,
                strict=self._strict,
            )
            record_request(request, api="gigachat.achat.create")
            response = await self._client.achat.create(ChatCompletionRequest(**request))
        return _result(response)