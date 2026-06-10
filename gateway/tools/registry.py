"""Tool registry: converte o OpenAPI de cada microsserviço em tools Ollama.

O contrato IA↔negócio é o OpenAPI — a única fonte de verdade é a API.
Cada operation vira `{"type": "function", "function": {...}}`; o executor
monta a chamada HTTP real e devolve `{"status", "body"}` SEM levantar
exceção em erro de negócio (422/404 voltam como observação para o modelo).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import httpx

from gateway.tools.circuit import CircuitBreaker, open_response, transport_error_response

_EXCLUDED_PATHS = frozenset({"/health"})
_JSON_CONTENT = "application/json"


class ToolNotFound(KeyError):
    """Tool inexistente no registry do domínio (ex.: nome alucinado pelo modelo)."""

    def __init__(self, domain: str, name: str, known: list[str]) -> None:
        super().__init__(name)
        self.domain = domain
        self.name = name
        self.known = known


@dataclass(frozen=True)
class ToolSpec:
    """Operation do OpenAPI normalizada: schema p/ o modelo + roteamento HTTP."""

    name: str
    description: str
    method: str
    path: str
    path_params: tuple[str, ...]
    query_params: tuple[str, ...]
    body_params: tuple[str, ...]
    parameters: dict[str, Any]

    def as_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _resolve_ref(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if ref is None:
        return schema
    name = ref.rsplit("/", 1)[-1]
    return components.get(name, {})


def _flatten_schema(schema: dict[str, Any], components: dict[str, Any]) -> dict[str, Any]:
    """Achata um schema de propriedade: resolve $ref e anyOf-com-null (1 nível).

    Os schemas dos serviços são planos; um nível de resolução é suficiente.
    FastAPI representa opcionais como `anyOf: [{...}, {"type": "null"}]`.
    """
    schema = _resolve_ref(schema, components)
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        merged = dict(_resolve_ref(non_null[0], components)) if non_null else {}
        for key in ("description", "default", "title"):
            if key in schema and key not in merged:
                merged[key] = schema[key]
        schema = merged
    if schema.get("type") == "array" and "$ref" in schema.get("items", {}):
        schema = {**schema, "items": _resolve_ref(schema["items"], components)}
    return schema


def _request_body_schema(operation: dict[str, Any], components: dict[str, Any]) -> dict[str, Any] | None:
    body = operation.get("requestBody")
    if body is None:
        return None
    schema = body.get("content", {}).get(_JSON_CONTENT, {}).get("schema")
    if schema is None:
        return None
    schema = _resolve_ref(schema, components)
    if "anyOf" in schema:  # body opcional (ex.: `SettleRequest | None`)
        schema = _flatten_schema(schema, components)
    return schema


def parse_openapi(spec: dict[str, Any]) -> dict[str, ToolSpec]:
    """Converte um documento OpenAPI em ToolSpecs indexados por operation_id."""
    components = spec.get("components", {}).get("schemas", {})
    tools: dict[str, ToolSpec] = {}
    for path, methods in spec.get("paths", {}).items():
        if path in _EXCLUDED_PATHS:
            continue
        for method, operation in methods.items():
            name = operation["operationId"]
            description = ". ".join(
                part.strip().rstrip(".")
                for part in (operation.get("summary", ""), operation.get("description", ""))
                if part.strip()
            )
            properties: dict[str, Any] = {}
            required: list[str] = []
            path_params: list[str] = []
            query_params: list[str] = []
            body_params: list[str] = []

            for param in operation.get("parameters", []):
                prop = _flatten_schema(dict(param.get("schema", {})), components)
                if param.get("description") and "description" not in prop:
                    prop["description"] = param["description"]
                properties[param["name"]] = prop
                if param.get("required", False):
                    required.append(param["name"])
                (path_params if param["in"] == "path" else query_params).append(param["name"])

            body_schema = _request_body_schema(operation, components)
            if body_schema is not None:
                for prop_name, prop_schema in body_schema.get("properties", {}).items():
                    properties[prop_name] = _flatten_schema(dict(prop_schema), components)
                    body_params.append(prop_name)
                body_required = body_schema.get("required", [])
                required.extend(p for p in body_required if p not in required)

            tools[name] = ToolSpec(
                name=name,
                description=description,
                method=method.upper(),
                path=path,
                path_params=tuple(path_params),
                query_params=tuple(query_params),
                body_params=tuple(body_params),
                parameters={"type": "object", "properties": properties, "required": required},
            )
    return tools


class ToolRegistry:
    """Cache em memória das tools por domínio + executor HTTP real."""

    def __init__(
        self,
        service_urls: dict[str, str],
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 30.0,
        internal_api_key: str | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._service_urls = service_urls
        self._client = client or httpx.Client(timeout=timeout_s)
        # Enviada em TODA chamada (incl. openapi.json, isento no serviço) por simetria.
        self._headers = {"X-Internal-Key": internal_api_key} if internal_api_key else None
        self._breaker = breaker or CircuitBreaker()
        self._cache: dict[str, dict[str, ToolSpec]] = {}
        self._lock = threading.Lock()

    def _specs(self, domain: str) -> dict[str, ToolSpec]:
        with self._lock:
            if domain not in self._cache:
                if domain not in self._service_urls:
                    raise ToolNotFound(domain, "", [])
                response = self._client.get(
                    f"{self._service_urls[domain]}/openapi.json", headers=self._headers
                )
                response.raise_for_status()
                self._cache[domain] = parse_openapi(response.json())
            return self._cache[domain]

    def preload(self, domain: str, specs: dict[str, ToolSpec]) -> None:
        """Injeta specs já parseadas (testes/fixtures)."""
        with self._lock:
            self._cache[domain] = specs

    def tools_for(self, domain: str) -> list[dict[str, Any]]:
        """Tools no formato Ollama, escopadas ao serviço do domínio."""
        return [spec.as_ollama_tool() for spec in self._specs(domain).values()]

    def execute(self, domain: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Executa a chamada HTTP real da tool e devolve {"status", "body"}.

        Erros de negócio (422/404) NÃO levantam exceção: o corpo estruturado
        {error, detail, rule} volta como observação para o modelo corrigir.

        Resiliência: falhas de TRANSPORTE (timeout/conexão/5xx) alimentam o
        circuit breaker do domínio; circuito aberto → resposta degradada
        imediata sem chamada HTTP. 4xx é erro de negócio e não conta.
        """
        if not self._breaker.allow(domain):
            return open_response(domain)
        try:
            specs = self._specs(domain)
        except httpx.TransportError:
            self._breaker.record_failure(domain)
            return transport_error_response(domain)
        if name not in specs:
            raise ToolNotFound(domain, name, sorted(specs))
        spec = specs[name]
        args = dict(args or {})

        path = spec.path
        for param in spec.path_params:
            path = path.replace("{" + param + "}", str(args.pop(param, "")))
        query = {k: args.pop(k) for k in spec.query_params if args.get(k) is not None}
        body = {k: v for k, v in args.items() if k in spec.body_params and v is not None} or None

        try:
            response = self._client.request(
                spec.method,
                f"{self._service_urls[domain]}{path}",
                params=query or None,
                json=body,
                headers=self._headers,
            )
        except httpx.TransportError:
            self._breaker.record_failure(domain)
            return transport_error_response(domain)
        if response.status_code >= 500:
            self._breaker.record_failure(domain)
        else:
            self._breaker.record_success(domain)
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return {"status": response.status_code, "body": payload}
