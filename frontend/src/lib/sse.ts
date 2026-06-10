/**
 * Consumo do SSE do gateway via fetch + ReadableStream.
 *
 * POST /chat exige body JSON, então EventSource (GET-only) não serve.
 * O parser acumula bytes, decodifica incrementalmente e separa blocos
 * por linha em branco ("\n\n"), o delimitador de eventos do SSE.
 */

export interface RoutePlan {
  domains: string[];
  plan: string;
  clarification: string | null;
}

export interface AgentResult {
  domain: string;
  answer: string;
}

export interface FinalAnswer {
  answer: string;
  trace_id?: string;
}

export interface ErrorPayload {
  detail?: string;
  error?: string;
  trace_id?: string;
}

export type ChatEvent =
  | { type: "route"; data: RoutePlan }
  | { type: "agent"; data: AgentResult }
  | { type: "final"; data: FinalAnswer }
  | { type: "error"; data: ErrorPayload };

export class ChatHttpError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ChatHttpError";
    this.status = status;
  }
}

function parseBlock(block: string): ChatEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  try {
    return { type: event, data: JSON.parse(dataLines.join("\n")) } as ChatEvent;
  } catch {
    return null;
  }
}

export async function streamChat(
  question: string,
  token: string | null,
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["X-Access-Token"] = token;

  const response = await fetch("/chat", {
    method: "POST",
    headers,
    body: JSON.stringify({ question }),
    signal,
  });

  if (!response.ok) {
    let detail = `Erro HTTP ${response.status}`;
    try {
      const body = (await response.json()) as ErrorPayload;
      if (body.detail) detail = body.detail;
    } catch {
      // corpo não-JSON: mantém o detail genérico
    }
    throw new ChatHttpError(response.status, detail);
  }

  if (!response.body) throw new ChatHttpError(response.status, "Resposta sem corpo de streaming.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseBlock(block);
      if (parsed) onEvent(parsed);
    }
  }
}
