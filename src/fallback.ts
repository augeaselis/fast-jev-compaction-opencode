import { estimateTokens } from './state.js';
import type { Message } from './types.js';

/**
 * Share of the fallback summarizer's context window one chunk may use; the
 * rest is left for the prompt framing and the generated summary.
 */
export const CHUNK_SHARE = 0.6;

/** OpenCode core's compaction summarizer window. */
export const FALLBACK_CONTEXT_TOKENS = 131_072;

/** Generation caps for the per-part summaries and the final reduce call. */
export const PART_MAX_TOKENS = 1200;
export const REDUCE_MAX_TOKENS = 3000;

/** Per-request deadline for the fallback summarizer calls. */
export const FALLBACK_REQUEST_TIMEOUT_MS = 120_000;

function json(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return '[unserializable]';
  }
}

/**
 * Serializes messages the way the compaction override does, minus the
 * checkpoint header: one verbatim block per message, tool output included.
 */
export function serializeChunk(messages: readonly Message[]): string {
  const blocks: string[] = [];
  for (const message of messages) {
    const lines: string[] = [`<${message.role}>`];
    if (message.text.length > 0) lines.push(message.text);

    for (const tool of message.toolUses) {
      lines.push(`[tool-call id=${tool.tool_use_id} name=${tool.tool}]`);
      lines.push(json(tool.input));
      if (tool.text !== undefined) {
        lines.push(`[tool-result id=${tool.tool_use_id}${tool.isError ? ' error=true' : ''}]`);
        lines.push(tool.text);
      }
    }

    for (const result of message.toolResults ?? []) {
      if (message.toolUses.some((tool) => tool.tool_use_id === result.tool_use_id)) continue;
      lines.push(`[tool-result id=${result.tool_use_id}${result.isError ? ' error=true' : ''}]`);
      lines.push(result.text);
    }

    lines.push(`</${message.role}>`);
    blocks.push(lines.join('\n'));
  }
  return blocks.join('\n\n');
}

const omittedMarker = (omitted: number): string => `\n[… ${omitted} chars omitted …]\n`;

/**
 * Longest head+tail form of `text` estimated to fit `maxTokens`: the tail
 * keeps a quarter of the budget, the head the rest.
 */
export function headTailTruncate(text: string, maxTokens: number): string {
  if (estimateTokens(text) <= maxTokens) return text;
  const charsPerToken = text.length / estimateTokens(text);
  const tailChars = Math.max(1, Math.floor((maxTokens / 4) * charsPerToken));
  const tail = text.slice(-tailChars);
  let low = 0;
  let high = text.length - tailChars;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    const candidate =
      text.slice(0, mid) + omittedMarker(text.length - mid - tailChars) + tail;
    if (estimateTokens(candidate) <= maxTokens) low = mid;
    else high = mid - 1;
  }
  return text.slice(0, low) + omittedMarker(text.length - low - tailChars) + tail;
}

export interface TranscriptChunks {
  chunks: string[];
  /** Messages that alone exceeded the budget, head+tail truncated to fit. */
  truncatedMessages: number;
}

/**
 * Splits the transcript into consecutive serialized chunks that each fit
 * `budgetTokens` estimated tokens. Never splits a message; a single message
 * over the budget is head+tail truncated to fit on its own.
 */
export function chunkTranscript(
  messages: readonly Message[],
  budgetTokens: number,
): TranscriptChunks {
  const chunks: string[] = [];
  let current: string[] = [];
  let currentTokens = 0;
  let truncatedMessages = 0;

  const flush = (): void => {
    if (current.length === 0) return;
    chunks.push(current.join('\n\n'));
    current = [];
    currentTokens = 0;
  };

  for (const message of messages) {
    const text = serializeChunk([message]);
    const tokens = estimateTokens(text);
    if (tokens > budgetTokens) {
      flush();
      chunks.push(headTailTruncate(text, budgetTokens));
      truncatedMessages += 1;
      continue;
    }
    if (current.length > 0 && currentTokens + tokens > budgetTokens) flush();
    current.push(text);
    currentTokens += tokens;
  }
  flush();
  return { chunks, truncatedMessages };
}

export interface FallbackSummaryOptions {
  /** OpenAI-compatible base URL the summarizer model is served at. */
  baseUrl: string;
  model: string;
  apiKey?: string;
  /** Runs the part requests; gate on the classifier's concurrency cap. */
  schedule: <T>(task: () => Promise<T>) => Promise<T>;
  fetch: typeof fetch;
  /** The summarizer's context window; the chunk budget derives from it. */
  contextTokens: number;
  /** Override for tests; default {@link FALLBACK_REQUEST_TIMEOUT_MS}. */
  timeoutMs?: number;
}

export interface FallbackSummary {
  summary: string;
  chunkCount: number;
  truncatedMessages: number;
}

const SECTION_PROMPT =
  'extract durable facts for continuing the work (objective, requirements, decisions, work state, files, context) as terse single-line bullets under those headings; preserve exact paths, commands, identifiers, error strings';

async function chatCompletion(
  options: FallbackSummaryOptions,
  content: string,
  maxTokens: number,
): Promise<string> {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (options.apiKey) headers.authorization = `Bearer ${options.apiKey}`;
  const url = `${options.baseUrl.replace(/\/+$/, '')}/chat/completions`;
  const response = await options.fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: options.model,
      messages: [{ role: 'user', content }],
      temperature: 0,
      max_tokens: maxTokens,
    }),
    signal: AbortSignal.timeout(options.timeoutMs ?? FALLBACK_REQUEST_TIMEOUT_MS),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`fallback summarizer request failed (${response.status}): ${text.slice(0, 200)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('fallback summarizer returned malformed JSON');
  }
  const choices = (parsed as { choices?: Array<{ message?: { content?: unknown } }> } | null)
    ?.choices;
  const reply = choices?.[0]?.message?.content;
  if (typeof reply !== 'string' || reply.trim().length === 0) {
    throw new Error('fallback summarizer returned no content');
  }
  return reply;
}

/**
 * Summarizes a transcript too large for one summarizer request: chunks it,
 * maps each chunk to a part summary through the shared scheduler, and reduces
 * the parts in order into one final summary. Throws on any failure so the
 * caller can leave the fallback to OpenCode core.
 */
export async function chunkedSummary(
  transcript: readonly Message[],
  options: FallbackSummaryOptions,
): Promise<FallbackSummary> {
  const budget = Math.floor(options.contextTokens * CHUNK_SHARE);
  const { chunks, truncatedMessages } = chunkTranscript(transcript, budget);
  if (chunks.length === 0) throw new Error('fallback summarizer got an empty transcript');

  const parts = await Promise.all(
    chunks.map((text, index) =>
      options.schedule(async () =>
        chatCompletion(
          options,
          `part ${index + 1} of ${chunks.length} of a coding session transcript; ${SECTION_PROMPT}\n\n${text}`,
          PART_MAX_TOKENS,
        ),
      ),
    ),
  );

  // The section template mirrors OpenCode core's
  // `packages/core/src/session/compaction.ts` SUMMARY_TEMPLATE — the checkpoint
  // format is the contract; core owns its wording.
  const summary = await chatCompletion(
    options,
    [
      'Combine these part summaries of one coding session transcript into the final checkpoint summary.',
      'Use exactly these sections: Objective / Requirements / Decisions / Work State (with Completed, Active, Blocked) / Next Move / Relevant Files / Important Context.',
      'Later parts take precedence where they conflict; keep exact paths, commands, identifiers, and error strings.',
      '',
      ...parts.map((part, index) => `--- part ${index + 1} of ${parts.length} ---\n${part}`),
    ].join('\n'),
    REDUCE_MAX_TOKENS,
  );
  return { summary, chunkCount: chunks.length, truncatedMessages };
}
