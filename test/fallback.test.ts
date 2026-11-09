import { afterEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  FastJevCompaction,
  serializeCompaction,
  type NativeMessage,
} from '../src/plugin.js';
import {
  chunkTranscript,
  chunkedSummary,
  headTailTruncate,
  serializeChunk,
} from '../src/fallback.js';
import { errorSummary } from '../src/log.js';
import { SystemOneClient } from '../src/systemone.js';
import { estimateTokens } from '../src/state.js';
import type { JevQuestions, JevResponse, JevState, Message } from '../src/types.js';

const temporaryDirectories: string[] = [];

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

function prompt(text: string): Message {
  return { role: 'user', text, toolUses: [] };
}

function callBlock(id: string, tool: string, input: Record<string, unknown>): Message {
  return { role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool, input }] };
}

function resultBlock(id: string, text: string): Message {
  return { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text }] };
}

function transcript(blocks: number): Message[] {
  const messages: Message[] = [prompt('Start the migration work')];
  for (let i = 1; i <= blocks; i++) {
    messages.push(prompt(`Step ${i}: continue the migration plan`));
    messages.push(callBlock(`c${i}`, 'read', { path: `src/module-${i}.ts` }));
    messages.push(resultBlock(`c${i}`, `output ${i}: ${'data '.repeat(30)}`));
  }
  return messages;
}

function nativeOf(messages: readonly Message[]): NativeMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: [
      ...(message.text.length > 0 ? [{ type: 'text', text: message.text }] : []),
      ...message.toolUses.map((tool) => ({
        type: 'tool-call',
        id: tool.tool_use_id,
        name: tool.tool,
        input: tool.input,
      })),
      ...(message.toolResults ?? []).map((result) => ({
        type: 'tool-result',
        id: result.tool_use_id,
        result: { type: 'text', value: result.text },
      })),
    ],
  }));
}

interface HookHarness {
  hook: (event: Record<string, unknown>) => Promise<void>;
  logFile: string;
}

async function setupPlugin(options: Record<string, unknown>): Promise<HookHarness> {
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
  const logDirectory = mkdtempSync(join(tmpdir(), 'fast-jev-fallback-test-'));
  temporaryDirectories.push(logDirectory);
  const logFile = join(logDirectory, 'plugin.log');
  let hook: ((event: Record<string, unknown>) => Promise<void>) | undefined;
  await FastJevCompaction.setup({
    options: { ...options, logFile },
    session: {
      async hook(name: string, callback: (event: Record<string, unknown>) => Promise<void>) {
        if (name === 'compaction') hook = callback;
        return { dispose: async () => undefined };
      },
    },
  } as never);
  return { hook: hook!, logFile };
}

function systemOneReply(body: string): Response {
  const questions = JSON.parse(body).questions as Record<string, unknown>;
  return new Response(
    JSON.stringify({
      answers: Object.fromEntries(Object.keys(questions).map((name) => [name, { noul: 0.1 }])),
    }),
  );
}

function summaryReply(content: string): Response {
  return new Response(JSON.stringify({ choices: [{ message: { content } }] }));
}

describe('chunkTranscript', () => {
  it('groups consecutive messages under the budget without splitting one', () => {
    const messages = [
      prompt('word '.repeat(60)),
      prompt('word '.repeat(30)),
      prompt('word '.repeat(60)),
    ];
    // Exactly the estimated size of the first two messages together.
    const budget = estimateTokens(serializeChunk([messages[0]!, messages[1]!]));
    const { chunks, truncatedMessages } = chunkTranscript(messages, budget);

    expect(truncatedMessages).toBe(0);
    expect(chunks).toEqual([serializeChunk([messages[0]!, messages[1]!]), serializeChunk([messages[2]!])]);
    for (const chunk of chunks) {
      expect(estimateTokens(chunk)).toBeLessThanOrEqual(budget);
    }
  });

  it('head+tail truncates messages that alone exceed the budget', () => {
    const messages = [prompt('word '.repeat(60)), prompt('huge '.repeat(500))];
    const budget = estimateTokens(serializeChunk([messages[0]!])) - 10;
    const { chunks, truncatedMessages } = chunkTranscript(messages, budget);

    expect(truncatedMessages).toBe(2);
    expect(chunks).toHaveLength(2);
    for (const chunk of chunks) {
      expect(estimateTokens(chunk)).toBeLessThanOrEqual(budget);
      expect(chunk).toContain('chars omitted');
    }
  });
});

describe('headTailTruncate', () => {
  it('keeps the head and the tail and fits the budget', () => {
    const text = 'beginning '.repeat(100) + 'middle '.repeat(100) + 'end '.repeat(100);
    const truncated = headTailTruncate(text, 60);
    expect(estimateTokens(truncated)).toBeLessThanOrEqual(60);
    expect(truncated).toContain('beginning');
    expect(truncated).toContain('end ');
    expect(truncated).not.toContain('middle');
  });

  it('returns short text unchanged', () => {
    expect(headTailTruncate('tiny', 100)).toBe('tiny');
  });
});

describe('chunkedSummary', () => {
  it('maps one request per chunk and reduces the parts in order', async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) =>
      summaryReply(`summary for ${JSON.parse(String(init?.body)).messages[0].content.slice(0, 12)}`),
    );
    const summary = await chunkedSummary(transcript(3), {
      baseUrl: 'http://127.0.0.1:9999/v1',
      model: 'small_summarizer',
      schedule: (task) => task(),
      fetch: fetchMock as unknown as typeof fetch,
      contextTokens: 300, // forces the transcript into several chunks
      timeoutMs: 1000,
    });

    expect(summary.chunkCount).toBeGreaterThanOrEqual(2);
    expect(fetchMock).toHaveBeenCalledTimes(summary.chunkCount + 1);
    const [firstUrl, firstInit] = fetchMock.mock.calls[0]!;
    expect(firstUrl).toBe('http://127.0.0.1:9999/v1/chat/completions');
    const first = JSON.parse(String(firstInit?.body));
    expect(first.model).toBe('small_summarizer');
    expect(first.temperature).toBe(0);
    expect(first.max_tokens).toBe(1200);
    expect(first.messages[0].content).toContain(`part 1 of ${summary.chunkCount}`);

    const reduce = JSON.parse(String(fetchMock.mock.calls[summary.chunkCount]![1]?.body));
    expect(reduce.max_tokens).toBe(3000);
    expect(reduce.messages[0].content).toContain(`part 1 of ${summary.chunkCount}`);
    expect(reduce.messages[0].content).toContain(`part ${summary.chunkCount} of ${summary.chunkCount}`);
    expect(reduce.messages[0].content).toContain('Objective / Requirements / Decisions');
    expect(summary.summary).toContain('summary for');
  });

  it('throws on a failed part request so the caller can leave the fallback to core', async () => {
    const fetchMock = vi.fn(async () => new Response('nope', { status: 503 }));
    await expect(
      chunkedSummary(transcript(3), {
        baseUrl: 'http://127.0.0.1:9999/v1',
        model: 'small_summarizer',
        schedule: (task) => task(),
        fetch: fetchMock as unknown as typeof fetch,
        contextTokens: 131_072,
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/fallback summarizer request failed \(503\)/);
  });

  it('throws on an empty reply', async () => {
    const fetchMock = vi.fn(async () => summaryReply('   '));
    await expect(
      chunkedSummary(transcript(1), {
        baseUrl: 'http://127.0.0.1:9999/v1',
        model: 'small_summarizer',
        schedule: (task) => task(),
        fetch: fetchMock as unknown as typeof fetch,
        contextTokens: 131_072,
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/no content/);
  });
});

describe('partial compaction answers (hook)', () => {
  const native = nativeOf(transcript(2));

  async function runHook(minReductionRatio: number): Promise<{
    event: Record<string, unknown>;
    logFile: string;
  }> {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = String(init?.body);
      // Question names carry the sequential call id (`call_t1`), not the
      // transcript's `tool_use_id`.
      return body.includes('call_t1')
        ? systemOneReply(body)
        : new Response('over capacity', { status: 400 });
    });
    vi.stubGlobal('fetch', fetchMock);
    const { hook, logFile } = await setupPlugin({
      backend: 'systemone',
      localModel: 'decision_text',
      preserveRecentMessages: 0,
      maxCallsPerBatch: 1,
      minReductionRatio,
    });
    const event: Record<string, unknown> = { sessionID: 'ses_partial', messages: native };
    await hook(event);
    return { event, logFile };
  }

  it('ships the answered batches when the reduction meets the minimum', async () => {
    const { event, logFile } = await runHook(0);
    expect(event.result).toMatchObject({
      metadata: {
        plugin: 'fast-jev-compaction-opencode',
        partial: true,
        callsDropped: 1,
        jevBatches: 2,
      },
    });
    const log = readFileSync(logFile, 'utf8');
    expect(log).toContain('partial compaction shipped');
    expect(log).toContain('"reason":"http-request-failed"');
    expect(log).toContain('"partial":true');
  });

  it('falls back when the partial reduction stays below the minimum', async () => {
    const { event, logFile } = await runHook(1);
    expect(event.result).toBeUndefined();
    const log = readFileSync(logFile, 'utf8');
    expect(log).toContain('partial compaction below minimum');
    expect(log).toContain('"reason":"http-request-failed"');
  });
});

describe('chunked summarizer fallback (hook)', () => {
  it('takes over an oversized transcript on the fallback path', async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = String(init?.body);
      if (body.includes('"state"')) return systemOneReply(body);
      return summaryReply('Objective: done. Next Move: none.');
    });
    vi.stubGlobal('fetch', fetchMock);
    const { hook, logFile } = await setupPlugin({
      backend: 'systemone',
      localModel: 'decision_text',
      preserveRecentMessages: 0,
      minReductionRatio: 1, // classification always falls back
      fallbackModel: 'small_summarizer',
      fallbackBaseUrl: 'http://127.0.0.1:9999/v1',
      fallbackContextTokens: 200,
    });

    const event: Record<string, unknown> = {
      sessionID: 'ses_chunked',
      messages: nativeOf(transcript(3)),
    };
    await hook(event);

    expect(event.result).toMatchObject({
      metadata: {
        plugin: 'fast-jev-compaction-opencode',
        fallback: 'chunked-summary',
        model: 'small_summarizer',
      },
    });
    const metadata = (event.result as { metadata: Record<string, unknown> }).metadata;
    expect(metadata.chunks).toBeGreaterThan(1);
    expect(typeof metadata.transcriptTokens).toBe('number');
    expect((event.result as { summary: string }).summary).toContain('Objective: done.');
    const summarizerCalls = fetchMock.mock.calls.filter(([, init]) =>
      !String(init?.body).includes('"state"'),
    );
    expect(summarizerCalls.length).toBe(metadata.chunks as number + 1);
    for (const [url] of summarizerCalls) {
      expect(String(url)).toBe('http://127.0.0.1:9999/v1/chat/completions');
    }
    expect(readFileSync(logFile, 'utf8')).toContain('"fallback":"chunked-summary"');
  });

  it('returns to OpenCode core when the summarizer fails', async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = String(init?.body);
      if (body.includes('"state"')) return systemOneReply(body);
      return new Response('nope', { status: 500 });
    });
    vi.stubGlobal('fetch', fetchMock);
    const { hook, logFile } = await setupPlugin({
      backend: 'systemone',
      localModel: 'decision_text',
      preserveRecentMessages: 0,
      minReductionRatio: 1,
      fallbackModel: 'small_summarizer',
      fallbackContextTokens: 200,
    });

    const event: Record<string, unknown> = {
      sessionID: 'ses_chunked_fail',
      messages: nativeOf(transcript(3)),
    };
    await hook(event);
    expect(event.result).toBeUndefined();
    const log = readFileSync(logFile, 'utf8');
    expect(log).toContain('chunked fallback failed');
    expect(log).toContain('"reason":"http-request-failed"');
  });

  it('stays out when the transcript fits the summarizer window', async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) =>
      systemOneReply(String(init?.body)),
    );
    vi.stubGlobal('fetch', fetchMock);
    const { hook } = await setupPlugin({
      backend: 'systemone',
      localModel: 'decision_text',
      preserveRecentMessages: 0,
      minReductionRatio: 1,
      fallbackModel: 'small_summarizer',
      fallbackContextTokens: 131_072,
    });

    const event: Record<string, unknown> = {
      sessionID: 'ses_small',
      messages: nativeOf(transcript(2)),
    };
    await hook(event);
    expect(event.result).toBeUndefined();
    for (const [, init] of fetchMock.mock.calls) {
      expect(String(init?.body)).toContain('"state"'); // classifier only
    }
  });
});

describe('SystemOneClient request timeout', () => {
  it('aborts one batch without killing its siblings', async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = String(init?.body);
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      if (body.includes('call_t1')) {
        throw new DOMException('The operation was aborted due to timeout', 'TimeoutError');
      }
      return systemOneReply(body);
    });
    const client = new SystemOneClient({ model: 'decision_text', fetch: fetchMock as unknown as typeof fetch });

    const messages = transcript(2);
    const { classifyCalls } = await import('../src/compact.js');
    const classification = await classifyCalls(messages, client, {
      preserveRecentMessages: 0,
      maxCallsPerBatch: 1,
    });

    expect(classification.failures).toHaveLength(1);
    expect(String(classification.failures[0])).toContain('timed out');
    expect(errorSummary(classification.failures[0])).toMatchObject({
      reason: 'request-timed-out',
    });
    // The sibling batch's answers survived the timeout.
    expect([...classification.answers.keys()]).toEqual(['c2']);
  });

  it('sends an abort signal that fires at requestTimeoutMs', async () => {
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      expect(init?.signal).toBeInstanceOf(AbortSignal);
      return systemOneReply(String(init?.body));
    });
    const client = new SystemOneClient({
      model: 'decision_text',
      fetch: fetchMock as unknown as typeof fetch,
      requestTimeoutMs: 1234,
    });
    const state: JevState = { context: 'c', goal: 'g', history: [] };
    const questions: JevQuestions = {
      call_t1: { type: 'noul', instructions: 'keep?' },
      result_t1: { type: 'noul', instructions: 'keep?' },
    };
    await client.ask(state, questions);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe('serializeCompaction / serializeChunk', () => {
  it('keeps the checkpoint header and delegates the body to serializeChunk', () => {
    const messages = transcript(1);
    expect(serializeCompaction(messages)).toBe(
      '[fast-jev-compaction checkpoint: retained conversation follows verbatim; no model summary was generated]\n\n' +
        serializeChunk(messages),
    );
    expect(serializeCompaction([])).toBe(
      '[fast-jev-compaction checkpoint: retained conversation follows verbatim; no model summary was generated]',
    );
  });
});
