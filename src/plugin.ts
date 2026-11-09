import { Plugin } from '@opencode/plugin';

import { JevClient } from './client.js';
import { PartialCompactionError, compact, reductionRatio, resolveOptions } from './compact.js';
import { CHUNK_SHARE, FALLBACK_CONTEXT_TOKENS, chunkedSummary, serializeChunk } from './fallback.js';
import { LOCAL_BASE_URL, LOCAL_MODEL, LocalJevAsker, createScheduler } from './local.js';
import { createLogger, defaultLogPath, errorSummary, safeUrl, type Logger } from './log.js';
import { collectToolCalls, estimateTokens } from './state.js';
import {
  DEFAULT_REQUEST_TIMEOUT_MS,
  LOCAL_SYSTEM_ONE_BASE_URL,
  SystemOneClient,
  isRequestTimeout,
  systemOneEndpoint,
} from './systemone.js';
import type { CompactOptions, CompactResult, JevAsker, Message } from './types.js';

type Backend = 'typesafe' | 'local' | 'systemone';
type PluginOptions = Record<string, unknown>;

type NativeTextPart = {
  type: 'text';
  text: string;
  [key: string]: unknown;
};

type NativeToolCallPart = {
  type: 'tool-call';
  id: string;
  name: string;
  input: unknown;
  [key: string]: unknown;
};

type NativeToolResultPart = {
  type: 'tool-result';
  id: string;
  result: {
    type: string;
    value?: unknown;
    [key: string]: unknown;
  };
  [key: string]: unknown;
};

type NativePart =
  | NativeTextPart
  | NativeToolCallPart
  | NativeToolResultPart
  | { type: string; [key: string]: unknown };

export type NativeMessage = {
  id?: string;
  role: string;
  content: NativePart[];
  [key: string]: unknown;
};

export interface FastJevConfig extends CompactOptions {
  backend: Backend;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  localModel?: string;
  localBaseUrl?: string;
  /** Bearer token for `localBaseUrl`; needed when a proxy fronts the model. */
  localApiKey?: string;
  logFile?: string;
  localConcurrency: number;
  localContextTokens: number;
  minReductionRatio: number;
  /** Model for the plugin's chunked summarizer fallback; unset = core handles it. */
  fallbackModel?: string;
  /** Defaults to `localBaseUrl`. */
  fallbackBaseUrl?: string;
  /** Defaults to `localApiKey`. */
  fallbackApiKey?: string;
  fallbackContextTokens: number;
}

const DEFAULTS = {
  backend: 'typesafe' as const,
  localConcurrency: 2,
  localContextTokens: 64_000,
  minReductionRatio: 0.25,
  fallbackContextTokens: FALLBACK_CONTEXT_TOKENS,
};

function stringOption(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function numberOption(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveConfig(options: PluginOptions = {}): FastJevConfig {
  const backend = options.backend === 'systemone' ? 'systemone' : options.backend === 'local' ? 'local' : 'typesafe';
  const config: FastJevConfig = {
    backend,
    localConcurrency: Math.max(1, Math.floor(numberOption(options, 'localConcurrency', DEFAULTS.localConcurrency))),
    localContextTokens: Math.max(1, numberOption(options, 'localContextTokens', DEFAULTS.localContextTokens)),
    minReductionRatio: Math.max(0, Math.min(1, numberOption(options, 'minReductionRatio', DEFAULTS.minReductionRatio))),
    fallbackContextTokens: Math.max(1, numberOption(options, 'fallbackContextTokens', DEFAULTS.fallbackContextTokens)),
  };

  // Smaller default state avoids exceeding Ollama's 64 KiB text/schema limit.
  if (backend === 'systemone') {
    config.maxStateTokens = 8_000;
    // A gateway request holds at most 64 questions; 32 calls x 2 questions fill it,
    // and the token ceiling is set just past that so the call count is what binds.
    config.maxRequestTokens = 12_500;
    config.maxCallsPerBatch = 32;
  }

  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'maxCallsPerBatch',
    'truncateHeadChars',
    'serverContextTokens',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) config[key] = value;
  }

  for (const key of [
    'apiKey',
    'model',
    'baseUrl',
    'localModel',
    'localBaseUrl',
    'localApiKey',
    'logFile',
    'goal',
    'fallbackModel',
    'fallbackBaseUrl',
    'fallbackApiKey',
  ] as const) {
    const value = stringOption(options, key);
    if (value) config[key] = value;
  }

  return config;
}

function textOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === undefined) return '';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function inputOf(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return { value };
}

function isText(part: NativePart): part is NativeTextPart {
  return part.type === 'text' && typeof (part as NativeTextPart).text === 'string';
}

function isToolCall(part: NativePart): part is NativeToolCallPart {
  return (
    part.type === 'tool-call' &&
    typeof (part as NativeToolCallPart).id === 'string' &&
    typeof (part as NativeToolCallPart).name === 'string'
  );
}

function isToolResult(part: NativePart): part is NativeToolResultPart {
  return (
    part.type === 'tool-result' &&
    typeof (part as NativeToolResultPart).id === 'string' &&
    !!(part as NativeToolResultPart).result &&
    typeof (part as NativeToolResultPart).result === 'object'
  );
}

function resultInfo(part: NativeToolResultPart): { text: string; isError: boolean } {
  const value = part.result.value;
  return {
    text: value === undefined ? textOf(part.result) : textOf(value),
    isError: part.result.type === 'error',
  };
}

/**
 * Adapts OpenCode V2 model messages to the transport-neutral transcript used by
 * the upstream fast-JEV compaction core. Tool-result-only messages are treated
 * as user turns because the core models Claude-style tool results that way.
 */
export function toCompactionMessages(messages: readonly NativeMessage[]): Message[] {
  return messages.flatMap((message) => {
    const toolUses = message.content.filter(isToolCall).map((part) => ({
      tool_use_id: part.id,
      tool: part.name,
      input: inputOf(part.input),
    }));
    const toolResults = message.content.filter(isToolResult).map((part) => {
      const result = resultInfo(part);
      return {
        tool_use_id: part.id,
        text: result.text,
        isError: result.isError,
      };
    });
    const text =
      message.role === 'user' || message.role === 'assistant'
        ? message.content.filter(isText).map((part) => part.text).join('\n')
        : '';

    if (text.length === 0 && toolUses.length === 0 && toolResults.length === 0) return [];

    const adapted: Message = {
      role: message.role === 'assistant' ? 'assistant' : 'user',
      text,
      toolUses,
    };
    if (toolResults.length > 0) adapted.toolResults = toolResults;
    return [adapted];
  });
}

/**
 * OpenCode V2's compaction override accepts a summary string rather than an
 * arbitrary replacement message list. Serialize the JEV-pruned transcript
 * deterministically, keeping retained user/assistant text and tool output
 * verbatim instead of asking another model to rewrite it.
 */
export function serializeCompaction(messages: readonly Message[]): string {
  const header =
    '[fast-jev-compaction checkpoint: retained conversation follows verbatim; no model summary was generated]';
  return messages.length === 0
    ? header
    : `${header}\n\n${serializeChunk(messages)}`;
}

function partCensus(messages: readonly NativeMessage[]): {
  roles: Record<string, number>;
  parts: Record<string, number>;
} {
  const roles: Record<string, number> = Object.create(null) as Record<string, number>;
  const parts: Record<string, number> = Object.create(null) as Record<string, number>;
  for (const message of messages) {
    roles[message.role] = (roles[message.role] ?? 0) + 1;
    for (const part of message.content) parts[part.type] = (parts[part.type] ?? 0) + 1;
  }
  return { roles, parts };
}

export function createAsker(config: FastJevConfig, logger: Logger): JevAsker {
  if (config.backend === 'systemone') {
    // Batches are classified concurrently; this is the node-wide cap on requests
    // in flight, so one compaction cannot oversubscribe the decision pool.
    const schedule = createScheduler(config.localConcurrency);
    const client = new SystemOneClient({
      baseUrl: config.localBaseUrl,
      model: config.localModel ?? '',
      apiKey: config.localApiKey,
      fetch: async (url, init) => {
        const started = Date.now();
        logger.info('classifier request', {
          method: init?.method ?? 'POST',
          url: safeUrl(String(url)),
        });
        try {
          const response = await fetch(url, init);
          logger.info('classifier response', {
            status: response.status,
            ok: response.ok,
            durationMs: Date.now() - started,
          });
          return response;
        } catch (error) {
          logger.warn('classifier transport error', {
            durationMs: Date.now() - started,
            ...errorSummary(error),
          });
          if (isRequestTimeout(error)) {
            throw new Error(
              `System One request timed out after ${DEFAULT_REQUEST_TIMEOUT_MS} ms`,
            );
          }
          throw new Error(
            `local System One unreachable at ${String(url)}: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      },
    });
    return {
      ask: (state, questions) => schedule(() => client.ask(state, questions)),
    };
  }
  if (config.backend === 'local') {
    return new LocalJevAsker({
      baseUrl: config.localBaseUrl,
      model: config.localModel,
      apiKey: config.localApiKey,
      concurrency: config.localConcurrency,
      contextTokens: config.localContextTokens,
      onDecision: (name, probability) => {
        logger.info('classifier decision', { question: name, noul: probability });
      },
      fetch: async (url, init) => {
        const started = Date.now();
        logger.info('classifier request', { method: init.method, url: safeUrl(url) });
        try {
          const response = await fetch(url, init);
          const text = await response.text();
          logger.info('classifier response', {
            status: response.status,
            ok: response.ok,
            durationMs: Date.now() - started,
          });
          return { status: response.status, ok: response.ok, text };
        } catch (error) {
          logger.warn('classifier transport error', {
            durationMs: Date.now() - started,
            ...errorSummary(error),
          });
          throw error;
        }
      },
    });
  }

  return new JevClient({
    apiKey: config.apiKey,
    model: config.model,
    baseUrl: config.baseUrl,
  });
}

export async function runOpenCodeCompaction(
  messages: readonly NativeMessage[],
  asker: JevAsker,
  config: FastJevConfig,
): Promise<CompactResult | undefined> {
  const transcript = toCompactionMessages(messages);
  if (transcript.length === 0) return;
  return compact(transcript, asker, config);
}

export async function compactOpenCodeMessages(
  messages: readonly NativeMessage[],
  asker: JevAsker,
  config: FastJevConfig,
): Promise<CompactResult | undefined> {
  const result = await runOpenCodeCompaction(messages, asker, config);
  if (!result || reductionRatio(result) < config.minReductionRatio) return;
  return result;
}

function percent(value: number): string {
  return Math.round(value * 100) + '%';
}

export const FastJevCompaction = Plugin.define({
  id: 'fast-jev-compaction-opencode',

  async setup(ctx) {
    const config = resolveConfig(ctx.options as PluginOptions);
    const resolved = resolveOptions(config);
    const logFile = config.logFile ?? defaultLogPath();
    const logger = createLogger(logFile);
    const asker = createAsker(config, logger);
    // The chunked fallback maps its part requests through this cap too, so a
    // summarizing compaction cannot oversubscribe the endpoint either.
    const fallbackSchedule = createScheduler(config.localConcurrency);
    const backendDetails =
      config.backend === 'local' || config.backend === 'systemone'
        ? {
            endpoint: safeUrl(
              config.backend === 'systemone'
                ? systemOneEndpoint(config.localBaseUrl ?? LOCAL_SYSTEM_ONE_BASE_URL)
                : (config.localBaseUrl ?? LOCAL_BASE_URL),
            ),
            model: config.localModel ?? (config.backend === 'systemone' ? '[required]' : LOCAL_MODEL),
            // Whether a bearer token is sent, never the token itself.
            apiKey: config.localApiKey ? 'set' : 'none',
          }
        : {};

    logger.info('initialized', {
      backend: config.backend,
      ...backendDetails,
      localConcurrency: config.localConcurrency,
      localContextTokens: config.localContextTokens,
      preserveRecentMessages: resolved.preserveRecentMessages,
      keepThreshold: resolved.keepThreshold,
      serverContextTokens: resolved.serverContextTokens,
      requestBudget: resolved.requestBudget,
      maxStateTokens: resolved.maxStateTokens,
      maxRequestTokens: resolved.maxRequestTokens,
      truncateHeadChars: resolved.truncateHeadChars,
      minReductionRatio: config.minReductionRatio,
      fallbackModel: config.fallbackModel,
      logFile,
    });

    await ctx.session.hook('compaction', async (event) => {
      const sessionID = event.sessionID;
      const nativeMessages = event.messages as unknown as NativeMessage[];
      let transcript: Message[] | undefined;

      /**
       * The plugin's chunked summarizer fallback: OpenCode core's compaction
       * summarizer has a finite window and would silently drop the oldest
       * history to fit, so an oversized transcript is summarized here instead.
       * Any failure leaves the fallback to core, as when `fallbackModel` is
       * unset.
       */
      const runChunkedFallback = async (): Promise<void> => {
        const adapted = transcript ?? toCompactionMessages(nativeMessages);
        if (!config.fallbackModel || adapted.length === 0) return;
        const transcriptTokens = estimateTokens(serializeCompaction(adapted));
        if (transcriptTokens <= Math.floor(config.fallbackContextTokens * CHUNK_SHARE)) return;
        try {
          const summary = await chunkedSummary(adapted, {
            baseUrl: config.fallbackBaseUrl ?? config.localBaseUrl ?? LOCAL_BASE_URL,
            model: config.fallbackModel,
            apiKey: config.fallbackApiKey ?? config.localApiKey,
            schedule: fallbackSchedule,
            fetch,
            contextTokens: config.fallbackContextTokens,
          });
          event.result = {
            summary: summary.summary,
            metadata: {
              plugin: 'fast-jev-compaction-opencode',
              backend: config.backend,
              fallback: 'chunked-summary',
              chunks: summary.chunkCount,
              model: config.fallbackModel,
              transcriptTokens,
              truncatedMessages: summary.truncatedMessages,
            },
          };
          logger.info('compaction outcome', {
            session: sessionID,
            outcome: 'override',
            fallback: 'chunked-summary',
            chunks: summary.chunkCount,
            truncatedMessages: summary.truncatedMessages,
            transcriptTokens,
          });
        } catch (error) {
          logger.warn('chunked fallback failed', {
            session: sessionID,
            ...errorSummary(error),
          });
        }
      };

      const shipOverride = (result: CompactResult, ratio: number, candidateCount: number, partial: boolean): void => {
        event.result = {
          summary: serializeCompaction(result.messages),
          metadata: {
            plugin: 'fast-jev-compaction-opencode',
            backend: config.backend,
            calls: result.stats.calls,
            candidateCalls: candidateCount,
            pinned: result.stats.pinned,
            jevBatches: result.stats.requests,
            kept: result.stats.kept,
            resultsDropped: result.stats.resultsDropped,
            callsDropped: result.stats.callsDropped,
            reductionRatio: ratio,
            ...(partial ? { partial: true } : {}),
          },
        };
        logger.info('compaction outcome', {
          session: sessionID,
          outcome: 'override',
          backend: config.backend,
          reduction: percent(ratio),
          ...(partial ? { partial: true } : {}),
        });
      };

      try {
        const census = partCensus(nativeMessages);
        logger.info('compaction hook entered', {
          session: sessionID,
          rawMessages: nativeMessages.length,
          roles: census.roles,
          parts: census.parts,
          backend: config.backend,
        });

        transcript = toCompactionMessages(nativeMessages);
        const calls = collectToolCalls(transcript, resolved.preserveRecentMessages);
        const candidateCalls = calls.filter((call) => !call.pinned).length;
        logger.info('transcript adapted', {
          session: sessionID,
          adaptedMessages: transcript.length,
          emptyDropped: nativeMessages.length - transcript.length,
          completedCalls: calls.length,
          pinned: calls.length - candidateCalls,
          candidates: candidateCalls,
        });

        if (transcript.length === 0) {
          logger.info('compaction outcome', {
            session: sessionID,
            outcome: 'fallback',
            reason: 'no-adaptable-messages',
          });
          return;
        }

        let result: CompactResult;
        try {
          result = await compact(transcript, asker, config);
        } catch (error) {
          if (!(error instanceof PartialCompactionError)) throw error;
          // A failed batch must not discard the batches that answered: the
          // result already applies every answered batch, the rest kept.
          result = error.result;
          const partialRatio = reductionRatio(result);
          if (partialRatio >= config.minReductionRatio) {
            logger.warn('partial compaction shipped', {
              session: sessionID,
              ...errorSummary(error),
              reduction: percent(partialRatio),
            });
            shipOverride(result, partialRatio, candidateCalls, true);
            return;
          }
          logger.warn('partial compaction below minimum', {
            session: sessionID,
            ...errorSummary(error),
            reduction: percent(partialRatio),
            minimum: percent(config.minReductionRatio),
          });
        }

        const ratio = reductionRatio(result);
        logger.info('classification complete', {
          session: sessionID,
          calls: result.stats.calls,
          candidates: Math.max(0, result.stats.calls - result.stats.pinned),
          pinned: result.stats.pinned,
          jevBatches: result.stats.requests,
          stateTokens: result.stats.stateTokens,
          stateStage: result.stats.stateStage,
          stateBuckets: result.stats.stateBuckets,
          reduction: percent(ratio),
        });

        if (ratio < config.minReductionRatio) {
          const reason =
            result.stats.requests === 0
              ? 'no-eligible-completed-tool-calls'
              : 'below-minimum-reduction';
          logger.info('compaction outcome', {
            session: sessionID,
            outcome: 'fallback',
            reason,
            reduction: percent(ratio),
            minimum: percent(config.minReductionRatio),
          });
          await runChunkedFallback();
          return;
        }

        shipOverride(result, ratio, candidateCalls, false);
      } catch (error) {
        logger.warn('compaction outcome', {
          session: sessionID,
          outcome: 'fallback',
          ...errorSummary(error),
        });
        await runChunkedFallback();
      }
    });
  },
});

export default FastJevCompaction;
