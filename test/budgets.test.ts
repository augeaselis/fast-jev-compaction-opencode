import { describe, expect, it } from 'vitest';

import {
  REQUEST_MARGIN,
  TOKENS_PER_CALL,
  questionsFor,
  resolveOptions,
} from '../src/compact.js';
import { resolveConfig } from '../src/plugin.js';
import { estimateTokens } from '../src/state.js';
import type { ToolCall } from '../src/types.js';

function callOf(
  id: string,
  tool: string,
  input: Record<string, unknown>,
  resultChars: number,
): ToolCall {
  return {
    id,
    tool_use_id: id,
    tool,
    input,
    callIndex: 0,
    resultIndex: 1,
    resultChars,
    isError: false,
    pinned: false,
  };
}

describe('serverContextTokens budget derivation', () => {
  it('derives the request budget and caps state and request tokens', () => {
    const resolved = resolveOptions({
      serverContextTokens: 12288,
      maxStateTokens: 8000,
      maxRequestTokens: 12500,
      maxCallsPerBatch: 32,
    });
    expect(resolved.requestBudget).toBe(Math.floor(12288 * (1 - REQUEST_MARGIN)));
    expect(resolved.requestBudget).toBe(11059);
    expect(resolved.maxRequestTokens).toBe(11059);
    // 11059 minus the 32-call question reserve (32 * 130).
    expect(resolved.maxStateTokens).toBe(6899);
  });

  it('lets explicit values lower but never raise the derived caps', () => {
    const resolved = resolveOptions({
      serverContextTokens: 12288,
      maxStateTokens: 1000,
      maxRequestTokens: 9000,
      maxCallsPerBatch: 32,
    });
    expect(resolved.maxStateTokens).toBe(1000);
    expect(resolved.maxRequestTokens).toBe(9000);
  });

  it('leaves the options untouched when no server ceiling is configured', () => {
    const resolved = resolveOptions({ maxStateTokens: 8000, maxRequestTokens: 12500 });
    expect(resolved.serverContextTokens).toBeUndefined();
    expect(resolved.requestBudget).toBeUndefined();
    expect(resolved.maxStateTokens).toBe(8000);
    expect(resolved.maxRequestTokens).toBe(12500);
  });

  it('carries serverContextTokens and fallback options through the plugin config', () => {
    const config = resolveConfig({
      backend: 'systemone',
      serverContextTokens: 12288,
      fallbackModel: 'small_summarizer',
    });
    expect(config.serverContextTokens).toBe(12288);
    expect(config.fallbackModel).toBe('small_summarizer');
    expect(config.fallbackContextTokens).toBe(131072);
  });
});

describe('TOKENS_PER_CALL', () => {
  it('bounds the two noul questions of a realistic call', () => {
    const calls = [
      callOf('t1', 'read', { filePath: 'src/deeply/nested/module.ts' }, 45_000),
      callOf('t2', 'edit', { filePath: 'a.ts', oldText: 'x'.repeat(200), newText: 'y'.repeat(300) }, 1_200_000),
      callOf('t3', 'bash', { command: 'npm test -- --filter=very-long-suite-name' }, 12_345),
      callOf('t4', 'notebookedit', { notebookPath: 'analysis.ipynb' }, 9_999_999),
      callOf('t5', 'todowrite', { todos: [] }, 1),
    ];
    const measured = Math.max(
      ...calls.map((c) => estimateTokens(JSON.stringify(questionsFor(c)))),
    );
    // The probe's purpose is to report the measured bound next to the constant.
    console.info(`measured TOKENS_PER_CALL bound: ${measured}`);
    expect(measured).toBeLessThanOrEqual(TOKENS_PER_CALL);
  });
});
