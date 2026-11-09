import { describe, expect, it } from 'vitest';

import { classifyCalls, compact } from '../src/compact.js';
import { collectToolCalls, fitState, fitStateBuckets, isPinned } from '../src/state.js';
import type { JevQuestions, JevResponse, JevState, Message } from '../src/types.js';

function prompt(text: string): Message {
  return { role: 'user', text, toolUses: [] };
}

function callBlock(id: string, tool: string, input: Record<string, unknown>): Message {
  return { role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool, input }] };
}

function resultBlock(id: string, text: string): Message {
  return { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text }] };
}

/** `prompt → tool call → tool result` repeated `blocks` times, after an opener. */
function transcript(
  blocks: number,
  input: Record<string, unknown> = { path: 'src/module.ts' },
): Message[] {
  const messages: Message[] = [prompt('Start the migration work')];
  for (let i = 1; i <= blocks; i++) {
    messages.push(prompt(`Step ${i}: continue the migration plan with care`));
    messages.push(callBlock(`c${i}`, 'read', input));
    messages.push(resultBlock(`c${i}`, `output ${i}: ${'data '.repeat(20)}`));
  }
  return messages;
}

interface AskerLogEntry {
  state: JevState;
  questions: JevQuestions;
}

function dropAllAsker(log: AskerLogEntry[] = []) {
  return {
    async ask(state: JevState, questions: JevQuestions): Promise<JevResponse> {
      log.push({ state, questions });
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((name) => [name, { noul: 0.1 }]),
        ),
      };
    },
  };
}

describe('fitStateBuckets — unsliced', () => {
  it('returns exactly one ladder-built bucket when the whole history fits', () => {
    const messages = transcript(2);
    const options = { maxStateTokens: 2000, preserveRecentMessages: 0 };
    const calls = collectToolCalls(messages, options.preserveRecentMessages);

    const { buckets, sliced } = fitStateBuckets(messages, calls, options);
    expect(sliced).toBe(false);
    expect(buckets).toHaveLength(1);
    const fitted = fitState(messages, calls, options);
    expect(buckets[0]!.state).toEqual(fitted.state);
    expect(buckets[0]!.tokens).toBe(fitted.tokens);
    expect(buckets[0]!.stage).toBe(fitted.stage);
    expect(buckets[0]!.candidates).toEqual(calls.filter((call) => !call.pinned));
  });
});

describe('fitStateBuckets — sliced', () => {
  const options = { maxStateTokens: 500, preserveRecentMessages: 6 };
  const messages = transcript(40);
  const calls = collectToolCalls(messages, options.preserveRecentMessages);
  const { buckets, sliced } = fitStateBuckets(messages, calls, options);

  it('splits a history that cannot fit into at least two buckets', () => {
    // The whole-history ladder cannot shrink 40 calls into 500 tokens.
    expect(() => fitState(messages, calls, options)).toThrow(/too large/);
    expect(sliced).toBe(true);
    expect(buckets.length).toBeGreaterThanOrEqual(2);
  });

  it('fits every bucket state within maxStateTokens', () => {
    for (const bucket of buckets) {
      expect(bucket.tokens).toBeLessThanOrEqual(options.maxStateTokens);
    }
  });

  it('places every non-pinned candidate in exactly one bucket, in order', () => {
    const expected = calls.filter((call) => !call.pinned).map((call) => call.tool_use_id);
    const placed = buckets.flatMap((bucket) => bucket.candidates.map((call) => call.tool_use_id));
    expect(placed).toEqual(expected);
  });

  it('never offers a pinned call as a candidate, judging by original indices', () => {
    expect(calls.some((call) => call.pinned)).toBe(true);
    const total = messages.length;
    for (const bucket of buckets) {
      for (const call of bucket.candidates) {
        expect(isPinned(call.callIndex, total, options.preserveRecentMessages)).toBe(false);
        expect(isPinned(call.resultIndex, total, options.preserveRecentMessages)).toBe(false);
      }
    }
  });

  it('cuts at turn boundaries, so later buckets open with a user prompt', () => {
    for (const bucket of buckets.slice(1)) {
      const first = bucket.state.history[0]!;
      expect(first.role).toBe('user');
      expect(first.text.length).toBeGreaterThan(0);
      expect(first.tool_calls).toBeUndefined();
    }
  });

  it('keeps the goal in every bucket state', () => {
    const goals = new Set(buckets.map((bucket) => bucket.state.goal));
    expect(goals.size).toBe(1);
    expect(buckets[0]!.state.goal).toContain('Step 40');
  });

  it('keeps the pinned first message verbatim in the first bucket', () => {
    expect(buckets[0]!.state.history[0]).toMatchObject({
      role: 'user',
      text: 'Start the migration work',
    });
  });

  it('degrades a single oversized message through the per-bucket ladder', () => {
    // A mid-transcript prompt far larger than the per-bucket target gets its
    // own bucket, which the ladder then abridges.
    const bulky = transcript(40);
    bulky[58] = prompt('giant status report '.repeat(300));
    const bulkyCalls = collectToolCalls(bulky, options.preserveRecentMessages);
    const { buckets: bulkyBuckets, sliced: bulkySliced } = fitStateBuckets(
      bulky,
      bulkyCalls,
      options,
    );
    expect(bulkySliced).toBe(true);
    const degraded = bulkyBuckets.filter((bucket) => bucket.stage !== 'full');
    expect(degraded.length).toBeGreaterThan(0);
    for (const bucket of degraded) {
      expect(bucket.tokens).toBeLessThanOrEqual(options.maxStateTokens);
      expect(JSON.stringify(bucket.state)).toContain('chars omitted');
    }
  });
});

describe('classifyCalls over sliced buckets', () => {
  it('asks each candidate exactly once, against more than one state', async () => {
    const log: AskerLogEntry[] = [];
    const messages = transcript(40);
    const classification = await classifyCalls(messages, dropAllAsker(log), {
      maxStateTokens: 500,
      preserveRecentMessages: 6,
    });

    expect(classification.stateBuckets).toBeGreaterThan(1);
    const asked = log.flatMap((entry) => Object.keys(entry.questions));
    expect(asked).toHaveLength(classification.candidates * 2);
    expect(new Set(asked).size).toBe(asked.length);
    expect(new Set(log.map((entry) => JSON.stringify(entry.state))).size).toBeGreaterThan(1);
  });

  it('applies every bucket decision to the full transcript', async () => {
    const messages = transcript(40);
    const result = await compact(messages, dropAllAsker(), {
      maxStateTokens: 500,
      preserveRecentMessages: 6,
    });
    expect(result.stats.stateBuckets).toBeGreaterThan(1);
    expect(result.stats.callsDropped).toBeGreaterThan(0);
    expect(result.stats.pinned).toBeGreaterThan(0);
  });
});
