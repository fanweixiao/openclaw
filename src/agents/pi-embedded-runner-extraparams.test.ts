import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { Context, Model, SimpleStreamOptions } from "@mariozechner/pi-ai";
import { AssistantMessageEventStream } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { applyExtraParamsToAgent, resolveExtraParams } from "./pi-embedded-runner.js";

describe("resolveExtraParams", () => {
  it("returns undefined with no model config", () => {
    const result = resolveExtraParams({
      cfg: undefined,
      provider: "zai",
      modelId: "glm-4.7",
    });

    expect(result).toBeUndefined();
  });

  it("returns params for exact provider/model key", () => {
    const result = resolveExtraParams({
      cfg: {
        agents: {
          defaults: {
            models: {
              "openai/gpt-4": {
                params: {
                  temperature: 0.7,
                  maxTokens: 2048,
                },
              },
            },
          },
        },
      },
      provider: "openai",
      modelId: "gpt-4",
    });

    expect(result).toEqual({
      temperature: 0.7,
      maxTokens: 2048,
    });
  });

  it("ignores unrelated model entries", () => {
    const result = resolveExtraParams({
      cfg: {
        agents: {
          defaults: {
            models: {
              "openai/gpt-4": {
                params: {
                  temperature: 0.7,
                },
              },
            },
          },
        },
      },
      provider: "openai",
      modelId: "gpt-4.1-mini",
    });

    expect(result).toBeUndefined();
  });
});

describe("applyExtraParamsToAgent", () => {
  it("adds OpenRouter attribution headers to stream options", () => {
    const calls: Array<SimpleStreamOptions | undefined> = [];
    const baseStreamFn: StreamFn = (_model, _context, options) => {
      calls.push(options);
      return new AssistantMessageEventStream();
    };
    const agent = { streamFn: baseStreamFn };

    applyExtraParamsToAgent(agent, undefined, "openrouter", "openrouter/auto");

    const model = {
      api: "openai-completions",
      provider: "openrouter",
      id: "openrouter/auto",
    } as Model<"openai-completions">;
    const context: Context = { messages: [] };

    void agent.streamFn?.(model, context, { headers: { "X-Custom": "1" } });

    expect(calls).toHaveLength(1);
    expect(calls[0]?.headers).toEqual({
      "HTTP-Referer": "https://openclaw.ai",
      "X-Title": "OpenClaw",
      "X-Custom": "1",
    });
  });

  it("normalizes qianfan assistant text-part arrays to plain strings in payload", () => {
    let capturedOptions: SimpleStreamOptions | undefined;
    const onPayloadCalls: unknown[] = [];

    const baseStreamFn: StreamFn = (_model, _context, options) => {
      capturedOptions = options;
      return new AssistantMessageEventStream();
    };
    const agent = { streamFn: baseStreamFn };

    applyExtraParamsToAgent(agent, undefined, "qianfan", "deepseek-v3.2");

    const model = {
      api: "openai-completions",
      provider: "qianfan",
      id: "deepseek-v3.2",
    } as Model<"openai-completions">;
    const context: Context = { messages: [] };

    void agent.streamFn?.(model, context, {
      onPayload: (payload) => onPayloadCalls.push(payload),
    });

    const payload = {
      messages: [
        {
          role: "assistant",
          content: [
            { type: "text", text: "hello " },
            { type: "text", text: "world" },
          ],
        },
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "x", arguments: "{}" } }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "unchanged" }],
        },
      ],
    };

    capturedOptions?.onPayload?.(payload);

    expect(payload.messages[0]?.content).toBe("hello world");
    expect(payload.messages[1]?.content).toBeNull();
    expect(payload.messages[2]?.content).toEqual([{ type: "text", text: "unchanged" }]);
    expect(onPayloadCalls).toEqual([payload]);
  });

  it("normalizes vivgrid assistant text-part arrays to plain strings in payload", () => {
    let capturedOptions: SimpleStreamOptions | undefined;
    const onPayloadCalls: unknown[] = [];

    const baseStreamFn: StreamFn = (_model, _context, options) => {
      capturedOptions = options;
      return new AssistantMessageEventStream();
    };
    const agent = { streamFn: baseStreamFn };

    applyExtraParamsToAgent(agent, undefined, "vivgrid", "auto");

    const model = {
      api: "openai-completions",
      provider: "vivgrid",
      id: "auto",
    } as Model<"openai-completions">;
    const context: Context = { messages: [] };

    void agent.streamFn?.(model, context, {
      onPayload: (payload) => onPayloadCalls.push(payload),
    });

    const payload = {
      messages: [
        {
          role: "assistant",
          content: [
            { type: "text", text: "alpha " },
            { type: "text", text: "beta" },
          ],
        },
      ],
    };

    capturedOptions?.onPayload?.(payload);

    expect(payload.messages[0]?.content).toBe("alpha beta");
    expect(onPayloadCalls).toEqual([payload]);
  });
});
