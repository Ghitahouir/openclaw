import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import * as authModule from "../../agents/model-auth.js";
import {
  createJsonResponseFetchMock,
  installFetchMock,
  mockResolvedProviderKey,
  type JsonFetchMock,
} from "./embeddings-provider.test-support.js";
import { mockPublicPinnedHostname } from "./test-helpers/ssrf.js";

vi.mock("../../agents/model-auth.js", async () => {
  const { createModelAuthMockModule } = await import("../../test-utils/model-auth-mock.js");
  return createModelAuthMockModule();
});

let createZeroEntropyEmbeddingProvider: typeof import("./embeddings-zeroentropy.js").createZeroEntropyEmbeddingProvider;
let normalizeZeroEntropyModel: typeof import("./embeddings-zeroentropy.js").normalizeZeroEntropyModel;

beforeAll(async () => {
  ({ createZeroEntropyEmbeddingProvider, normalizeZeroEntropyModel } =
    await import("./embeddings-zeroentropy.js"));
});

beforeEach(() => {
  vi.useRealTimers();
  vi.doUnmock("undici");
});

function createZeroEntropyFetchMock(embeddingValues = [0.1, 0.2, 0.3]) {
  return createJsonResponseFetchMock({ results: [{ embedding: embeddingValues }] });
}

async function createDefaultZeroEntropyProvider(model: string, fetchMock: JsonFetchMock) {
  installFetchMock(fetchMock as unknown as typeof globalThis.fetch);
  mockPublicPinnedHostname();
  mockResolvedProviderKey(authModule.resolveApiKeyForProvider, "ze-key-123");
  return createZeroEntropyEmbeddingProvider({
    config: {} as never,
    provider: "zeroentropy",
    model,
    fallback: "none",
  });
}

describe("zeroentropy embedding provider", () => {
  afterEach(() => {
    vi.doUnmock("undici");
    vi.resetAllMocks();
    vi.unstubAllGlobals();
  });

  it("configures client with correct defaults and headers", async () => {
    const fetchMock = createZeroEntropyFetchMock();
    const result = await createDefaultZeroEntropyProvider("zembed-1", fetchMock);

    await result.provider.embedQuery("test query");

    expect(authModule.resolveApiKeyForProvider).toHaveBeenCalledWith(
      expect.objectContaining({ provider: "zeroentropy" }),
    );

    const call = fetchMock.mock.calls[0];
    expect(call).toBeDefined();
    const [url, init] = call as [RequestInfo | URL, RequestInit | undefined];
    expect(url).toBe("https://api.zeroentropy.dev/v1/models/embed");

    const headers = (init?.headers ?? {}) as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer ze-key-123");
    expect(headers["Content-Type"]).toBe("application/json");

    const body = JSON.parse(init?.body as string);
    expect(body).toEqual({
      model: "zembed-1",
      input: ["test query"],
      input_type: "query",
      encoding_format: "float",
    });
  });

  it("respects remote overrides for baseUrl and apiKey", async () => {
    const fetchMock = createZeroEntropyFetchMock();
    installFetchMock(fetchMock as unknown as typeof globalThis.fetch);
    mockPublicPinnedHostname();

    const result = await createZeroEntropyEmbeddingProvider({
      config: {} as never,
      provider: "zeroentropy",
      model: "zembed-1",
      fallback: "none",
      remote: {
        baseUrl: "https://example.com",
        apiKey: "remote-override-key",
        headers: { "X-Custom": "123" },
      },
    });

    await result.provider.embedQuery("test");

    const call = fetchMock.mock.calls[0];
    expect(call).toBeDefined();
    const [url, init] = call as [RequestInfo | URL, RequestInit | undefined];
    expect(url).toBe("https://example.com/models/embed");

    const headers = (init?.headers ?? {}) as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer remote-override-key");
    expect(headers["X-Custom"]).toBe("123");
  });

  it("passes input_type=document for embedBatch", async () => {
    const fetchMock = createJsonResponseFetchMock({
      results: [{ embedding: [0.1, 0.2] }, { embedding: [0.3, 0.4] }],
    });
    const result = await createDefaultZeroEntropyProvider("zembed-1", fetchMock);

    await result.provider.embedBatch(["doc1", "doc2"]);

    const call = fetchMock.mock.calls[0];
    expect(call).toBeDefined();
    const [, init] = call as [RequestInfo | URL, RequestInit | undefined];
    const body = JSON.parse(init?.body as string);
    expect(body).toEqual({
      model: "zembed-1",
      input: ["doc1", "doc2"],
      input_type: "document",
      encoding_format: "float",
    });
  });

  it("includes dimensions when outputDimensionality is set", async () => {
    const fetchMock = createZeroEntropyFetchMock();
    installFetchMock(fetchMock as unknown as typeof globalThis.fetch);
    mockPublicPinnedHostname();
    mockResolvedProviderKey(authModule.resolveApiKeyForProvider, "ze-key-123");

    const result = await createZeroEntropyEmbeddingProvider({
      config: {} as never,
      provider: "zeroentropy",
      model: "zembed-1",
      fallback: "none",
      outputDimensionality: 1280,
    });

    await result.provider.embedQuery("test");

    const call = fetchMock.mock.calls[0];
    expect(call).toBeDefined();
    const [, init] = call as [RequestInfo | URL, RequestInit | undefined];
    const body = JSON.parse(init?.body as string);
    expect(body.dimensions).toBe(1280);
  });

  it("returns empty array for empty batch", async () => {
    const fetchMock = createZeroEntropyFetchMock();
    const result = await createDefaultZeroEntropyProvider("zembed-1", fetchMock);

    const embeddings = await result.provider.embedBatch([]);
    expect(embeddings).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("normalizes model names", async () => {
    expect(normalizeZeroEntropyModel("zeroentropy/zembed-1")).toBe("zembed-1");
    expect(normalizeZeroEntropyModel("zembed-1")).toBe("zembed-1");
    expect(normalizeZeroEntropyModel("  zembed-1  ")).toBe("zembed-1");
    expect(normalizeZeroEntropyModel("")).toBe("zembed-1"); // Default
  });

  it("sets maxInputTokens", async () => {
    const fetchMock = createZeroEntropyFetchMock();
    const result = await createDefaultZeroEntropyProvider("zembed-1", fetchMock);
    expect(result.provider.maxInputTokens).toBe(8192);
  });

  it("parses ZeroEntropy response format (results array)", async () => {
    const fetchMock = createJsonResponseFetchMock({
      results: [{ embedding: [0.5, 0.6, 0.7] }],
      usage: { total_bytes: 100, total_tokens: 10 },
    });
    const result = await createDefaultZeroEntropyProvider("zembed-1", fetchMock);

    const embedding = await result.provider.embedQuery("hello");
    expect(embedding).toEqual([0.5, 0.6, 0.7]);
  });
});
