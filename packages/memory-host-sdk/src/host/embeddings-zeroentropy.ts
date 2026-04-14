import type { SsrFPolicy } from "../../../../src/infra/net/ssrf.js";
import { normalizeEmbeddingModelWithPrefixes } from "./embeddings-model-normalize.js";
import { resolveRemoteEmbeddingBearerClient } from "./embeddings-remote-client.js";
import { postJson } from "./post-json.js";
import type { EmbeddingProvider, EmbeddingProviderOptions } from "./embeddings.js";

export type ZeroEntropyEmbeddingClient = {
  baseUrl: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  model: string;
  outputDimensionality?: number;
};

export const DEFAULT_ZEROENTROPY_EMBEDDING_MODEL = "zembed-1";
const DEFAULT_ZEROENTROPY_BASE_URL = "https://api.zeroentropy.dev/v1";
const ZEROENTROPY_MAX_INPUT_TOKENS = 8192;

export function normalizeZeroEntropyModel(model: string): string {
  return normalizeEmbeddingModelWithPrefixes({
    model,
    defaultModel: DEFAULT_ZEROENTROPY_EMBEDDING_MODEL,
    prefixes: ["zeroentropy/"],
  });
}

function fetchZeroEntropyEmbeddingVectors(params: {
  url: string;
  headers: Record<string, string>;
  ssrfPolicy?: SsrFPolicy;
  body: unknown;
  errorPrefix: string;
}): Promise<number[][]> {
  return postJson({
    url: params.url,
    headers: params.headers,
    ssrfPolicy: params.ssrfPolicy,
    body: params.body,
    errorPrefix: params.errorPrefix,
    parse: (payload) => {
      const typedPayload = payload as {
        results?: Array<{ embedding?: number[] }>;
      };
      const results = typedPayload.results ?? [];
      return results.map((entry) => entry.embedding ?? []);
    },
  });
}

export async function createZeroEntropyEmbeddingProvider(
  options: EmbeddingProviderOptions,
): Promise<{ provider: EmbeddingProvider; client: ZeroEntropyEmbeddingClient }> {
  const client = await resolveZeroEntropyEmbeddingClient(options);
  const url = `${client.baseUrl.replace(/\/$/, "")}/models/embed`;

  const embed = async (
    input: string[],
    input_type?: "query" | "document",
  ): Promise<number[][]> => {
    if (input.length === 0) {
      return [];
    }
    const body: {
      model: string;
      input: string[];
      input_type?: "query" | "document";
      dimensions?: number;
      encoding_format: string;
    } = {
      model: client.model,
      input,
      encoding_format: "float",
    };
    if (input_type) {
      body.input_type = input_type;
    }
    if (client.outputDimensionality) {
      body.dimensions = client.outputDimensionality;
    }

    return await fetchZeroEntropyEmbeddingVectors({
      url,
      headers: client.headers,
      ssrfPolicy: client.ssrfPolicy,
      body,
      errorPrefix: "zeroentropy embeddings failed",
    });
  };

  return {
    provider: {
      id: "zeroentropy",
      model: client.model,
      maxInputTokens: ZEROENTROPY_MAX_INPUT_TOKENS,
      embedQuery: async (text) => {
        const [vec] = await embed([text], "query");
        return vec ?? [];
      },
      embedBatch: async (texts) => embed(texts, "document"),
    },
    client,
  };
}

export async function resolveZeroEntropyEmbeddingClient(
  options: EmbeddingProviderOptions,
): Promise<ZeroEntropyEmbeddingClient> {
  const { baseUrl, headers, ssrfPolicy } = await resolveRemoteEmbeddingBearerClient({
    provider: "zeroentropy",
    options,
    defaultBaseUrl: DEFAULT_ZEROENTROPY_BASE_URL,
  });
  const model = normalizeZeroEntropyModel(options.model);
  return {
    baseUrl,
    headers,
    ssrfPolicy,
    model,
    outputDimensionality: options.outputDimensionality,
  };
}
