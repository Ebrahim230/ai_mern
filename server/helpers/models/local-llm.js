import Transformer from "./transformer.js";
import { tokenize, detokenize, vocab } from "./tokenizer.js";

let transformer = null;

export async function initLLM() {
  transformer = new Transformer(vocab);
}

export async function generateLocalResponse(prompt) {
  if (!transformer) throw new Error("Transformer not initialized. Call initLLM() first.");
  const tokens = tokenize(prompt.toLowerCase());
  const outputTokens = transformer.generate(tokens);
  return detokenize(outputTokens);
}