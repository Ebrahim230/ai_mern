export const vocab = {};
export const ivocab = {};
let id = 0;

function getTokenId(token) {
  if (!(token in vocab)) {
    vocab[token] = id;
    ivocab[id] = token;
    id++;
  }
  return vocab[token];
}

export function tokenize(text) {
  return text.split(/\s+/).map(getTokenId);
}

export function detokenize(tokens) {
  return tokens.map(t => ivocab[t]).join(" ");
}