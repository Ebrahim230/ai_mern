function randn(shape) {
  return Array.from({ length: shape[0] }, () => Math.random() * 2 - 1);
}

function matmul(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) {
    const row = [];
    for (let j = 0; j < b[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < b.length; k++) {
        sum += a[i][k] * b[k][j];
      }
      row.push(sum);
    }
    out.push(row);
  }
  return out;
}

function add(a, b) {
  return a.map((r, i) => r.map((v, j) => v + b[i][j]));
}

function relu(x) {
  return x.map(r => r.map(v => Math.max(0, v)));
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

export default class Transformer {
  constructor(vocab, d_model = 32, d_ff = 64) {
    this.vocab = vocab;
    this.vocabSize = Object.keys(vocab).length;
    this.d_model = d_model;
    this.d_ff = d_ff;

    this.W1 = randn([d_model, d_ff]);
    this.b1 = randn([1, d_ff]);
    this.W2 = randn([d_ff, this.vocabSize]);
    this.b2 = randn([1, this.vocabSize]);
    this.E = randn([this.vocabSize, d_model]);
  }

  forward(input) {
    const x = input.map(i => this.E[i]);
    const sumVec = x.reduce((a, b) => a.map((v, i) => v + b[i]));
    const h1 = relu(add([sumVec], this.b1).map(v => matmul([v], this.W1)[0]))[0];
    const logits = add([matmul([h1], this.W2)[0]], this.b2)[0];
    return softmax(logits);
  }

  generate(prompt, maxLen = 20) {
    const input = [...prompt];
    for (let i = 0; i < maxLen; i++) {
      const probs = this.forward(input);
      let next = probs.indexOf(Math.max(...probs));
      input.push(next);
      if (this.vocab["."] && next === this.vocab["."]) break;
    }
    return input;
  }
}