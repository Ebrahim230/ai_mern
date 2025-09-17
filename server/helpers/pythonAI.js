import { spawn } from "child_process";
import readline from "readline";

const pythonProcess = spawn("python3", ["./helpers/selfTrainedAI.py"]);

const rl = readline.createInterface({
  input: pythonProcess.stdout,
  output: pythonProcess.stdin,
  terminal: false,
});

let pendingResolve = null;

rl.on("line", (line) => {
  if (pendingResolve) {
    pendingResolve(line);
    pendingResolve = null;
  }
});

pythonProcess.stderr.on("data", (data) => {
  console.error("[Python AI error]:", data.toString());
});

function sendPrompt(prompt) {
  return new Promise((resolve) => {
    pendingResolve = resolve;
    pythonProcess.stdin.write(prompt + "\n");
  });
}

export default {
  init: () => Promise.resolve(),
  generateResponse: sendPrompt,
};