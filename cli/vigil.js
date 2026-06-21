#!/usr/bin/env node
/*
 * Vigil CLI — `vigil init | demo | prompt`.
 *
 * Dependency-free (Node built-ins only). `init` prints the one-line integration; `prompt` prints a
 * copy-paste prompt for an AI coding agent; `demo` runs a fully self-contained sabotage scenario
 * (a scripted looping upstream) against a running proxy and shows the breaker trip + cost freeze.
 */
"use strict";

const http = require("http");

const DEFAULT_PROXY = process.env.VIGIL_PROXY || "http://localhost:8765";

// The canonical "give this to your AI coding agent" prompt — keep in sync with the README and the
// dashboard Docs page.
const AGENT_PROMPT = `Integrate Vigil into this codebase. Vigil is a transparent LLM proxy that
watches an agent's trajectory and halts runaway loops before they burn budget. It speaks the
OpenAI and Anthropic APIs and forwards to the real provider, so the ONLY change needed is the
client's base URL — do not change prompts, models, keys, or any other logic.

Steps:
1. Find every place an OpenAI or Anthropic client is constructed (e.g. OpenAI(), AsyncOpenAI(),
   Anthropic(), LangChain/LlamaIndex model configs, or a raw base_url/baseURL setting).
2. Point base_url at the Vigil proxy, read from an env var so it is easy to toggle:
     OpenAI:    base_url = os.environ.get("VIGIL_BASE_URL", "http://localhost:8765/v1")
     Anthropic: base_url = os.environ.get("VIGIL_BASE_URL", "http://localhost:8765")
3. Leave the API key exactly as-is — Vigil passes it straight through and never stores it.
4. Recommended: set a per-run session id header so Vigil groups one agent run into one trajectory:
     default_headers={"x-vigil-session-id": "<a stable id for this run>"}
5. Verify: \`curl http://localhost:8765/health\` returns {"status":"ok"}; after a request,
   \`curl http://localhost:8765/metrics/session/<id>\` shows the captured steps.

Make the minimal edit, keep it behind the VIGIL_BASE_URL env var, and do not alter behavior.`;

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function request(method, url, body, headers) {
  return new Promise((resolve) => {
    const u = new URL(url);
    const data = body ? Buffer.from(JSON.stringify(body)) : null;
    const req = http.request(
      {
        method,
        hostname: u.hostname,
        port: u.port,
        path: u.pathname + u.search,
        headers: Object.assign(
          data ? { "content-type": "application/json", "content-length": data.length } : {},
          headers || {}
        ),
      },
      (res) => {
        let buf = "";
        res.on("data", (c) => (buf += c));
        res.on("end", () => {
          let json = null;
          try { json = JSON.parse(buf); } catch (_) {}
          resolve({ status: res.statusCode, json, raw: buf });
        });
      }
    );
    req.on("error", () => resolve({ status: 0, json: null, raw: "" }));
    if (data) req.write(data);
    req.end();
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------- init / prompt

function cmdInit() {
  const base = arg("--base-url", "http://localhost:8765");
  console.log(`\nVigil — one-line integration\n${"=".repeat(40)}\n`);
  console.log("Change ONLY your client's base_url. Your API key is passed straight through.\n");
  console.log("Python (OpenAI):");
  console.log("  from openai import OpenAI");
  console.log(`  client = OpenAI(base_url="${base}/v1", api_key="sk-...")\n`);
  console.log("Python (Anthropic):");
  console.log("  from anthropic import Anthropic");
  console.log(`  client = Anthropic(base_url="${base}")\n`);
  console.log("Node (OpenAI):");
  console.log("  import OpenAI from 'openai';");
  console.log(`  const client = new OpenAI({ baseURL: '${base}/v1', apiKey: process.env.OPENAI_API_KEY });\n`);
  console.log(`Health check:  curl ${base}/health\n`);
  console.log("Then watch the dashboard (packages/dashboard: npm run dev) or:");
  console.log(`  curl ${base}/metrics/aggregate\n`);
  console.log("Integrating with an AI coding agent? Run:  vigil prompt\n");
}

function cmdPrompt() {
  console.log(AGENT_PROMPT);
}

// --------------------------------------------------------------------------- demo

function startMockUpstream(port) {
  // A scripted looping upstream: every call returns the SAME tool call -> a tight loop.
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const payload = JSON.stringify({
        id: "chatcmpl-demo",
        object: "chat.completion",
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                { id: "call_1", type: "function",
                  function: { name: "check_status", arguments: '{"job":"deploy-1"}' } },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 140, completion_tokens: 18 },
      });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(payload);
    });
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

async function cmdDemo() {
  const proxy = arg("--proxy", DEFAULT_PROXY);
  const steps = parseInt(arg("--steps", "10"), 10);
  const mockPort = parseInt(arg("--mock-port", "8799"), 10);
  const sid = "vigil-demo";

  let server, port;
  try {
    ({ server, port } = await startMockUpstream(mockPort));
  } catch (_) {
    console.log(`\nMock-upstream port ${mockPort} is busy. Pass a free one: vigil demo --mock-port <port>\n`);
    return;
  }
  const upstream = `http://127.0.0.1:${port}/v1`;
  console.log(`\nVigil demo — scripted runaway loop\n${"=".repeat(40)}\n`);
  console.log(`Started a looping mock upstream at ${upstream}`);
  console.log("\nIf the proxy isn't already pointed at it, start it in another terminal:\n");
  console.log(`  OPENAI_BASE_URL=${upstream} VIGIL_EMBED_HASHING=true \\`);
  console.log("    uv run uvicorn vigil_proxy.app:app --port 8765\n");
  process.stdout.write("Waiting for the proxy");

  let up = false;
  for (let i = 0; i < 60; i++) {
    const h = await request("GET", `${proxy}/health`);
    if (h.status === 200) { up = true; break; }
    process.stdout.write(".");
    await sleep(1000);
  }
  console.log("");
  if (!up) {
    console.log(`\nNo proxy at ${proxy} yet. Start it with the command above, then re-run \`vigil demo\`.`);
    server.close();
    return;
  }

  let tripped = false;
  for (let i = 1; i <= steps; i++) {
    const r = await request(
      "POST",
      `${proxy}/v1/chat/completions`,
      { model: "gpt-4o", messages: [{ role: "user", content: "keep checking the deploy status" }] },
      { authorization: "Bearer sk-demo", "x-vigil-session-id": sid }
    );
    await sleep(150); // let the async capture/analysis land
    const b = await request("GET", `${proxy}/sessions/${sid}/breaker`);
    const state = (b.json && b.json.state) || "CLOSED";
    const blocked = r.status === 503;
    console.log(`  step ${String(i).padStart(2)}  →  breaker ${state}${blocked ? "  (request blocked)" : ""}`);
    if (state === "OPEN" || blocked) {
      tripped = true;
      const saved = (b.json && b.json.saved_estimate_usd) || 0;
      console.log(`\n  ⛔ Loop halted. Projected cost capped — ~$${saved.toFixed(4)} saved.`);
      break;
    }
  }

  const m = await request("GET", `${proxy}/metrics/session/${sid}`);
  if (m.json) {
    console.log(`\nSession ${sid}: ${m.json.steps} steps, $${(m.json.cost_usd || 0).toFixed(5)} spend.`);
  }
  if (!tripped) {
    console.log("\nLoop did not trip in this run — try more --steps, or check the watchdog thresholds.");
  }
  console.log("\nReset the breaker with:  curl -X POST " + `${proxy}/sessions/${sid}/override\n`);
  server.close();
}

// --------------------------------------------------------------------------- main

function help() {
  console.log(`vigil — setup and demo for the Vigil LLM proxy

  vigil init [--base-url URL]    print the one-line base_url integration
  vigil prompt                   print a copy-paste prompt for an AI coding agent
  vigil demo [--proxy URL] [--steps N]   run a scripted loop and watch the breaker trip
`);
}

async function main() {
  const cmd = process.argv[2];
  if (cmd === "init") cmdInit();
  else if (cmd === "prompt") cmdPrompt();
  else if (cmd === "demo") await cmdDemo();
  else help();
}

main();
