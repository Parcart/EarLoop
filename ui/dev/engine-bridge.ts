import path from "path";
import { randomUUID } from "crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import { createInterface, type Interface } from "readline";

import type { IncomingMessage, ServerResponse } from "http";
import type { Plugin, ViteDevServer } from "vite";

type BridgeCommand = {
  request_id: string;
  command: string;
  payload: Record<string, unknown>;
};

type PythonResponse = {
  request_id: string;
  ok: boolean;
  result?: unknown;
  error?: {
    code?: string;
    message?: string;
    details?: Record<string, unknown>;
  };
};

type PendingRequest = {
  resolve: (value: PythonResponse) => void;
  reject: (reason?: unknown) => void;
  timeout: NodeJS.Timeout;
};

function joinPythonPath(repoRoot: string): string {
  const srcPath = path.join(repoRoot, "src");
  const existing = process.env.PYTHONPATH;
  return existing ? `${srcPath}${path.delimiter}${existing}` : srcPath;
}

function readJsonBody(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
    req.on("end", () => {
      try {
        const raw = Buffer.concat(chunks).toString("utf-8").trim();
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on("error", reject);
  });
}

function writeJson(res: ServerResponse, statusCode: number, payload: unknown): void {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

class PythonProcessManager {
  private readonly repoRoot: string;
  private readonly pythonExecutable: string;
  private readonly engineModule: string;
  private readonly timeoutMs: number;
  private process: ChildProcessWithoutNullStreams | null = null;
  private stdoutReader: Interface | null = null;
  private stderrReader: Interface | null = null;
  private pending = new Map<string, PendingRequest>();
  private startPromise: Promise<void> | null = null;

  constructor(repoRoot: string) {
    this.repoRoot = repoRoot;
    this.pythonExecutable = process.env.EARLOOP_PYTHON_EXECUTABLE || "python";
    this.engineModule = process.env.EARLOOP_ENGINE_MODULE || "earloop.engine.server";
    this.timeoutMs = Number(process.env.EARLOOP_ENGINE_TIMEOUT_MS || 6000);
  }

  async ensureStarted(): Promise<void> {
    if (this.process && !this.process.killed && this.process.exitCode === null) return;
    if (this.startPromise) return this.startPromise;

    this.startPromise = new Promise<void>((resolve, reject) => {
      const child = spawn(this.pythonExecutable, ["-m", this.engineModule], {
        cwd: this.repoRoot,
        env: {
          ...process.env,
          PYTHONPATH: joinPythonPath(this.repoRoot),
        },
        stdio: ["pipe", "pipe", "pipe"],
      });

      const failStart = (error: unknown) => {
        this.cleanupProcess();
        reject(error);
      };

      child.once("error", failStart);
      child.once("spawn", () => {
        child.off("error", failStart);
        this.attachProcess(child);
        resolve();
      });
    }).finally(() => {
      this.startPromise = null;
    });

    return this.startPromise;
  }

  async restart(): Promise<void> {
    this.stop();
    await this.ensureStarted();
  }

  stop(): void {
    const child = this.process;
    this.rejectAllPending(new Error("Python engine bridge stopped"));
    this.cleanupReaders();
    this.process = null;
    if (child && child.exitCode === null && !child.killed) {
      child.kill();
    }
  }

  async send(command: string, payload: Record<string, unknown> = {}, requestId: string = randomUUID()): Promise<PythonResponse> {
    await this.ensureStarted();
    if (!this.process || this.process.exitCode !== null || this.process.killed) {
      throw new Error("Python engine process is not running");
    }

    return new Promise<PythonResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`Engine request timed out: ${command}`));
      }, this.timeoutMs);

      this.pending.set(requestId, { resolve, reject, timeout });

      const message: BridgeCommand = {
        request_id: requestId,
        command,
        payload,
      };

      this.process!.stdin.write(`${JSON.stringify(message)}\n`, "utf-8", (error) => {
        if (!error) return;
        clearTimeout(timeout);
        this.pending.delete(requestId);
        reject(error);
      });
    });
  }

  async health(): Promise<{ ok: boolean; pid: number | null }> {
    await this.ensureStarted();
    const response = await this.send("get_engine_status", {}, randomUUID());
    if (!response.ok) {
      throw new Error(response.error?.message || "Engine health check failed");
    }
    return { ok: true, pid: this.process?.pid ?? null };
  }

  private attachProcess(child: ChildProcessWithoutNullStreams): void {
    this.process = child;
    this.stdoutReader = createInterface({ input: child.stdout });
    this.stderrReader = createInterface({ input: child.stderr });

    this.stdoutReader.on("line", (line) => {
      if (!line.trim()) return;
      try {
        const response = JSON.parse(line) as PythonResponse;
        const requestId = response.request_id;
        if (!requestId) return;
        const pending = this.pending.get(requestId);
        if (!pending) return;
        clearTimeout(pending.timeout);
        this.pending.delete(requestId);
        pending.resolve(response);
      } catch (error) {
        console.error("[earloop-engine] invalid stdout line", error, line);
      }
    });

    this.stderrReader.on("line", (line) => {
      if (!line.trim()) return;
      console.error(`[earloop-engine stderr] ${line}`);
    });

    child.on("exit", (code, signal) => {
      const reason = new Error(`Python engine exited (code=${code ?? "null"}, signal=${signal ?? "null"})`);
      this.rejectAllPending(reason);
      this.cleanupReaders();
      this.process = null;
    });
  }

  private rejectAllPending(error: Error): void {
    for (const [requestId, pending] of this.pending.entries()) {
      clearTimeout(pending.timeout);
      pending.reject(error);
      this.pending.delete(requestId);
    }
  }

  private cleanupReaders(): void {
    this.stdoutReader?.removeAllListeners();
    this.stderrReader?.removeAllListeners();
    this.stdoutReader?.close();
    this.stderrReader?.close();
    this.stdoutReader = null;
    this.stderrReader = null;
  }

  private cleanupProcess(): void {
    this.cleanupReaders();
    if (this.process && this.process.exitCode === null && !this.process.killed) {
      this.process.kill();
    }
    this.process = null;
  }
}

export function earloopEngineBridgePlugin(): Plugin {
  let manager: PythonProcessManager | null = null;

  return {
    name: "earloop-engine-bridge",
    apply: "serve",
    configureServer(server: ViteDevServer) {
      const repoRoot = path.resolve(server.config.root, "..");
      manager = new PythonProcessManager(repoRoot);

      server.httpServer?.once("close", () => {
        manager?.stop();
      });

      server.middlewares.use("/__engine/health", async (_req, res) => {
        try {
          const result = await manager!.health();
          writeJson(res, 200, result);
        } catch (error) {
          writeJson(res, 503, {
            ok: false,
            error: error instanceof Error ? error.message : "Engine bridge health check failed",
          });
        }
      });

      server.middlewares.use("/__engine/restart", async (req, res) => {
        if (req.method !== "POST") {
          writeJson(res, 405, { ok: false, error: "Method not allowed" });
          return;
        }
        try {
          await manager!.restart();
          writeJson(res, 200, { ok: true });
        } catch (error) {
          writeJson(res, 500, {
            ok: false,
            error: error instanceof Error ? error.message : "Failed to restart engine bridge",
          });
        }
      });

      server.middlewares.use("/__engine/command", async (req, res) => {
        if (req.method !== "POST") {
          writeJson(res, 405, { ok: false, error: "Method not allowed" });
          return;
        }

        try {
          const body = await readJsonBody(req);
          if (!body || typeof body !== "object") {
            writeJson(res, 400, { ok: false, error: "Request body must be a JSON object" });
            return;
          }

          const request = body as { request_id?: string; command?: string; payload?: Record<string, unknown> };
          const requestId = typeof request.request_id === "string" && request.request_id ? request.request_id : randomUUID();
          const command = typeof request.command === "string" ? request.command : "";
          const payload = request.payload && typeof request.payload === "object" ? request.payload : {};

          if (!command) {
            writeJson(res, 400, { ok: false, error: "command is required" });
            return;
          }

          const response = await manager!.send(command, payload, requestId);
          writeJson(res, 200, {
            ok: response.ok,
            data: response.result,
            error: response.ok ? undefined : (response.error?.message || "Engine command failed"),
          });
        } catch (error) {
          writeJson(res, 500, {
            ok: false,
            error: error instanceof Error ? error.message : "Engine bridge request failed",
          });
        }
      });
    },
  };
}
