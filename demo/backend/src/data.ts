import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dirname = path.dirname(fileURLToPath(import.meta.url));
const defaultDataDir = path.resolve(dirname, "../data");

export const dataDir = process.env.DEMO_DATA_DIR
  ? path.resolve(process.env.DEMO_DATA_DIR)
  : defaultDataDir;

const jsonNamePattern = /^[a-z0-9_.-]+\.json$/i;

export async function readJson<T = unknown>(name: string): Promise<T> {
  if (!jsonNamePattern.test(name)) {
    throw new Error(`Invalid data file name: ${name}`);
  }

  const filePath = path.join(dataDir, name);
  const raw = await readFile(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

export async function listDataFiles(): Promise<Array<{ name: string; bytes: number }>> {
  const names = await readdir(dataDir);
  const files = await Promise.all(
    names
      .filter((name) => name.endsWith(".json"))
      .sort()
      .map(async (name) => {
        const fileStat = await stat(path.join(dataDir, name));
        return { name, bytes: fileStat.size };
      }),
  );
  return files;
}

export function sliceRows<T>(rows: T[], limit?: unknown, offset?: unknown): T[] {
  const parsedLimit = Number(limit ?? rows.length);
  const parsedOffset = Number(offset ?? 0);
  const safeLimit = Number.isFinite(parsedLimit)
    ? Math.max(1, Math.min(Math.trunc(parsedLimit), 1000))
    : rows.length;
  const safeOffset = Number.isFinite(parsedOffset) ? Math.max(0, Math.trunc(parsedOffset)) : 0;
  return rows.slice(safeOffset, safeOffset + safeLimit);
}
