/** The content the site renders: written by `python3 tools/studies.py export site/public/content`. */

export type Status = "active" | "review" | "validated" | "refuted" | "inconclusive" | "superseded";

export interface StudyFile {
  path: string;
  size: number;
  /** false when the file was too large to publish */
  copied: boolean;
}

export interface Study {
  id: string;
  title: string;
  status: Status;
  evidence: string;
  track: string;
  builds_on: string[];
  tags: string[];
  summary: string;
  updated: string;
  superseded_by: string | null;
  authors: string[];
  folder: string;
  built_on_by: string[];
  docs: { readme: string; review?: string; notebook?: string };
  files: StudyFile[];
}

export interface Page {
  slug: string;
  title: string;
  path: string;
  markdown: string;
}

export interface Content {
  generated: string;
  tracks: Record<string, string>;
  statuses: Status[];
  evidence: string[];
  studies: Study[];
  pages: Page[];
}

export async function loadContent(): Promise<Content> {
  const res = await fetch("content/index.json", { cache: "no-cache" });
  if (!res.ok) throw new Error(`content/index.json returned HTTP ${res.status}`);
  return (await res.json()) as Content;
}

export const STATUS_INFO: Record<Status, { label: string; hint: string }> = {
  active: { label: "Active", hint: "Work in progress" },
  review: { label: "In review", hint: "Written up; the review is pending or has open issues" },
  validated: { label: "Validated", hint: "Claims hold at their evidence levels and the review passed" },
  refuted: { label: "Refuted", hint: "The central claim failed, or the method was already published" },
  inconclusive: { label: "Inconclusive", hint: "The budget ran out before a decision" },
  superseded: { label: "Superseded", hint: "Replaced by a later study" },
};

export const EVIDENCE_INFO: Record<string, { name: string; hint: string }> = {
  E0: { name: "Stated", hint: "Idea and hypothesis written down; nothing checked" },
  E1: { name: "Derived", hint: "Derivation or proof written, assumptions explicit" },
  E2: { name: "Verified", hint: "Matches a direct reference in float64, edge cases included" },
  E3: { name: "Benchmarked", hint: "Measured on the GPU against a fair baseline" },
  E4: { name: "Small-model evidence", hint: "Small models on synthetic tasks, 3+ seeds, tuned baselines" },
  E5: { name: "Language-model evidence", hint: "A tiny or small language model against a tuned baseline" },
};

export function evidenceLevel(evidence: string): number {
  const n = Number(evidence.replace(/^E/, ""));
  return Number.isFinite(n) ? n : 0;
}

export function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

/** URL of a study file published next to index.json. */
export function fileUrl(study: Study, path: string): string {
  return `content/studies/${encodePath(study.folder)}/${encodePath(path)}`;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}

/** Markdown without its leading level-1 heading: pages show the title in their own header. */
export function stripTitle(markdown: string): string {
  return markdown.replace(/^\s*# [^\n]*\n+/, "");
}
