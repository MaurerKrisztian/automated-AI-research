/** Full-text search over every study's paper, review and notebook, and the site pages, built in the browser. */
import MiniSearch from "minisearch";
import type { Content } from "./data";

export type DocKind = "paper" | "review" | "notebook" | "page";

interface Doc {
  id: string;
  kind: DocKind;
  studyId?: string;
  slug?: string;
  title: string;
  tags: string;
  files: string;
  text: string;
}

export interface Hit {
  id: string;
  kind: DocKind;
  studyId?: string;
  slug?: string;
  title: string;
  score: number;
  terms: string[];
  text: string;
}

/** Markdown reduced to readable text for indexing and snippets. */
export function plainText(md: string): string {
  return md
    .replace(/```[^\n]*\n([\s\S]*?)```/g, " $1 ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-*:?\s*\|?\s*$/gm, " ")
    .replace(/[|*`>]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function buildSearch(content: Content): (query: string) => Hit[] {
  const docs: Doc[] = [];
  for (const s of content.studies) {
    const tags = [...s.tags, s.status, s.evidence, content.tracks[s.track] ?? ""].join(" ");
    docs.push({
      id: `${s.id}:paper`, kind: "paper", studyId: s.id, title: `${s.id} ${s.title}`, tags,
      files: s.files.map((f) => f.path).join(" "), text: plainText(`${s.summary}\n\n${s.docs.readme}`),
    });
    for (const kind of ["review", "notebook"] as const) {
      const md = s.docs[kind];
      if (md) docs.push({ id: `${s.id}:${kind}`, kind, studyId: s.id, title: s.title, tags: "", files: "", text: plainText(md) });
    }
  }
  for (const p of content.pages) {
    docs.push({ id: `page:${p.slug}`, kind: "page", slug: p.slug, title: p.title, tags: "", files: "", text: plainText(p.markdown) });
  }
  const index = new MiniSearch<Doc>({
    fields: ["title", "tags", "files", "text"],
    storeFields: ["kind", "studyId", "slug", "title"],
    searchOptions: {
      boost: { title: 4, tags: 3, files: 1.5 },
      prefix: true,
      fuzzy: (term) => (term.length > 5 ? 0.15 : false), // one typo in words of 6–13 letters
      combineWith: "AND",
    },
  });
  index.addAll(docs);
  const textById = new Map(docs.map((d) => [d.id, d.text]));
  return (query) => {
    if (!query.trim()) return [];
    return index.search(query).map((r) => ({
      id: String(r.id),
      kind: r.kind as DocKind,
      studyId: r.studyId as string | undefined,
      slug: r.slug as string | undefined,
      title: r.title as string,
      score: r.score,
      terms: r.terms,
      text: textById.get(String(r.id)) ?? "",
    }));
  };
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** A short excerpt around the first matched term, split into plain and highlighted parts. */
export function snippet(text: string, terms: string[], width = 220): { text: string; mark: boolean }[] {
  const lower = text.toLowerCase();
  let first = -1;
  for (const t of terms) {
    const i = lower.indexOf(t.toLowerCase());
    if (i >= 0 && (first < 0 || i < first)) first = i;
  }
  const start = Math.max(0, first - 70);
  let s = text.slice(start, start + width);
  if (start > 0) s = "…" + s.replace(/^\S*\s/, "");
  if (start + width < text.length) s = s.replace(/\s\S*$/, "") + "…";
  if (!terms.length) return [{ text: s, mark: false }];
  const re = new RegExp(`(${[...terms].sort((a, b) => b.length - a.length).map(escapeRe).join("|")})`, "gi");
  return s
    .split(re)
    .map((part, i) => ({ text: part, mark: i % 2 === 1 }))
    .filter((p) => p.text);
}
