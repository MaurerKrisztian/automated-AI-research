/** Map links inside a markdown document (written for GitHub) to routes and files on this site. */
import { site } from "./config";
import { fileUrl, type Content } from "./data";
import { href, type Tab } from "./router";

export interface Resolved {
  /** null when the target isn't published on the site */
  href: string | null;
  external: boolean;
}

const DOC_TABS: Record<string, Tab> = { "README.md": "paper", "REVIEW.md": "review", "NOTEBOOK.md": "notebook" };

function splitAnchor(url: string): [string, string | undefined] {
  const i = url.indexOf("#");
  return i < 0 ? [url, undefined] : [url.slice(0, i), url.slice(i + 1) || undefined];
}

function normalize(path: string): string {
  const out: string[] = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") out.pop();
    else out.push(part);
  }
  return out.join("/");
}

function resolvePath(base: string, rel: string): string {
  if (rel.startsWith("/")) return normalize(rel);
  const dir = base.includes("/") ? base.slice(0, base.lastIndexOf("/")) : "";
  return normalize(dir ? `${dir}/${rel}` : rel);
}

/**
 * `base` is the document's path in the repository (e.g. "studies/0001-x/README.md");
 * `anchorHref` turns an in-page anchor into a route for the current view.
 */
export function makeResolvers(content: Content, base: string, anchorHref: (anchor: string) => string) {
  const studies = new Map(content.studies.map((s) => [s.folder, s]));
  const pages = new Map(content.pages.map((p) => [p.path, p]));

  function link(url: string): Resolved {
    if (/^(https?:|mailto:)/i.test(url)) return { href: url, external: true };
    if (url.startsWith("#")) return { href: anchorHref(url.slice(1)), external: false };
    const [rel, anchor] = splitAnchor(url);
    const path = resolvePath(base, rel);
    const m = /^studies\/([^/]+)(?:\/(.*))?$/.exec(path);
    const study = m ? studies.get(m[1]) : undefined;
    if (study) {
      const rest = m?.[2] ?? "";
      if (!rest || DOC_TABS[rest]) return { href: href.study(study.id, DOC_TABS[rest] ?? "paper", anchor), external: false };
      const isFile = study.files.some((f) => f.path === rest);
      return { href: isFile ? href.file(study.id, rest) : href.study(study.id, "files"), external: false };
    }
    const page = pages.get(path);
    if (page) return { href: href.page(page.slug, anchor), external: false };
    if (path === "STUDIES.md") return { href: href.studies(), external: false };
    if (site.repoUrl) {
      return { href: `${site.repoUrl}/blob/${site.branch}/${path}${anchor ? `#${anchor}` : ""}`, external: true };
    }
    return { href: null, external: false };
  }

  function image(url: string): string {
    if (/^(https?:|data:)/i.test(url)) return url;
    const path = resolvePath(base, url);
    const m = /^studies\/([^/]+)\/(.+)$/.exec(path);
    const study = m ? studies.get(m[1]) : undefined;
    return study && m ? fileUrl(study, m[2]) : url;
  }

  return { link, image };
}
