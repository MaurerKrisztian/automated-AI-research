/** Hash-based routes (#/study/0001/review), so the static site needs no server rewrites on GitHub Pages. */
import { useEffect, useState } from "react";
import { encodePath } from "./data";

export type Tab = "paper" | "review" | "notebook" | "files";

export type Route =
  | { name: "home" }
  | { name: "studies"; query: string }
  | { name: "study"; id: string; tab: Tab; anchor?: string }
  | { name: "file"; id: string; path: string }
  | { name: "page"; slug: string; anchor?: string }
  | { name: "notfound" };

const TABS: Tab[] = ["paper", "review", "notebook", "files"];

function decode(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const q = raw.indexOf("?");
  const parts = (q < 0 ? raw : raw.slice(0, q)).split("/").filter(Boolean).map(decode);
  const params = new URLSearchParams(q < 0 ? "" : raw.slice(q + 1));
  const anchor = params.get("h") ?? undefined;
  if (parts.length === 0) return { name: "home" };
  if (parts[0] === "studies" && parts.length === 1) return { name: "studies", query: params.get("q") ?? "" };
  if (parts[0] === "study" && parts[1]) {
    if (parts[2] === "file" && parts.length > 3) return { name: "file", id: parts[1], path: parts.slice(3).join("/") };
    const tab = TABS.find((t) => t === parts[2]);
    if (parts.length === 2 || tab) return { name: "study", id: parts[1], tab: tab ?? "paper", anchor };
  }
  if (parts[0] === "page" && parts[1] && parts.length === 2) return { name: "page", slug: parts[1], anchor };
  return { name: "notfound" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

const withAnchor = (base: string, anchor?: string) => (anchor ? `${base}?h=${encodeURIComponent(anchor)}` : base);

export const href = {
  home: () => "#/",
  studies: (query = "") => (query ? `#/studies?q=${encodeURIComponent(query)}` : "#/studies"),
  study: (id: string, tab: Tab = "paper", anchor?: string) =>
    withAnchor(tab === "paper" ? `#/study/${id}` : `#/study/${id}/${tab}`, anchor),
  file: (id: string, path: string) => `#/study/${id}/file/${encodePath(path)}`,
  page: (slug: string, anchor?: string) => withAnchor(`#/page/${slug}`, anchor),
};
