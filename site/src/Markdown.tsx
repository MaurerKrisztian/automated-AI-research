import type { Root, RootContent } from "hast";
import katex from "katex";
import { Children, isValidElement, useRef, useState, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { Resolved } from "./links";

/** GitHub-style heading anchor: lowercase, punctuation dropped, spaces to hyphens. */
export function slugify(text: string): string {
  return text.trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s/g, "-");
}

export interface OutlineItem {
  id: string;
  text: string;
}

/** The document's level-2 headings, with the same ids rehypeHeadingIds gives them. */
export function outline(markdown: string): OutlineItem[] {
  const seen = new Map<string, number>();
  const items: OutlineItem[] = [];
  let fenced = false;
  for (const line of markdown.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    const m = fenced ? null : /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (!m) continue;
    const text = m[2].replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/[`*~]/g, "").trim();
    const base = slugify(text) || "section";
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    if (m[1].length === 2) items.push({ id: n ? `${base}-${n}` : base, text });
  }
  return items;
}

function hastText(node: Root | RootContent): string {
  if (node.type === "text") return node.value;
  return "children" in node ? node.children.map(hastText).join("") : "";
}

/** Give every heading a unique id (once per document, so anchors work and match GitHub's). */
function rehypeHeadingIds() {
  return (tree: Root) => {
    const seen = new Map<string, number>();
    const walk = (node: Root | RootContent) => {
      if (node.type === "element" && /^h[1-6]$/.test(node.tagName)) {
        const base = slugify(hastText(node)) || "section";
        const n = seen.get(base) ?? 0;
        seen.set(base, n + 1);
        node.properties.id = n ? `${base}-${n}` : base;
      }
      if ("children" in node) node.children.forEach(walk);
    };
    walk(tree);
  };
}

/** A code block with a Copy button. ```text blocks (the prompts) wrap instead of scrolling sideways. */
function CodeBlock({ wrap, children }: { wrap: boolean; children: ReactNode }) {
  const pre = useRef<HTMLPreElement>(null);
  const [label, setLabel] = useState("Copy");
  const copy = async () => {
    const text = pre.current?.innerText ?? "";
    try {
      await navigator.clipboard.writeText(text);
      setLabel("Copied");
    } catch {
      // clipboard blocked: select the text so it can be copied by hand
      if (pre.current) window.getSelection()?.selectAllChildren(pre.current);
      setLabel("Selected");
    }
    window.setTimeout(() => setLabel("Copy"), 1600);
  };
  return (
    <div className="code-block">
      <button type="button" className="copy-button" onClick={copy}>
        {label}
      </button>
      <pre ref={pre} className={wrap ? "wrap" : undefined}>
        {children}
      </pre>
    </div>
  );
}

interface Props {
  markdown: string;
  resolveLink: (url: string) => Resolved;
  resolveImage: (url: string) => string;
}

export function Markdown({ markdown, resolveLink, resolveImage }: Props) {
  const components: Components = {
    a({ href = "", children }) {
      const r = resolveLink(href);
      if (!r.href) return <span className="dead-link" title="Not published on this site">{children}</span>;
      return r.external ? <a href={r.href} target="_blank" rel="noreferrer">{children}</a> : <a href={r.href}>{children}</a>;
    },
    img({ src, alt }) {
      return <img src={resolveImage(String(src ?? ""))} alt={alt ?? ""} loading="lazy" />;
    },
    table({ children }) {
      return <div className="table-wrap"><table>{children}</table></div>;
    },
    // ```math fences (GitHub's display-math syntax), in case rehype-katex leaves them as code
    code({ className, children }) {
      if (className?.includes("language-math")) {
        const html = katex.renderToString(String(children).trim(), { displayMode: true, throwOnError: false });
        return <span className="math-display" dangerouslySetInnerHTML={{ __html: html }} />;
      }
      return <code className={className}>{children}</code>;
    },
    pre({ children }) {
      const child = Children.toArray(children)[0];
      const lang = isValidElement<{ className?: string }>(child) ? child.props.className ?? "" : "";
      if (lang.includes("language-math")) return <>{children}</>;
      return <CodeBlock wrap={lang.includes("language-text")}>{children}</CodeBlock>;
    },
  };
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex, rehypeHeadingIds]}
      components={components}
      skipHtml
    >
      {markdown}
    </ReactMarkdown>
  );
}
