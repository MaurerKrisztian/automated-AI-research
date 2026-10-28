import { useEffect, useState, type ReactNode } from "react";
import { NotFound } from "../components";
import { fileUrl, formatBytes, type Content } from "../data";
import { makeResolvers } from "../links";
import { Markdown } from "../Markdown";
import { href } from "../router";

const TEXT = /\.(log|txt|py|sh|md|json|csv|tsv|ya?ml|toml|ts|tsx|js|cfg|ini|tex)$/i;
const IMAGE = /\.(png|jpe?g|gif|svg|webp)$/i;

function prettyJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

export function FileView({ content, id, path }: { content: Content; id: string; path: string }) {
  const study = content.studies.find((s) => s.id === id);
  const file = study?.files.find((f) => f.path === path);
  const url = study ? fileUrl(study, path) : "";
  const isText = TEXT.test(path);
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    window.scrollTo(0, 0);
    setText(null);
    setError(null);
    if (!file?.copied || !isText) return;
    let cancelled = false;
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(
        (t) => {
          if (!cancelled) setText(t);
        },
        (e: unknown) => {
          if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        },
      );
    return () => {
      cancelled = true;
    };
  }, [url, file, isText]);

  if (!study || !file) return <NotFound what="file" />;
  const name = path.split("/").pop() ?? path;
  let body: ReactNode;
  if (!file.copied) {
    body = <p className="state">This file is too large to publish here ({formatBytes(file.size)}).</p>;
  } else if (IMAGE.test(path)) {
    body = <img className="file-image" src={url} alt={name} />;
  } else if (!isText) {
    body = (
      <p className="state">
        Binary file: <a href={url} download={name}>download it</a> to open it.
      </p>
    );
  } else if (error) {
    body = <p className="state">Couldn't load this file ({error}).</p>;
  } else if (text === null) {
    body = <p className="state">Loading…</p>;
  } else if (/\.md$/i.test(path)) {
    const r = makeResolvers(content, `studies/${study.folder}/${path}`, () => href.file(study.id, path));
    body = (
      <article className="prose">
        <Markdown markdown={text} resolveLink={r.link} resolveImage={r.image} />
      </article>
    );
  } else {
    body = <pre className="code-view">{/\.json$/i.test(path) ? prettyJson(text) : text}</pre>;
  }

  return (
    <>
      <nav className="crumbs" aria-label="Breadcrumb">
        <a href={href.studies()}>Studies</a>
        <span aria-hidden="true">/</span>
        <a href={href.study(study.id)}>{study.id}</a>
        <span aria-hidden="true">/</span>
        <a href={href.study(study.id, "files")}>Files</a>
      </nav>
      <header className="file-head">
        <h1>{path}</h1>
        <div className="file-actions">
          <span className="muted">{formatBytes(file.size)}</span>
          {file.copied ? (
            <a className="button" href={url} download={name}>
              Download
            </a>
          ) : null}
        </div>
      </header>
      {body}
    </>
  );
}
