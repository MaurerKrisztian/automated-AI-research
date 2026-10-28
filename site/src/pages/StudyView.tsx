import { useEffect } from "react";
import { DocLayout, EvidenceMeter, NotFound, StatusBadge, Tags } from "../components";
import { formatBytes, stripTitle, type Content, type Study, type StudyFile } from "../data";
import { makeResolvers } from "../links";
import { Markdown, outline } from "../Markdown";
import { href, type Tab } from "../router";

const TABS: Tab[] = ["paper", "review", "notebook", "files"];
const TAB_LABEL: Record<Tab, string> = { paper: "Paper", review: "Review", notebook: "Notebook", files: "Files" };
const TAB_FILE: Record<Exclude<Tab, "files">, string> = { paper: "README.md", review: "REVIEW.md", notebook: "NOTEBOOK.md" };
const GROUP_TITLE: Record<string, string> = { code: "Code", logs: "Logs", results: "Results", archive: "Archive" };
const GROUP_ORDER = ["code", "logs", "results", "archive"];

function StudyLinks({ label, ids, content }: { label: string; ids: string[]; content: Content }) {
  if (!ids.length) return null;
  return (
    <p className="relations">
      <span className="muted">{label}</span>{" "}
      {ids.map((id, i) => {
        const s = content.studies.find((x) => x.id === id);
        return (
          <span key={id}>
            {i ? ", " : null}
            <a href={href.study(id)}>{s ? `${id} · ${s.title}` : id}</a>
          </span>
        );
      })}
    </p>
  );
}

function FileList({ study }: { study: Study }) {
  const groups = new Map<string, StudyFile[]>();
  for (const f of study.files) {
    const group = f.path.includes("/") ? f.path.split("/")[0] : "code";
    groups.set(group, [...(groups.get(group) ?? []), f]);
  }
  const rank = (g: string) => (GROUP_ORDER.includes(g) ? GROUP_ORDER.indexOf(g) : GROUP_ORDER.length);
  const sorted = [...groups].sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b));
  if (!sorted.length) return <p className="state">No files yet.</p>;
  return (
    <div className="files">
      {sorted.map(([group, files]) => (
        <section key={group}>
          <h2>
            {GROUP_TITLE[group] ?? group} <span className="muted">{files.length}</span>
          </h2>
          <ul className="file-list">
            {files.map((f) => (
              <li key={f.path}>
                <a href={href.file(study.id, f.path)}>{group === "code" ? f.path : f.path.slice(group.length + 1)}</a>
                <span className="muted">{formatBytes(f.size)}</span>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

export function StudyView({ content, id, tab, anchor }: { content: Content; id: string; tab: Tab; anchor?: string }) {
  const study = content.studies.find((s) => s.id === id);
  useEffect(() => {
    const target = anchor ? document.getElementById(anchor) : null;
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [id, tab, anchor]);
  if (!study) return <NotFound what="study" />;

  let body;
  if (tab === "files") {
    body = <FileList study={study} />;
  } else {
    const raw = tab === "paper" ? study.docs.readme : study.docs[tab];
    const markdown = raw ? stripTitle(raw) : "";
    const anchorHref = (a: string) => href.study(study.id, tab, a);
    const r = makeResolvers(content, `studies/${study.folder}/${TAB_FILE[tab]}`, anchorHref);
    body = markdown ? (
      <DocLayout items={outline(markdown)} hrefFor={anchorHref}>
        <article className="prose">
          <Markdown markdown={markdown} resolveLink={r.link} resolveImage={r.image} />
        </article>
      </DocLayout>
    ) : (
      <p className="state">No {TAB_LABEL[tab].toLowerCase()} yet.</p>
    );
  }

  return (
    <>
      <nav className="crumbs" aria-label="Breadcrumb">
        <a href={href.studies()}>Studies</a>
        <span aria-hidden="true">/</span>
        <span>{study.id}</span>
      </nav>
      <header className="study-head">
        <h1>{study.title}</h1>
        <div className="study-meta">
          <span className="study-id">{study.id}</span>
          <StatusBadge status={study.status} />
          <EvidenceMeter level={study.evidence} showName />
          <span className="track">{content.tracks[study.track] ?? `Track ${study.track}`}</span>
          <span className="updated">Updated {study.updated}</span>
        </div>
        {study.authors?.length ? <p className="authors">By {study.authors.join(", ")}</p> : null}
        <p className="study-summary">{study.summary}</p>
        <StudyLinks label="Builds on" ids={study.builds_on} content={content} />
        <StudyLinks label="Built on by" ids={study.built_on_by} content={content} />
        <StudyLinks label="Superseded by" ids={study.superseded_by ? [study.superseded_by] : []} content={content} />
        <Tags tags={study.tags} />
      </header>
      <nav className="tabs" aria-label="Study sections">
        {TABS.map((t) => (
          <a key={t} href={href.study(study.id, t)} aria-current={t === tab ? "page" : undefined}>
            {TAB_LABEL[t]}
            {t === "files" ? <span className="tab-count">{study.files.length}</span> : null}
          </a>
        ))}
      </nav>
      {body}
    </>
  );
}
