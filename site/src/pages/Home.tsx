import { useState } from "react";
import { EvidenceMeter, SearchBox, StudyCard } from "../components";
import { site } from "../config";
import { EVIDENCE_INFO, STATUS_INFO, type Content, type Status } from "../data";
import { href } from "../router";

const STEPS: [string, string][] = [
  ["Hypothesis", "One falsifiable claim, and the result that would refute it."],
  ["Prior work", "A documented literature search before any heavy work."],
  ["Math first", "Every identity and gradient checked in float64 against a direct reference."],
  ["Fair experiments", "The strongest baselines, an equal tuning budget, three or more seeds."],
  ["Independent review", "A referee agent tries to break every claim before it counts."],
  ["Build on it", "Validated code moves into a shared library for later studies."],
];

export function Home({ content }: { content: Content }) {
  const [query, setQuery] = useState("");
  const counts = new Map<Status, number>();
  for (const s of content.studies) counts.set(s.status, (counts.get(s.status) ?? 0) + 1);
  const recent = [...content.studies]
    .sort((a, b) => b.updated.localeCompare(a.updated) || b.id.localeCompare(a.id))
    .slice(0, 4);
  return (
    <>
      <section className="hero">
        <h1>{site.name}</h1>
        <p className="lede">{site.tagline}</p>
        <SearchBox
          value={query}
          onChange={setQuery}
          onSubmit={(q) => {
            window.location.hash = href.studies(q);
          }}
          large
        />
        <dl className="stats">
          <div>
            <dt>Studies</dt>
            <dd>{content.studies.length}</dd>
          </div>
          {content.statuses
            .filter((s) => counts.get(s))
            .map((s) => (
              <div key={s}>
                <dt>{STATUS_INFO[s].label}</dt>
                <dd>{counts.get(s)}</dd>
              </div>
            ))}
        </dl>
      </section>

      <section className="section cta">
        <div>
          <h2>Claude usage left this week?</h2>
          <p>
            Turn it into research before your limit resets. Pick a prompt, let Claude run a study on your fork, and open a
            pull request. Theory, reviews and literature checks work from your phone; experiments need a GPU.
          </p>
        </div>
        <div className="cta-actions">
          <a className="button" href={href.page("prompts")}>Get a prompt</a>
          <a className="button button-quiet" href={href.page("contribute")}>How to contribute</a>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Latest studies</h2>
          <a href={href.studies()}>All studies</a>
        </div>
        <div className="grid">
          {recent.map((s) => <StudyCard key={s.id} study={s} tracks={content.tracks} />)}
        </div>
      </section>

      <section className="section two-col">
        <div>
          <h2>How a study works</h2>
          <ol className="steps">
            {STEPS.map(([title, text]) => (
              <li key={title}>
                <strong>{title}.</strong> {text}
              </li>
            ))}
          </ol>
          <p>
            <a href={href.page("protocol")}>Read the full protocol</a>
          </p>
        </div>
        <div>
          <h2>Evidence levels</h2>
          <ul className="ladder">
            {Object.entries(EVIDENCE_INFO).map(([level, info]) => (
              <li key={level}>
                <EvidenceMeter level={level} />
                <div>
                  <strong>{info.name}</strong>
                  <span>{info.hint}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </>
  );
}
