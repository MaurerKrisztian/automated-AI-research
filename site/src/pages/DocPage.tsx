import { useEffect } from "react";
import { DocLayout, NotFound } from "../components";
import { stripTitle, type Content } from "../data";
import { makeResolvers } from "../links";
import { Markdown, outline } from "../Markdown";
import { href } from "../router";

/** A repository document published as a page: roadmap, backlog, protocol, compute budget. */
export function DocPage({ content, slug, anchor }: { content: Content; slug: string; anchor?: string }) {
  const page = content.pages.find((p) => p.slug === slug);
  useEffect(() => {
    const target = anchor ? document.getElementById(anchor) : null;
    if (target) target.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [slug, anchor]);
  if (!page) return <NotFound />;
  const markdown = stripTitle(page.markdown);
  const anchorHref = (a: string) => href.page(slug, a);
  const r = makeResolvers(content, page.path, anchorHref);
  return (
    <>
      <header className="page-head">
        <h1>{page.title}</h1>
      </header>
      <DocLayout items={outline(markdown)} hrefFor={anchorHref}>
        <article className="prose">
          <Markdown markdown={markdown} resolveLink={r.link} resolveImage={r.image} />
        </article>
      </DocLayout>
    </>
  );
}
