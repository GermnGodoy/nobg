const REPOSITORY = 'https://github.com/GermnGodoy/nobg';
const SOURCE_ROOT = `${REPOSITORY}/blob/main/src/nobg/`;

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function paragraphs(parent, lines, className = '') {
  for (const line of lines) parent.append(node('p', className, line));
}

function heading(parent, text, level, className) {
  const item = node(`h${level}`, className, text);
  parent.append(item);
  return item;
}

function section(parent, id, info) {
  const el = node('section', 'content-section');
  el.id = id;
  const head = node('div', 'section-heading');
  head.append(node('div', 'section-index', info.eyebrow));
  heading(head, info.title, 2);
  head.append(node('p', 'section-lede', info.intro));
  el.append(head);
  parent.append(el);
  return el;
}

function diagram(parent, id, caption, hint) {
  const fig = node('figure', 'diagram');
  fig.dataset.type = id;
  const scroll = node('div', 'diagram-scroll');
  scroll.id = `diagram-${id}`;
  scroll.setAttribute('role', 'img');
  scroll.setAttribute('aria-label', caption);
  fig.append(scroll, node('p', 'diagram-hint', hint), node('figcaption', '', caption));
  parent.append(fig);
}

function explainGrid(parent, items, className = 'explain-grid') {
  const grid = node('div', className);
  for (const info of items) {
    const item = node('div', 'explain-item');
    heading(item, info.title, 4);
    item.append(node('p', '', info.description));
    grid.append(item);
  }
  parent.append(grid);
}

function renderHero(main, content, ui) {
  const hero = node('section', 'hero');
  hero.id = 'overview';
  hero.append(node('div', 'eyebrow', content.eyebrow));
  heading(hero, content.title, 1);
  hero.append(node('p', 'hero-lede', content.lede));
  paragraphs(hero, content.paragraphs);
  const meta = node('div', 'hero-meta');
  meta.append(node('span', 'status', ui.status));
  meta.append(node('span', 'meta-divider'));
  meta.append(node('span', '', ui.metaText));
  hero.append(meta);
  main.append(hero);
}

function renderPipeline(main, data, ui) {
  const el = section(main, 'pipeline', data);
  heading(el, data.stepsHeading, 3, 'subheading');
  const list = node('ol', 'steps');
  for (const step of data.steps) {
    const li = node('li', 'step');
    const marker = node('span', 'step-number', step.number);
    marker.setAttribute('aria-hidden', 'true');
    const body = node('div', 'step-body');
    heading(body, step.title, 4);
    body.append(node('p', '', step.description));
    li.append(marker, body);
    list.append(li);
  }
  el.append(list);
}

function renderArchitecture(main, data, ui) {
  const el = section(main, 'architecture', data);
  diagram(el, 'architecture', data.diagram.caption, ui.diagramHint);
  heading(el, data.explainHeading, 3, 'subheading');
  explainGrid(el, data.explanations);
  heading(el, data.detailsHeading, 3, 'subheading');
  const group = node('div', 'details-group');
  for (const info of data.details) {
    const detail = node('details');
    detail.append(node('summary', '', info.title));
    const body = node('div', 'detail-body');
    paragraphs(body, info.paragraphs);
    if (info.rows?.length) {
      const dl = node('dl', 'data-rows');
      for (const row of info.rows) {
        const wrapper = node('div', 'data-row');
        wrapper.append(node('dt', '', row.label), node('dd', '', row.value));
        dl.append(wrapper);
      }
      body.append(dl);
    }
    detail.append(body);
    group.append(detail);
  }
  el.append(group);
}

function renderTraining(main, data, ui) {
  const el = section(main, 'training', data);
  heading(el, data.datasetHeading, 3, 'subheading');
  const prose = node('div', 'prose');
  paragraphs(prose, data.datasetParagraphs);
  el.append(prose);
  heading(el, data.lossHeading, 3, 'subheading');
  explainGrid(el, data.lossExplanations);

}

function renderCodeMap(main, data) {
  const el = section(main, 'code', data);
  const list = node('ul', 'code-map');
  for (const file of data.files) {
    const li = node('li');
    const link = node('a', '', file.path);
    link.href = `${SOURCE_ROOT}${file.path.split('/').map(encodeURIComponent).join('/')}`;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    li.append(link, node('p', '', file.description));
    list.append(li);
  }
  el.append(list);
}

function renderChrome(content) {
  document.title = content.meta.title;
  document.querySelector('meta[name="description"]').content = content.meta.description;
  document.querySelector('#skip-link').textContent = content.ui.skipLink;

  const header = document.querySelector('#site-header');
  const wordmark = node('a', 'wordmark', content.ui.brand);
  wordmark.href = '#overview';
  const nav = node('nav', 'nav');
  nav.setAttribute('aria-label', content.ui.brand);
  for (const item of content.nav) {
    const anchor = node('a', '', item.label);
    anchor.href = item.href;
    nav.append(anchor);
  }
  const source = node('a', 'external-link', content.ui.sourceLink);
  source.href = REPOSITORY;
  source.target = '_blank';
  source.rel = 'noopener noreferrer';
  source.append(node('span', 'external-mark', content.ui.externalMark));
  nav.append(source);
  header.append(wordmark, nav);

  const footer = document.querySelector('#site-footer');
  footer.append(node('p', '', content.ui.footerText));
  const link = node('a', '', content.ui.footerLink);
  link.href = REPOSITORY;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  footer.append(link);
}

function renderContent(content) {
  // This is the only source of user-facing wording, including diagram text.
  renderChrome(content);
  const main = document.querySelector('#main');
  renderHero(main, content.overview, content.ui);
  renderPipeline(main, content.pipeline, content.ui);
  renderArchitecture(main, content.architecture, content.ui);
  renderTraining(main, content.training, content.ui);
  renderCodeMap(main, content.codeMap);
}

function interpolateDiagram(template, labels) {
  const result = template.replace(/\{\{([a-zA-Z][a-zA-Z0-9_]*)\}\}/g, (_, key) => {
    if (typeof labels[key] !== 'string') throw new Error(`Unknown diagram label: ${key}`);
    // Mermaid labels are always quoted; remove unsafe quote / markup characters.
    return labels[key].replaceAll('&', '&amp;').replaceAll('"', '#quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  });
  if (/\{\{[^}]+\}\}/.test(result)) throw new Error('Unresolved Mermaid placeholder');
  return result;
}

async function renderDiagrams(content) {
  const mermaidModule = await import('https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs');
  const mermaid = mermaidModule.default;
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    themeVariables: {
      background: '#f8f8f4',
      primaryColor: '#f8f8f4',
      primaryTextColor: '#222925',
      primaryBorderColor: '#aab7ae',
      secondaryColor: '#f8f8f4',
      tertiaryColor: '#f2f5f0',
      mainBkg: '#f8f8f4',
      lineColor: '#87998d',
      clusterBkg: '#f5f7f3',
      clusterBorder: '#cdd7ce',
      fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif',
      fontSize: '13px'
    },
    flowchart: {
      htmlLabels: false,
      useMaxWidth: false,
      curve: 'linear',
      nodeSpacing: 24,
      rankSpacing: 38,
      padding: 12
    }
  });

  for (const name of ['architecture']) {
    const slot = document.getElementById(`diagram-${name}`);
    try {
      const response = await fetch(`./diagrams/${name}.mmd`);
      if (!response.ok) throw new Error(`Could not load ${name} diagram`);
      const source = await response.text();
      const text = interpolateDiagram(source, content[name].diagram.labels);
      const { svg, bindFunctions } = await mermaid.render(`nobg-${name}`, text);
      slot.innerHTML = svg;
      if (bindFunctions) bindFunctions(slot);
      // SVGs keep their native width, making diagrams readable when scrolled on mobile.
      slot.querySelector('svg')?.setAttribute('aria-hidden', 'true');
    } catch (error) {
      console.error(`Could not render the ${name} diagram:`, error);
      slot.append(node('p', 'section-lede', content.ui.diagramUnavailable));
    }
  }
}

async function main() {
  const response = await fetch('./content.json');
  if (!response.ok) throw new Error('Could not load content.json');
  const content = await response.json();
  renderContent(content);
  try {
    await renderDiagrams(content);
  } catch (error) {
    console.error('Diagram rendering failed:', error);
    for (const name of ['architecture']) {
      const slot = document.getElementById(`diagram-${name}`);
      if (slot && !slot.hasChildNodes()) {
        slot.append(node('p', 'section-lede', content.ui.diagramUnavailable));
      }
    }
  }
}

main().catch((error) => console.error('Site rendering failed:', error));
