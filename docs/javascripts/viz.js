/* ==========================================================================
   ODP-Reuse — interactive figures.
   Every chart reads a JSON payload emitted by pipeline/gen_site.py, so the
   figures cannot drift from the data. D3 only; no chart library.
   ========================================================================== */
(function () {
  'use strict';

  // viz.js is always served from <site-root>/javascripts/viz.js, so its own URL
  // yields the site root whatever page is being viewed.
  //
  // This matters because every data-src is site-root-relative ("data/foo.json").
  // fetch() resolves a relative URL against the PAGE, and mkdocs-material uses
  // directory URLs, so on /ODP-Reuse/filters/ the browser asked for
  // /ODP-Reuse/filters/data/rules.json and got a 404 -- every figure below the
  // site root failed, and navigation.instant made the root page fail too once it
  // had been navigated away from and back.
  const SITE_ROOT = (function () {
    const s = document.currentScript
      || document.querySelector('script[src*="javascripts/viz.js"]');
    try {
      return s ? new URL('../', s.src).href : new URL('./', location.href).href;
    } catch (e) {
      return new URL('./', location.href).href;
    }
  })();

  const tip = (() => {
    let el = null;
    return {
      show(html, ev) {
        if (!el) { el = document.createElement('div'); el.className = 'odp-tip';
                   document.body.appendChild(el); }
        el.innerHTML = html; el.style.opacity = '1';
        const pad = 14;
        let x = ev.clientX + pad, y = ev.clientY + pad;
        const r = el.getBoundingClientRect();
        if (x + r.width > innerWidth) x = ev.clientX - r.width - pad;
        if (y + r.height > innerHeight) y = ev.clientY - r.height - pad;
        el.style.left = x + 'px'; el.style.top = y + 'px';
      },
      hide() { if (el) el.style.opacity = '0'; }
    };
  })();

  const fmt = n => n.toLocaleString('en-US');
  // The colour tokens are declared on [data-md-color-scheme], which
  // mkdocs-material sets on <body>, not <html>. Reading them from <html>
  // returned "" for every token, so every token-coloured chart drew in black.
  const css = v => getComputedStyle(document.body || document.documentElement)
    .getPropertyValue(v).trim();

  async function load(el) {
    // Resolved against SITE_ROOT, not the page: see the note at the top.
    const url = new URL(el.dataset.src, SITE_ROOT).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ': ' + r.status);
    return r.json();
  }

  /* ---- funnel: horizontal stacked band, the corpus at a glance ---------- */
  async function funnel(el) {
    const data = await load(el);
    const total = d3.sum(data, d => d.n);
    const W = el.clientWidth || 800, H = 132, top = 34;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img')
      .attr('aria-label', 'Corpus composition by quality-gate verdict');
    const x = d3.scaleLinear().domain([0, total]).range([0, W]);

    let acc = 0;
    svg.selectAll('rect').data(data).join('rect')
      .attr('class', 'odp-band')
      .attr('x', d => { const v = x(acc); acc += d.n; return v; })
      .attr('y', top).attr('height', 54).attr('rx', 3)
      .attr('width', d => Math.max(1, x(d.n) - 1.5))
      .attr('fill', d => d.color)
      .on('mousemove', (ev, d) => tip.show(
        `<b>${d.label}</b><br>${fmt(d.n)} ontologies · ` +
        `${(100 * d.n / total).toFixed(1)}%`, ev))
      .on('mouseleave', tip.hide);

    const kept = data.find(d => d.rule === 'KEPT');
    svg.append('text').attr('x', 0).attr('y', 20)
      .attr('class', 'odp-value').attr('font-size', 15)
      .text(`${fmt(total)} harvested`);
    if (kept) {
      svg.append('text').attr('x', W).attr('y', 20).attr('text-anchor', 'end')
        .attr('class', 'odp-value').attr('font-size', 15)
        .attr('fill', kept.color)
        .text(`${fmt(kept.n)} usable (${(100 * kept.n / total).toFixed(0)}%)`);
    }
    acc = 0;
    svg.selectAll('.lab').data(data).join('text')
      .attr('class', 'lab')
      .attr('x', d => { const v = x(acc) + (x(d.n) / 2); acc += d.n; return v; })
      .attr('y', top + 76).attr('text-anchor', 'middle').attr('font-size', 10)
      .attr('fill', 'currentColor').attr('opacity', d => x(d.n) > 54 ? .75 : 0)
      .text(d => d.label);
  }

  /* ---- per-portion rule breakdown -------------------------------------- */
  async function rules(el) {
    const data = await load(el);
    const W = el.clientWidth || 800, rowH = 56, H = data.length * rowH + 24;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const max = d3.max(data, d => d.discovered);
    const x = d3.scaleLinear().domain([0, max]).range([0, W - 150]);

    data.forEach((row, i) => {
      const g = svg.append('g').attr('transform', `translate(0,${i * rowH + 16})`);
      g.append('text').attr('x', 0).attr('y', 20).attr('font-size', 12)
        .attr('font-weight', 600).text(row.portion);
      let acc = 0;
      g.selectAll('rect').data(row.bands.filter(b => b.n > 0)).join('rect')
        .attr('class', 'odp-band')
        .attr('x', d => { const v = 150 + x(acc); acc += d.n; return v; })
        .attr('y', 4).attr('height', 26).attr('rx', 3)
        .attr('width', d => Math.max(1, x(d.n) - 1))
        .attr('fill', d => d.color)
        .on('mousemove', (ev, d) => tip.show(
          `<b>${d.rule}</b><br>${fmt(d.n)} of ${fmt(row.discovered)}`, ev))
        .on('mouseleave', tip.hide);
    });
  }

  /* ---- reuse-threshold decay ------------------------------------------- */
  async function thresholds(el) {
    const data = await load(el);
    if (!data.length) return;
    const W = el.clientWidth || 800, H = 260, m = { t: 18, r: 18, b: 34, l: 46 };
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scalePoint().domain([10, 30, 50, 70, 90, 100]).range([m.l, W - m.r]);
    const max = d3.max(data, s => d3.max(s.points, p => p.n)) || 1;
    const y = d3.scaleLinear().domain([0, max]).nice().range([H - m.b, m.t]);
    const colors = [css('--odp-coral'), css('--odp-violet'), css('--odp-mint')];

    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickFormat(d => d + '%'));
    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));

    const line = d3.line().x(d => x(d.t)).y(d => y(d.n)).curve(d3.curveMonotoneX);
    data.forEach((s, i) => {
      svg.append('path').datum(s.points).attr('fill', 'none')
        .attr('stroke', colors[i % 3]).attr('stroke-width', 2.5)
        .attr('stroke-linecap', 'round').attr('d', line);
      svg.selectAll(null).data(s.points).join('circle')
        .attr('cx', d => x(d.t)).attr('cy', d => y(d.n)).attr('r', 4)
        .attr('fill', colors[i % 3])
        .on('mousemove', (ev, d) => tip.show(
          `<b>${s.portion}</b><br>${fmt(d.n)} ontologies reuse ≥${d.t}% of a pattern`, ev))
        .on('mouseleave', tip.hide);
      const last = s.points[s.points.length - 1];
      svg.append('text').attr('x', x(last.t)).attr('y', y(last.n) - 10)
        .attr('text-anchor', 'end').attr('font-size', 11)
        .attr('font-weight', 600).attr('fill', colors[i % 3]).text(s.portion);
    });
  }

  /* ---- reuse forms ------------------------------------------------------ */
  async function forms(el) {
    const data = await load(el);
    if (!data.length) return;
    const roles = data[0].roles.map(r => r.role);
    const W = el.clientWidth || 800, rowH = 46, top = 22;
    const color = d3.scaleOrdinal().domain(roles).range([
      css('--odp-coral'), css('--odp-violet'), css('--odp-amber'),
      css('--odp-mint'), '#4cc9f0', '#f72585', '#8b8b9e', '#06d6a0']);

    // Lay the legend out first, wrapping it to as many lines as the chart's
    // own width needs, then size the SVG to fit those lines. The legend used
    // to sit on one fixed line near the bottom regardless of how much room
    // its 8 chips actually needed, so on any but the widest viewport it ran
    // past the chart's own right edge and, with the container's overflow
    // left visible, past its border too.
    const legendPad = 4, chipGap = 22, lineH = 18;
    let lx = 0, ly = 0;
    const legendItems = roles.map(r => {
      const w = r.length * 5.6 + 16;
      if (lx > 0 && lx + w > W - legendPad) { lx = 0; ly += lineH; }
      const item = { role: r, x: lx, y: ly };
      lx += w + chipGap;
      return item;
    });
    const legendH = ly + lineH + 10;
    const H = data.length * rowH + top + legendH;

    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([0, 100]).range([0, W - 160]);

    data.forEach((row, i) => {
      const g = svg.append('g').attr('transform', `translate(0,${i * rowH + top})`);
      g.append('text').attr('y', 18).attr('font-size', 12).attr('font-weight', 600)
        .text(row.portion);
      let acc = 0;
      g.selectAll('rect').data(row.roles.filter(r => r.pct > 0)).join('rect')
        .attr('class', 'odp-band')
        .attr('x', d => { const v = 160 + x(acc); acc += d.pct; return v; })
        .attr('y', 2).attr('height', 24).attr('rx', 3)
        .attr('width', d => Math.max(1, x(d.pct) - 1))
        .attr('fill', d => color(d.role))
        .on('mousemove', (ev, d) => tip.show(
          `<b>${d.role}</b><br>${d.pct.toFixed(1)}% of reuse occurrences`, ev))
        .on('mouseleave', tip.hide);
    });

    const lg = svg.append('g')
      .attr('transform', `translate(0,${data.length * rowH + top + 14})`);
    legendItems.forEach(it => {
      const g = lg.append('g').attr('transform', `translate(${it.x},${it.y})`);
      g.append('rect').attr('width', 9).attr('height', 9).attr('rx', 2)
        .attr('y', -8).attr('fill', color(it.role));
      g.append('text').attr('x', 13).attr('font-size', 9.5).attr('opacity', .8)
        .text(it.role);
    });
  }

  /* ---- most-reused patterns, top 10 per portion -------------------------- */
  async function topPatterns(el) {
    const data = await load(el);
    if (!data.length) return;
    const portions = [...new Set(data.map(d => d.portion))];
    const byPortion = new Map(portions.map(p =>
      [p, data.filter(d => d.portion === p).sort((a, b) => a.rank - b.rank)]));
    const rowH = 20, top = 34, labelW = 118;
    const maxRows = d3.max(portions, p => byPortion.get(p).length);
    const W = el.clientWidth || 900, colW = W / portions.length;
    const H = top + maxRows * rowH + 10;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const color = d3.scaleOrdinal().domain(portions)
      .range([css('--odp-coral'), css('--odp-violet'), css('--odp-mint')]);

    portions.forEach((p, pi) => {
      const rows = byPortion.get(p);
      const x0 = pi * colW + 6;
      const max = d3.max(rows, d => d.n) || 1;
      const x = d3.scaleLinear().domain([0, max]).range([0, colW - labelW - 50]);
      svg.append('text').attr('x', x0).attr('y', 16).attr('font-size', 12)
        .attr('font-weight', 700).attr('fill', color(p)).text(p);
      rows.forEach((d, i) => {
        const y = top + i * rowH;
        svg.append('text').attr('x', x0).attr('y', y + rowH - 7).attr('font-size', 10.5)
          .attr('fill', 'currentColor')
          .text(d.pattern.length > 15 ? d.pattern.slice(0, 14) + '…' : d.pattern);
        svg.append('rect').attr('class', 'odp-band')
          .attr('x', x0 + labelW).attr('y', y + 2).attr('height', rowH - 7).attr('rx', 2)
          .attr('width', Math.max(1, x(d.n)))
          .attr('fill', color(p)).attr('fill-opacity', .82)
          .on('mousemove', ev => tip.show(
            `<b>${d.pattern}</b><br>reused by ${fmt(d.n)} ${p} ontolog${d.n === 1 ? 'y' : 'ies'}`,
            ev))
          .on('mouseleave', tip.hide);
        svg.append('text').attr('x', x0 + labelW + x(d.n) + 5).attr('y', y + rowH - 7)
          .attr('font-size', 9.5).attr('opacity', .65).text(fmt(d.n));
      });
    });
  }

  /* ---- view sizes at a glance -------------------------------------------- */
  async function views(el) {
    const data = await load(el);
    if (!data.length) return;
    const metrics = [
      { key: 'ontologies', label: 'Ontologies' },
      { key: 'pairs', label: 'Pairs' },
      { key: 'patterns', label: 'Patterns' },
      { key: 'complete', label: 'Complete pairs' },
    ];
    const left = 132, colGap = 16, rowH = 24, top = 24;
    const W = el.clientWidth || 800;
    const H = top + data.length * rowH + 24;
    const colW = (W - left - colGap * (metrics.length - 1)) / metrics.length;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const portionColor = v => v.startsWith('github') ? css('--odp-coral')
      : v.startsWith('noOBO') ? css('--odp-violet') : css('--odp-mint');

    data.forEach((d, i) => {
      svg.append('text').attr('x', 0).attr('y', top + i * rowH + rowH / 2)
        .attr('dy', '.32em').attr('font-size', 11).text(d.view);
    });
    metrics.forEach((m, mi) => {
      const x0 = left + mi * (colW + colGap);
      const max = d3.max(data, d => d[m.key]) || 1;
      const x = d3.scaleLinear().domain([0, max]).range([2, Math.max(2, colW - 4)]);
      svg.append('text').attr('x', x0).attr('y', 14).attr('font-size', 10.5)
        .attr('font-weight', 600).attr('opacity', .7).text(m.label);
      data.forEach((d, i) => {
        const y = top + i * rowH + 4;
        svg.append('rect').attr('class', 'odp-band')
          .attr('x', x0).attr('y', y).attr('height', rowH - 10).attr('rx', 2)
          .attr('width', Math.max(1, x(d[m.key])))
          .attr('fill', portionColor(d.view)).attr('fill-opacity', .82)
          .on('mousemove', ev => tip.show(
            `<b>${d.view}</b><br>${m.label}: ${fmt(d[m.key])}`, ev))
          .on('mouseleave', tip.hide);
      });
    });
    svg.append('text').attr('x', left).attr('y', H - 6).attr('font-size', 9.5)
      .attr('opacity', .55)
      .text('Median entities/ontology omitted here — the OBO views’ ~32,095 ' +
            'would flatten every other bar; see the table above.');
  }

  /* ---- ontology <-> pattern reuse graph -------------------------------- */
  /* Bipartite: 68 pattern discs, ~493 ontology dots, ~759 reuse edges.
     Three things make it readable at that size:
       - edges coloured by coverage band, so "how much of the pattern is
         present" is the figure's primary signal rather than a tooltip detail;
       - a bounded-repulsion force layout (see FORCES below) instead of a
         global one, which is what stops the hairball collapsing onto the frame;
       - click-to-focus with a stats panel, so the dense view is an overview and
         the detail is one click away.                                        */
  async function reuseGraph(el) {
    const g = await load(el);
    if (!g.nodes || !g.nodes.length) return;

    const BANDS = g.bands || [];
    const bandColor = b => (BANDS[b] && BANDS[b].color) || css('--odp-line');
    const bandLabel = b => (BANDS[b] && BANDS[b].label) || '';

    const patterns = g.nodes.filter(d => d.kind === 'pattern');
    const box = el.clientWidth || 900;
    // The detail panel sits beside the graph, so the drawing width is what is
    // left after it. Measuring the whole container instead would set a viewBox
    // wider than the box the SVG actually gets, and the layout radii would be
    // computed for a canvas that never exists. Below ~640px the flex wrapper
    // stacks the panel underneath and the graph gets the full width back.
    const stacked = box < 640;
    const panelW = stacked ? box : Math.min(250, Math.max(190, box * 0.26));
    const W = Math.max(340, stacked ? box : box - panelW - 12);

    // Lay the coverage legend out before sizing the SVG, wrapping it to as many
    // lines as W needs, so it can never run past the frame's right edge.
    const lineH = 15;
    let lx = 0, ly = 0;
    const legendItems = BANDS.map(b => {
      const w = 21 + b.label.length * 5.3;
      if (lx > 0 && lx + w > W) { lx = 0; ly += lineH; }
      const it = { b, x: lx, y: ly };
      lx += w + 20;
      return it;
    });
    const legendH = ly + lineH + 12;
    const plotH = 520, H = plotH + legendH;

    const wrap = d3.select(el).append('div')
      .style('position', 'relative').style('display', 'flex')
      .style('gap', '.6rem').style('align-items', 'flex-start')
      .style('flex-wrap', 'wrap');
    const svgBox = wrap.append('div').style('flex', '1 1 340px')
      .style('min-width', '0');
    const svg = svgBox.append('svg').attr('viewBox', `0 0 ${W} ${H}`)
      .attr('role', 'img')
      .attr('aria-label', 'Ontology-to-pattern reuse graph, edges coloured by '
        + 'how much of each pattern is reused');

    const lg = svg.append('g').attr('transform', 'translate(0,11)');
    legendItems.forEach(it => {
      const gg = lg.append('g').attr('transform', `translate(${it.x},${it.y})`);
      gg.append('line').attr('x1', 0).attr('x2', 16).attr('y1', 0).attr('y2', 0)
        .attr('stroke', it.b.color)
        .attr('stroke-width', it.b.band === 4 ? 2.4 : 1.6);
      gg.append('text').attr('x', 21).attr('dy', '.32em').attr('font-size', 9.5)
        .attr('opacity', .8).text(it.b.label);
    });

    const frame = svg.append('g').attr('transform', `translate(0,${legendH})`);
    frame.append('rect').attr('x', .5).attr('y', .5)
      .attr('width', W - 1).attr('height', plotH - 1).attr('rx', 10)
      .attr('fill', 'none').attr('stroke', css('--odp-line'));
    // Clip so a dragged node cannot be painted outside the frame. The layout
    // keeps nodes inside on its own; this only bounds the drag interaction,
    // which is user-driven and has no force acting against it.
    const clipId = 'odp-rg-clip-' + Math.abs(W | 0) + '-' + patterns.length;
    frame.append('clipPath').attr('id', clipId).append('rect')
      .attr('x', 0).attr('y', 0).attr('width', W).attr('height', plotH).attr('rx', 10);
    const root = frame.append('g').attr('clip-path', `url(#${clipId})`)
      .append('g');
    frame.append('rect').attr('width', W).attr('height', plotH)
      .attr('fill', 'transparent').style('cursor', 'default')
      .on('click', () => focus(null))
      .lower();

    svg.call(d3.zoom().scaleExtent([.5, 8])
      .filter(ev => !ev.target.closest('.odp-node'))
      .on('zoom', ev => root.attr('transform', ev.transform)));

    const rr = d3.scaleSqrt()
      .domain([1, d3.max(patterns, d => d.v) || 1]).range([4, 26]);
    const radiusOf = d => d.kind === 'pattern' ? rr(d.v) : 3.2;

    /* ---- adjacency, on the raw ids -------------------------------------
       Built before d3.forceLink rewrites link.source/target into node object
       references, so these stay plain strings.                            */
    const neighbors = new Map();
    const covByPattern = new Map();
    g.links.forEach(l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      if (!neighbors.has(s)) neighbors.set(s, new Set());
      if (!neighbors.has(t)) neighbors.set(t, new Set());
      neighbors.get(s).add(t); neighbors.get(t).add(s);
      if (!covByPattern.has(t)) covByPattern.set(t, []);
      covByPattern.get(t).push(l);
    });

    /* ---- FORCES ---------------------------------------------------------
       The previous version used a global forceManyBody at strength -52 with
       no distanceMax. On 561 nodes that is ~157k repelling pairs, all acting
       at unbounded range, so the graph inflated until every node sat against
       the frame and the clamp pinned it there -- the "everything gathered on
       the boundary" symptom. Bounding repulsion with distanceMax makes it a
       local force: nodes push their neighbours apart without the whole graph
       expanding. The radial force then does the global arranging, which is
       what a bipartite graph wants: patterns pulled toward the centre by
       their degree, ontologies pushed to a ring outside them.             */
    const cx = W / 2, cy = plotH / 2;
    const maxV = d3.max(patterns, d => d.v) || 1;
    const ringR = Math.min(W, plotH) * 0.46;
    const targetR = d => d.kind === 'ontology'
      ? ringR
      // A pattern's radius falls with how many ontologies reuse it, so DUL and
      // situation land near the middle and singletons sit out by the ring.
      : ringR * (0.62 - 0.52 * Math.sqrt(d.v / maxV));

    const sim = d3.forceSimulation(g.nodes)
      .force('link', d3.forceLink(g.links).id(d => d.id)
        .distance(d => d.target.kind === 'pattern' ? 26 + rr(d.target.v) : 30)
        .strength(0.22))
      .force('charge', d3.forceManyBody().strength(-26).distanceMax(110))
      .force('r', d3.forceRadial(targetR, cx, cy)
        .strength(d => d.kind === 'pattern' ? 0.34 : 0.12))
      .force('collide', d3.forceCollide(d => radiusOf(d) + 2.2).iterations(2))
      .velocityDecay(0.36)
      .alpha(1).alphaDecay(0.0135);

    const link = root.append('g').attr('fill', 'none')
      .selectAll('line').data(g.links).join('line')
      .attr('stroke', d => bandColor(d.band))
      .attr('stroke-width', d => d.band === 4 ? 1.9 : 1.1)
      .attr('stroke-opacity', d => d.band === 4 ? .85 : .5);

    const node = root.append('g').selectAll('circle').data(g.nodes).join('circle')
      .attr('class', 'odp-node')
      .attr('r', radiusOf)
      .attr('fill', d => d.kind === 'pattern' ? css('--odp-violet')
        : d.portion === 'GitHub' ? css('--odp-coral') : css('--odp-amber'))
      .attr('fill-opacity', d => d.kind === 'pattern' ? .95 : .62)
      .attr('stroke', d => d.kind === 'pattern' ? css('--odp-panel') : 'none')
      .attr('stroke-width', 1.3)
      .style('cursor', d => d.kind === 'pattern' ? 'pointer' : 'grab')
      .on('mousemove', (ev, d) => tip.show(hover(d), ev))
      .on('mouseleave', tip.hide)
      .on('click', (ev, d) => {
        ev.stopPropagation();
        focus(d.kind === 'pattern' ? d.label : null);
      })
      .call(d3.drag()
        .on('start', (ev, d) => {
          if (!ev.active) sim.alphaTarget(.12).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on('end', (ev, d) => {
          if (!ev.active) sim.alphaTarget(0);
          d.fx = null; d.fy = null;
        }));

    function hover(d) {
      if (d.kind !== 'pattern') {
        return `<b>${d.label}</b><br>${d.portion} ontology`;
      }
      const parts = [];
      if (d.n_classes != null) parts.push(`${d.n_classes} C`);
      if (d.n_object_properties != null) parts.push(`${d.n_object_properties} OP`);
      if (d.n_datatype_properties != null) parts.push(`${d.n_datatype_properties} DP`);
      return `<b>${d.label}</b><br>reused by ${fmt(d.v)} ontolog`
        + (d.v === 1 ? 'y' : 'ies')
        + (parts.length ? `<br>${parts.join(' · ')}` : '')
        + '<br><i>click for detail</i>';
    }

    /* ---- labels: only the patterns big enough to name ------------------- */
    const named = patterns.filter(d => d.v >= 8);
    const label = root.append('g').selectAll('g').data(named)
      .join('g').attr('pointer-events', 'none');
    label.append('rect').attr('fill', css('--odp-panel')).attr('opacity', .82)
      .attr('rx', 3);
    label.append('text').attr('font-size', 10).attr('font-weight', 600)
      .attr('text-anchor', 'middle').attr('fill', 'currentColor')
      .text(d => d.label)
      .each(function () {
        const b = this.getBBox();
        d3.select(this.previousSibling)
          .attr('x', b.x - 3).attr('y', b.y - 1)
          .attr('width', b.width + 6).attr('height', b.height + 2);
      });

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    sim.on('tick', () => {
      g.nodes.forEach(d => {
        const r = radiusOf(d) + 3;
        d.x = clamp(d.x, r, W - r);
        d.y = clamp(d.y, r, plotH - r);
      });
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('transform', d => `translate(${d.x},${d.y - rr(d.v) - 9})`);
    });
    setTimeout(() => sim.stop(), 12000);   // settle, then stop burning CPU

    /* ---- detail panel --------------------------------------------------- */
    const panel = wrap.append('div')
      .style('flex', `0 1 ${panelW}px`).style('min-width', '180px')
      .style('font-size', '.8rem').style('line-height', '1.45');

    function bar(rows, total) {
      // Coverage mix for one pattern, as a single stacked rule.
      const segs = rows.map(r =>
        `<span style="display:inline-block;width:${(100 * r.n / total).toFixed(2)}%;`
        + `height:8px;background:${bandColor(r.band)}" title="${bandLabel(r.band)}: `
        + `${r.n}"></span>`).join('');
      return `<div style="display:flex;border-radius:4px;overflow:hidden;`
        + `margin:.3rem 0 .45rem">${segs}</div>`;
    }

    function idle() {
      const complete = g.links.filter(l => l.band === 4).length;
      panel.html(
        `<div style="opacity:.75"><b>${fmt(patterns.length)} patterns</b> · `
        + `<b>${fmt(g.nodes.length - patterns.length)} ontologies</b> · `
        + `<b>${fmt(g.links.length)} reuse pairs</b>, of which `
        + `<b>${fmt(complete)}</b> are complete.<br><br>`
        + `Dots are ontologies: <span style="color:${css('--odp-coral')}">●</span> `
        + `GitHub, <span style="color:${css('--odp-amber')}">●</span> Paper &amp; `
        + `Repository. Discs are patterns, sized by how many ontologies reuse `
        + `them and pulled toward the centre as that count rises.<br><br>`
        + `<i>Click a pattern disc, or a chip below, for its detail.</i></div>`);
    }

    function detail(d) {
      const links = covByPattern.get(d.id) || [];
      const byBand = d3.rollups(links, v => v.length, l => l.band)
        .map(([band, n]) => ({ band, n })).sort((a, b) => a.band - b.band);
      const covs = links.map(l => l.coverage);
      const med = covs.length ? d3.median(covs) : null;
      const decl = [];
      if (d.n_classes != null) decl.push(['classes', d.n_classes]);
      if (d.n_object_properties != null) decl.push(['object properties', d.n_object_properties]);
      if (d.n_datatype_properties != null) decl.push(['data properties', d.n_datatype_properties]);
      if (d.n_individuals) decl.push(['named individuals', d.n_individuals]);

      // n_entities is the release's WIDE entity set and is what coverage is
      // measured against; C+OP+DP is the declared subset. Showing both, and
      // the gap between them, is more honest than showing one and implying it
      // is the whole story.
      const wide = d.n_entities != null ? d.n_entities : null;
      const declSum = decl.reduce((a, r) => a + r[1], 0);
      const rangeNote = (d.gt_min != null && d.gt_max != null && d.gt_min !== d.gt_max)
        ? `<div style="opacity:.6;font-size:.72rem;margin-top:.2rem">`
        + `Local copies disagree: ${d.gt_min}&ndash;${d.gt_max} entities across `
        + `the ${fmt(links.length)} pairs. Coverage uses each ontology's own copy.</div>`
        : '';

      panel.html(
        `<div style="font-weight:700;font-size:.92rem;word-break:break-word">`
        + `${d.label}</div>`
        + `<div style="opacity:.7;margin:.15rem 0 .5rem">reused by `
        + `<b>${fmt(d.v)}</b> ontolog${d.v === 1 ? 'y' : 'ies'}`
        + (d.github != null && d.noOBO != null
          ? ` — ${fmt(d.github)} GitHub, ${fmt(d.noOBO)} Paper &amp; Repository` : '')
        + `</div>`
        + (byBand.length
          ? `<div style="font-size:.72rem;opacity:.7">coverage mix</div>`
          + bar(byBand, links.length)
          + byBand.map(r => `<div style="display:flex;gap:.4rem;align-items:center">`
            + `<span style="width:9px;height:9px;border-radius:2px;background:`
            + `${bandColor(r.band)};display:inline-block"></span>`
            + `<span style="opacity:.8">${bandLabel(r.band)}</span>`
            + `<b style="margin-left:auto">${r.n}</b></div>`).join('')
          : '')
        + (med != null
          ? `<div style="margin-top:.45rem;opacity:.8">median coverage `
          + `<b>${med.toFixed(1)}%</b></div>` : '')
        + (decl.length
          ? `<div style="margin-top:.6rem;font-size:.72rem;opacity:.7">declares</div>`
          + decl.map(([k, v]) => `<div style="display:flex"><span style="opacity:.8">`
            + `${k}</span><b style="margin-left:auto">${fmt(v)}</b></div>`).join('')
          : '')
        + (wide != null
          ? `<div style="display:flex;margin-top:.3rem"><span style="opacity:.8">`
          + `entities (coverage denominator)</span>`
          + `<b style="margin-left:auto">${fmt(wide)}</b></div>`
          + (wide > declSum
            ? `<div style="opacity:.6;font-size:.72rem;margin-top:.2rem">`
            + `${fmt(wide - declSum)} of those are referenced but never declared `
            + `— restriction fillers, domains and ranges.</div>`
            : '')
          : '')
        + (d.n_triples != null
          ? `<div style="display:flex;margin-top:.3rem"><span style="opacity:.8">`
          + `triples (annotations stripped)</span>`
          + `<b style="margin-left:auto">${fmt(d.n_triples)}</b></div>` : '')
        + rangeNote);
    }

    let current = null;
    function focus(patternName) {
      current = patternName || null;
      const target = current ? `p:${current}` : null;
      const active = target ? (neighbors.get(target) || new Set()) : null;
      node.attr('opacity', d => !target || d.id === target || active.has(d.id) ? 1 : .08);
      link.attr('stroke-opacity', d => {
        if (!target) return d.band === 4 ? .85 : .5;
        return (d.source.id === target || d.target.id === target)
          ? .95 : .03;
      });
      label.attr('opacity', d => !target ? 1 : (d.id === target ? 1 : .08));
      d3.selectAll('.odp-chip').each(function () {
        this.setAttribute('aria-pressed',
          this.dataset.pattern === current ? 'true' : 'false');
      });
      const n = target && g.nodes.find(x => x.id === target);
      if (n) detail(n); else idle();
    }

    idle();
    el.__focus = focus;
  }

  /* ---- forest plot: paired improvements with CIs ------------------------ */
  async function forest(el) {
    const data = await load(el);
    if (!data.length) return;
    const W = el.clientWidth || 800, rowH = 52, H = data.length * rowH + 46;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const lo = Math.min(0, d3.min(data, d => d.lo)), hi = d3.max(data, d => d.hi);
    const x = d3.scaleLinear().domain([lo, hi]).nice().range([300, W - 30]);

    svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 12)
      .attr('y2', H - 34).attr('stroke', css('--odp-line')).attr('stroke-dasharray', '3 3');
    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(0,${H - 30})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('+.2f')));

    data.forEach((d, i) => {
      const y = i * rowH + 30;
      const c = d.sig ? css('--odp-mint') : css('--odp-line');
      svg.append('text').attr('x', 0).attr('y', y + 4).attr('font-size', 11)
        .text(d.label.length > 46 ? d.label.slice(0, 44) + '…' : d.label);
      svg.append('line').attr('x1', x(d.lo)).attr('x2', x(d.hi))
        .attr('y1', y).attr('y2', y).attr('stroke', c).attr('stroke-width', 5)
        .attr('stroke-linecap', 'round').attr('opacity', .55);
      svg.append('circle').attr('cx', x(d.delta)).attr('cy', y).attr('r', 6)
        .attr('fill', d.sig ? css('--odp-mint') : css('--odp-coral'))
        .on('mousemove', ev => tip.show(
          `<b>Δ MAP ${d.delta >= 0 ? '+' : ''}${d.delta.toFixed(3)}</b><br>` +
          `95% CI [${d.lo.toFixed(3)}, ${d.hi.toFixed(3)}]<br>` +
          (d.sig ? 'significant' : 'not significant'), ev))
        .on('mouseleave', tip.hide);
    });
  }

  /* ---- dumbbell: oracle vs unsupervised F1 ------------------------------ */
  async function oracleGap(el) {
    const data = await load(el);
    if (!data.length) return;
    const W = el.clientWidth || 800, rowH = 40, H = data.length * rowH + 46;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([0, d3.max(data, d => d.oracle) * 1.12])
      .range([230, W - 40]);
    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(0,${H - 30})`).call(d3.axisBottom(x).ticks(6));

    data.forEach((d, i) => {
      const y = i * rowH + 26;
      svg.append('text').attr('y', y + 4).attr('font-size', 11)
        .text(`${d.portion} · ${d.method === 'owl2vec' ? 'OWL2Vec*' : 'RDF2Vec'} · ` +
              `${d.clusterer === 'kmeans' ? 'K-Means' : 'HDBSCAN'}`);
      svg.append('line').attr('x1', x(d.unsup)).attr('x2', x(d.oracle))
        .attr('y1', y).attr('y2', y).attr('stroke', css('--odp-amber'))
        .attr('stroke-width', 3).attr('opacity', .5);
      svg.append('circle').attr('cx', x(d.unsup)).attr('cy', y).attr('r', 5.5)
        .attr('fill', css('--odp-mint'))
        .on('mousemove', ev => tip.show(
          `<b>Unsupervised F₁ ${d.unsup.toFixed(3)}</b><br>achievable · n=${d.n}`, ev))
        .on('mouseleave', tip.hide);
      svg.append('circle').attr('cx', x(d.oracle)).attr('cy', y).attr('r', 5.5)
        .attr('fill', css('--odp-coral'))
        .on('mousemove', ev => tip.show(
          `<b>Oracle F₁ ${d.oracle.toFixed(3)}</b><br>upper bound · n=${d.n}`, ev))
        .on('mouseleave', tip.hide);
    });
  }

  /* ---- version history: the dataset over time -------------------------- */
  /* One point per version on a time axis. A version before the quality gate
     has no comparable counts, so it is a marker on the axis, not a point. */
  async function history(el) {
    const data = await load(el);
    if (!data.length) return;
    const W = el.clientWidth || 800, H = 300, m = { t: 26, r: 24, b: 52, l: 54 };
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`)
      .attr('role', 'img').attr('aria-label', 'Corpus size by version');
    const dates = data.map(d => new Date(d.date));
    const [lo, hi] = d3.extent(dates);
    const pad = Math.max(20 * 864e5, (hi - lo) * 0.06);
    const x = d3.scaleTime().domain([new Date(+lo - pad), new Date(+hi + pad)])
      .range([m.l, W - m.r]);
    const counted = data.filter(d => d.comparable);
    const max = d3.max(counted, d => Math.max(d.discovered, d.kept, d.pairs)) || 1;
    const y = d3.scaleLinear().domain([0, max * 1.12]).nice().range([H - m.b, m.t]);
    const series = [
      { key: 'discovered', label: 'Corpus', color: css('--odp-violet') },
      { key: 'pairs', label: 'Pairs', color: css('--odp-mint') },
      { key: 'kept', label: 'Pass the gate', color: css('--odp-coral') },
    ];

    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(Math.min(8, Math.max(2, W / 120))));
    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));

    data.forEach(d => {
      const cx = x(new Date(d.date));
      svg.append('line').attr('x1', cx).attr('x2', cx).attr('y1', m.t)
        .attr('y2', H - m.b).attr('stroke', css('--odp-line'))
        .attr('stroke-dasharray', d.comparable ? null : '3 4');
      svg.append('text').attr('x', cx).attr('y', H - m.b + 32)
        .attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600)
        .text('v' + d.version + (d.comparable ? '' : ' · pre-gate'));
    });

    const line = k => d3.line().x(d => x(new Date(d.date))).y(d => y(d[k]));
    series.forEach(s => {
      if (counted.length > 1) {
        svg.append('path').datum(counted).attr('fill', 'none').attr('stroke', s.color)
          .attr('stroke-width', 2.5).attr('stroke-linecap', 'round').attr('d', line(s.key));
      }
      svg.selectAll(null).data(counted).join('circle')
        .attr('cx', d => x(new Date(d.date))).attr('cy', d => y(d[s.key])).attr('r', 5)
        .attr('fill', s.color)
        .on('mousemove', (ev, d) => tip.show(
          `<b>v${d.version}</b> · ${d.date}<br>${s.label}: ${fmt(d[s.key])}` +
          (d.added ? `<br>this version added ${fmt(d.added.kept)} gate-passing ` +
                     `ontologies, ${fmt(d.added.pairs)} pairs` : ''), ev))
        .on('mouseleave', tip.hide);
      const last = counted[counted.length - 1];
      if (last) {
        svg.append('text').attr('x', x(new Date(last.date)) - 9).attr('y', y(last[s.key]) - 9)
          .attr('text-anchor', 'end').attr('font-size', 11).attr('font-weight', 600)
          .attr('fill', s.color).text(`${s.label} ${fmt(last[s.key])}`);
      }
    });
  }

  /* ---- what each version added ----------------------------------------- */
  async function additions(el) {
    const data = await load(el);
    if (!data.length) return;
    const W = el.clientWidth || 800, H = 250, m = { t: 30, r: 16, b: 40, l: 54 };
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`)
      .attr('role', 'img').attr('aria-label', 'Ontologies added by each version');
    const x = d3.scaleBand().domain(data.map(d => 'v' + d.version))
      .range([m.l, W - m.r]).padding(0.35);
    const max = d3.max(data, d => d.added ? d.added.ontologies : 0) || 1;
    const y = d3.scaleLinear().domain([0, max * 1.15]).nice().range([H - m.b, m.t]);
    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x));
    svg.append('g').attr('class', 'odp-axis')
      .attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5));

    data.forEach(d => {
      const bx = x('v' + d.version), bw = x.bandwidth();
      if (!d.added) {
        svg.append('text').attr('x', bx + bw / 2).attr('y', H - m.b - 10)
          .attr('text-anchor', 'middle').attr('font-size', 10.5).attr('opacity', .6)
          .text(d.comparable ? 'baseline' : 'first deposit');
        return;
      }
      const parts = [
        { n: d.added.kept, label: 'pass the gate', color: css('--odp-coral'), op: .9 },
        { n: d.added.excluded, label: 'excluded by the gate', color: css('--odp-violet'), op: .45 },
      ];
      let acc = 0;
      parts.forEach(p => {
        svg.append('rect').attr('class', 'odp-band')
          .attr('x', bx).attr('width', bw).attr('rx', 2)
          .attr('y', y(acc + p.n)).attr('height', Math.max(0, y(acc) - y(acc + p.n)))
          .attr('fill', p.color).attr('fill-opacity', p.op)
          .on('mousemove', ev => tip.show(
            `<b>v${d.version}</b><br>${fmt(p.n)} added ontologies ${p.label}`, ev))
          .on('mouseleave', tip.hide);
        acc += p.n;
      });
      svg.append('text').attr('class', 'odp-value').attr('x', bx + bw / 2)
        .attr('y', y(acc) - 8).attr('text-anchor', 'middle').attr('font-size', 11)
        .text(`+${fmt(d.added.kept)} · +${fmt(d.added.pairs)} pairs`);
    });

    const legend = svg.append('g').attr('transform', `translate(${m.l},12)`);
    [['Pass the gate', css('--odp-coral'), .9],
     ['Excluded, kept with its reason', css('--odp-violet'), .45]].forEach(([t, c, o], i) => {
      const g = legend.append('g').attr('transform', `translate(${i * 150},0)`);
      g.append('rect').attr('width', 10).attr('height', 10).attr('y', -9).attr('rx', 2)
        .attr('fill', c).attr('fill-opacity', o);
      g.append('text').attr('x', 15).attr('font-size', 10.5).text(t);
    });
  }

  /* ---- boot ------------------------------------------------------------- */
  const CHARTS = {
    'odp-funnel': funnel, 'odp-rules': rules, 'odp-thresholds': thresholds,
    'odp-forms': forms, 'odp-toppat': topPatterns, 'odp-views': views,
    'odp-constellation': reuseGraph,
    'odp-uc1': oracleGap, 'odp-uc2': forest,
    'odp-history': history, 'odp-additions': additions,
  };

  function render() {
    Object.entries(CHARTS).forEach(([id, fn]) => {
      const el = document.getElementById(id);
      if (!el || el.dataset.rendered) return;
      el.dataset.rendered = '1';
      fn(el).catch(e => {
        el.dataset.rendered = '';
        el.innerHTML = '<p style="opacity:.6;font-size:.85rem">' +
          'Figure unavailable — the payload has not been generated for this ' +
          'release yet.</p>';
        console.warn('[odp]', id, e);
      });
    });
    // pattern chips filter the constellation
    document.querySelectorAll('.odp-chip').forEach(c => {
      c.addEventListener('click', () => {
        const on = c.getAttribute('aria-pressed') === 'true';
        document.querySelectorAll('.odp-chip')
          .forEach(o => o.setAttribute('aria-pressed', 'false'));
        const next = on ? null : c.dataset.pattern;
        c.setAttribute('aria-pressed', next ? 'true' : 'false');
        const viz = document.getElementById('odp-constellation');
        if (viz && viz.__focus) viz.__focus(next);
      });
    });
  }

  if (typeof document$ !== 'undefined') { document$.subscribe(render); }
  else { document.addEventListener('DOMContentLoaded', render); }
})();
