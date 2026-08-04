/* ==========================================================================
   ODP-Reuse — interactive figures.
   Every chart reads a JSON payload emitted by pipeline/gen_site.py, so the
   figures cannot drift from the data. D3 only; no chart library.
   ========================================================================== */
(function () {
  'use strict';

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
  const css = v => getComputedStyle(document.documentElement)
    .getPropertyValue(v).trim();

  async function load(el) {
    const r = await fetch(el.dataset.src);
    if (!r.ok) throw new Error(el.dataset.src + ': ' + r.status);
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
    const W = el.clientWidth || 800, rowH = 46, H = data.length * rowH + 40;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scaleLinear().domain([0, 100]).range([0, W - 160]);
    const color = d3.scaleOrdinal().domain(roles).range([
      css('--odp-coral'), css('--odp-violet'), css('--odp-amber'),
      css('--odp-mint'), '#4cc9f0', '#f72585', '#8b8b9e', '#06d6a0']);

    data.forEach((row, i) => {
      const g = svg.append('g').attr('transform', `translate(0,${i * rowH + 22})`);
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
    const lg = svg.append('g').attr('transform', `translate(160,${H - 10})`);
    let off = 0;
    roles.forEach(r => {
      const g = lg.append('g').attr('transform', `translate(${off},0)`);
      g.append('rect').attr('width', 9).attr('height', 9).attr('rx', 2)
        .attr('y', -8).attr('fill', color(r));
      g.append('text').attr('x', 13).attr('font-size', 9.5).attr('opacity', .8).text(r);
      off += r.length * 5.6 + 30;
    });
  }

  /* ---- ontology <-> pattern constellation ------------------------------ */
  async function constellation(el) {
    const g = await load(el);
    if (!g.nodes || !g.nodes.length) return;
    const W = el.clientWidth || 900, H = 460;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`);
    const root = svg.append('g');
    svg.call(d3.zoom().scaleExtent([.25, 5])
      .on('zoom', ev => root.attr('transform', ev.transform)));

    const rr = d3.scaleSqrt()
      .domain([1, d3.max(g.nodes, d => d.v) || 1]).range([3, 22]);
    const sim = d3.forceSimulation(g.nodes)
      .force('link', d3.forceLink(g.links).id(d => d.id).distance(46).strength(.35))
      .force('charge', d3.forceManyBody().strength(-58))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(d => rr(d.v) + 3));

    const link = root.append('g').attr('stroke', css('--odp-line'))
      .selectAll('line').data(g.links).join('line')
      .attr('stroke-width', d => d.complete ? 1.8 : .6)
      .attr('stroke', d => d.complete ? css('--odp-mint') : css('--odp-line'));

    const node = root.append('g').selectAll('circle').data(g.nodes).join('circle')
      .attr('r', d => d.kind === 'pattern' ? rr(d.v) : 3)
      .attr('fill', d => d.kind === 'pattern' ? css('--odp-violet') : css('--odp-coral'))
      .attr('fill-opacity', d => d.kind === 'pattern' ? .92 : .55)
      .style('cursor', 'grab')
      .on('mousemove', (ev, d) => tip.show(
        d.kind === 'pattern'
          ? `<b>${d.label}</b><br>reused by ${d.v} ontolog${d.v === 1 ? 'y' : 'ies'}`
          : `<b>${d.label}</b><br>ontology`, ev))
      .on('mouseleave', tip.hide)
      .call(d3.drag()
        .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(.25).restart();
                                  d.fx = d.x; d.fy = d.y; })
        .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0);
                                d.fx = null; d.fy = null; }));

    const label = root.append('g').selectAll('text')
      .data(g.nodes.filter(d => d.kind === 'pattern' && d.v >= 4)).join('text')
      .attr('font-size', 10).attr('font-weight', 600)
      .attr('text-anchor', 'middle').attr('pointer-events', 'none')
      .attr('fill', 'currentColor').text(d => d.label);

    sim.on('tick', () => {
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('x', d => d.x).attr('y', d => d.y - rr(d.v) - 4);
    });
    setTimeout(() => sim.stop(), 9000);   // settle, then stop burning CPU
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

  /* ---- boot ------------------------------------------------------------- */
  const CHARTS = {
    'odp-funnel': funnel, 'odp-rules': rules, 'odp-thresholds': thresholds,
    'odp-forms': forms, 'odp-constellation': constellation,
    'odp-uc1': oracleGap, 'odp-uc2': forest,
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
        c.setAttribute('aria-pressed', on ? 'false' : 'true');
      });
    });
  }

  if (typeof document$ !== 'undefined') { document$.subscribe(render); }
  else { document.addEventListener('DOMContentLoaded', render); }
})();
