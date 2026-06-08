// Results page: read ?job=, poll the API, render charts + insights.

(function () {
  const $ = (id) => document.getElementById(id);
  const loading = $("loading");
  const errorView = $("errorView");
  const resultsView = $("resultsView");

  const params = new URLSearchParams(location.search);
  const jobId = params.get("job");

  const INSIGHT_ICON = { positive: "✅", tip: "💡", warning: "⚠️" };

  function showError(msg) {
    loading.classList.add("hidden");
    resultsView.classList.add("hidden");
    errorView.classList.remove("hidden");
    $("errorMsg").textContent = msg;
  }

  function render(data) {
    loading.classList.add("hidden");
    resultsView.classList.remove("hidden");

    // index ring + value
    const idx = data.engagement_index ?? 0;
    $("indexVal").textContent = Math.round(idx);
    window.BCCharts.indexRing($("indexRing"), idx);

    // meta line
    const m = data.meta || {};
    const bits = [`Modality: ${data.modality}`];
    if (m.duration_s) bits.push(`${Number(m.duration_s).toFixed(0)}s`);
    if (m.processing_ms) bits.push(`analyzed in ${(m.processing_ms / 1000).toFixed(1)}s`);
    $("metaLine").textContent = bits.join(" · ");
    if (m.experimental) $("expPill").classList.remove("hidden");

    // charts
    window.BCCharts.networks($("networkChart"), data.networks || []);
    window.BCCharts.timeline($("timelineChart"), data.timeline || { t: [], overall: [] });

    // brain-activation maps (optional; hide the card when absent/empty)
    const maps = data.brain_maps || [];
    if (maps.length) {
      const grid = $("brainMaps");
      grid.innerHTML = "";
      maps.forEach((bm) => {
        const fig = document.createElement("figure");
        fig.className = "brain-cell";
        const img = document.createElement("img");
        img.src = bm.image;
        img.alt = `${bm.label} network activation`;
        img.loading = "lazy";
        const cap = document.createElement("figcaption");
        cap.textContent = bm.label;
        fig.append(img, cap);
        grid.appendChild(fig);
      });
      $("brainMapsCard").classList.remove("hidden");
    }

    // insights
    const wrap = $("insights");
    wrap.innerHTML = "";
    (data.insights || []).forEach((ins) => {
      const el = document.createElement("div");
      el.className = `insight ${ins.severity}`;
      el.innerHTML = `<span class="i-ico">${INSIGHT_ICON[ins.severity] || "•"}</span>
        <div><h4></h4><p></p></div>`;
      el.querySelector("h4").textContent = ins.title;
      el.querySelector("p").textContent = ins.body;
      wrap.appendChild(el);
    });
  }

  if (!jobId) {
    showError("No job specified. Start a new analysis.");
    return;
  }

  window.BC.pollJob(jobId, {
    onTick: (d) => {
      if (d.status === "running") $("loadStatus").textContent = "Analyzing your content… (this can take a minute or two)";
    },
  })
    .then((data) => {
      if (data.status === "error") return showError(data.error || "Analysis failed.");
      render(data);
    })
    .catch((e) => showError(e.message || "Something went wrong."));
})();
