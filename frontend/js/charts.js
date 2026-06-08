// Chart.js renderers for the results page. Exposes window.BCCharts.

const NET_COLORS = {
  visual: "#7c5cff",
  auditory: "#21d4fd",
  language: "#2ecc71",
  emotional_social: "#ff5e7e",
  default_mode: "#ffb648",
  multisensory: "#b56cff",
};

function scoreColor(v) {
  if (v >= 65) return "#2ecc71";
  if (v <= 38) return "#ff5e7e";
  return "#21d4fd";
}

const BCCharts = {
  indexRing(canvas, value) {
    return new Chart(canvas, {
      type: "doughnut",
      data: {
        datasets: [{
          data: [value, 100 - value],
          backgroundColor: [scoreColor(value), "rgba(255,255,255,0.08)"],
          borderWidth: 0,
        }],
      },
      options: {
        cutout: "78%", responsive: false,
        plugins: { tooltip: { enabled: false }, legend: { display: false } },
      },
    });
  },

  networks(canvas, networks) {
    return new Chart(canvas, {
      type: "bar",
      data: {
        labels: networks.map((n) => n.label),
        datasets: [{
          data: networks.map((n) => n.score),
          backgroundColor: networks.map((n) => NET_COLORS[n.key] || "#7c5cff"),
          borderRadius: 8,
          maxBarThickness: 28,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => ` ${c.raw}/100`,
              afterLabel: (c) => networks[c.dataIndex].description || "",
            },
          },
        },
        scales: {
          x: { min: 0, max: 100, grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#9aa3c0" } },
          y: { grid: { display: false }, ticks: { color: "#e8ebf5" } },
        },
      },
    });
  },

  timeline(canvas, timeline) {
    const t = timeline.t || [];
    return new Chart(canvas, {
      type: "line",
      data: {
        labels: t.map((s) => `${s}s`),
        datasets: [{
          label: "Overall",
          data: timeline.overall || [],
          borderColor: "#21d4fd",
          backgroundColor: "rgba(33,212,253,0.15)",
          fill: true, tension: 0.35, pointRadius: 2, borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.raw}/100` } } },
        scales: {
          y: { min: 0, max: 100, grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "#9aa3c0" } },
          x: { grid: { display: false }, ticks: { color: "#9aa3c0", maxTicksLimit: 10 } },
        },
      },
    });
  },
};

window.BCCharts = BCCharts;
