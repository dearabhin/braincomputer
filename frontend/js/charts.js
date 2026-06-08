// Chart.js renderers for the results page. Exposes window.BCCharts.

// Functional colouring (Spotify-style): green = strong, orange = mid, red = weak.
function scoreColor(v) {
  if (v >= 65) return "#1ed760"; // green
  if (v < 40) return "#f3727f";  // red
  return "#ffa42b";              // orange
}

const GRID = "rgba(255,255,255,0.07)";
const TICK = "#b3b3b3";

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
          backgroundColor: networks.map((n) => scoreColor(n.score)),
          borderRadius: 6,
          maxBarThickness: 26,
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
          x: { min: 0, max: 100, grid: { color: GRID }, ticks: { color: TICK } },
          y: { grid: { display: false }, ticks: { color: "#ffffff" } },
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
          borderColor: "#1ed760",
          backgroundColor: "rgba(30,215,96,0.15)",
          fill: true, tension: 0.35, pointRadius: 2, borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` ${c.raw}/100` } } },
        scales: {
          y: { min: 0, max: 100, grid: { color: GRID }, ticks: { color: TICK } },
          x: { grid: { display: false }, ticks: { color: TICK, maxTicksLimit: 10 } },
        },
      },
    });
  },
};

window.BCCharts = BCCharts;
