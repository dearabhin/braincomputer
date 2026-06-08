// Upload-page logic: drag-drop, file selection, submit -> redirect to results.

(function () {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("file");
  const chip = document.getElementById("filechip");
  const nameEl = document.getElementById("fileName");
  const metaEl = document.getElementById("fileMeta");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const errEl = document.getElementById("error");

  let selected = null;

  const MAX_BYTES = 100 * 1024 * 1024;
  const OK_EXT = [
    "mp4","mov","webm","mkv","m4v","avi",
    "jpg","jpeg","png","webp","bmp","gif",
    "mp3","wav","m4a","aac","ogg","flac",
    "txt","md",
  ];

  function fmtSize(b) {
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(0) + " KB";
    return (b / 1048576).toFixed(1) + " MB";
  }

  function setFile(file) {
    errEl.textContent = "";
    if (!file) return;
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (!OK_EXT.includes(ext)) {
      errEl.textContent = `Unsupported file type ".${ext}".`;
      return;
    }
    if (file.size > MAX_BYTES) {
      errEl.textContent = "File is larger than 100 MB.";
      return;
    }
    if (file.size === 0) {
      errEl.textContent = "That file is empty.";
      return;
    }
    selected = file;
    nameEl.textContent = file.name;
    metaEl.textContent = `${fmtSize(file.size)} · ${ext.toUpperCase()}`;
    chip.classList.add("show");
    clearBtn.classList.remove("hidden");
    analyzeBtn.disabled = false;
  }

  function clearFile() {
    selected = null;
    input.value = "";
    chip.classList.remove("show");
    clearBtn.classList.add("hidden");
    analyzeBtn.disabled = true;
    errEl.textContent = "";
  }

  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
  input.addEventListener("change", () => setFile(input.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  clearBtn.addEventListener("click", clearFile);

  analyzeBtn.addEventListener("click", async () => {
    if (!selected) return;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Uploading…";
    errEl.textContent = "";
    try {
      const { job_id } = await window.BC.analyze(selected);
      location.href = `results.html?job=${encodeURIComponent(job_id)}`;
    } catch (e) {
      errEl.textContent = e.message || "Upload failed.";
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze engagement →";
    }
  });
})();
