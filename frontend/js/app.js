// =====================================================================
// CONFIG — edit these three values to match your Render backend
// =====================================================================
const CONFIG = {
  // Your predict endpoint on Render
  API_URL: "https://tomato-classifier.onrender.com/predict",

  // The form field name your backend expects the file under.
  // Your FastAPI signature is `async def predict(file: UploadFile)`,
  // so this must be "file" to match the parameter name.
  FIELD_NAME: "file",

  // The 6 class names, IN THE EXACT ORDER your model's `classes` list
  // uses server-side (same order the logits/argmax correspond to).
  CLASS_NAMES: [
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4",
    "Class 5",
    "Class 6",
  ],
};
// =====================================================================

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const scanCard = document.getElementById("scan-card");
  const previewFrame = document.getElementById("preview-frame");
  const previewImg = document.getElementById("preview-img");
  const scanLine = document.getElementById("scan-line");
  const scanStatus = document.getElementById("scan-status");
  const analyzeBtn = document.getElementById("analyze-btn");
  const resetBtn = document.getElementById("reset-btn");
  const resultEl = document.getElementById("result");
  const errorBox = document.getElementById("error-box");
  const errorMsg = document.getElementById("error-msg");

  let currentFile = null;
  let objectUrl = null;
  let slowHintTimer = null;

  // ---------- helpers ----------

  function resetToIdle() {
    currentFile = null;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    fileInput.value = "";
    previewFrame.classList.add("hidden");
    dropzone.classList.remove("hidden");
    analyzeBtn.classList.add("hidden");
    resetBtn.classList.add("hidden");
    resultEl.classList.add("hidden");
    errorBox.classList.add("hidden");
    scanLine.classList.add("hidden");
    scanStatus.classList.add("hidden");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze leaf";
  }

  function showPreview(file) {
    currentFile = file;
    objectUrl = URL.createObjectURL(file);
    previewImg.src = objectUrl;
    dropzone.classList.add("hidden");
    previewFrame.classList.remove("hidden");
    analyzeBtn.classList.remove("hidden");
    resetBtn.classList.remove("hidden");
    resultEl.classList.add("hidden");
    errorBox.classList.add("hidden");
  }

  function setLoading(isLoading) {
    if (isLoading) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyzing…";
      scanLine.classList.remove("hidden");
      scanStatus.classList.remove("hidden");
      scanStatus.innerHTML = 'Reading leaf pattern<span class="cursor"></span>';
      slowHintTimer = setTimeout(() => {
        scanStatus.innerHTML =
          'Still working — free-tier servers can take up to a minute to wake up<span class="cursor"></span>';
      }, 6000);
    } else {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze leaf";
      scanLine.classList.add("hidden");
      scanStatus.classList.add("hidden");
      clearTimeout(slowHintTimer);
    }
  }

  function showError(message) {
    errorMsg.textContent = message;
    errorBox.classList.remove("hidden");
    resultEl.classList.add("hidden");
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // Matches your backend's actual response shape:
  //   { "predictions": [<6 raw logits>], "class": "<top class name>" }
  // Also falls back to a couple of other common shapes so this keeps
  // working if the backend response ever changes.
  function normalizeResponse(data) {
    let entries = null;
    let topOverrideName = null;

    const rawArr = Array.isArray(data?.predictions)
      ? data.predictions
      : Array.isArray(data?.scores)
        ? data.scores
        : Array.isArray(data)
          ? data
          : null;

    if (rawArr) {
      // Your model's forward pass has no softmax applied, so these are
      // raw logits — convert to probabilities before showing percentages.
      const probs = softmax(rawArr.map(Number));
      entries = probs.map((p, i) => [
        CONFIG.CLASS_NAMES[i] || `Class ${i + 1}`,
        p,
      ]);
      if (typeof data?.class === "string") {
        topOverrideName = data.class;
      }
    } else {
      const obj =
        (data?.predictions &&
          typeof data.predictions === "object" &&
          data.predictions) ||
        (data?.scores && typeof data.scores === "object" && data.scores) ||
        (typeof data === "object" && data ? data : null);

      if (obj) {
        entries = Object.entries(obj)
          .filter(([, v]) => typeof v === "number")
          .map(([k, v]) => [k, Number(v)]);
      }
    }

    if (!entries || entries.length === 0) {
      throw new Error("Unrecognized response shape from backend.");
    }

    entries.sort((a, b) => b[1] - a[1]);

    // Trust the backend's own predicted class name for the top result,
    // in case CONFIG.CLASS_NAMES order doesn't perfectly match yet.
    if (topOverrideName && entries[0][0] !== topOverrideName) {
      entries[0] = [topOverrideName, entries[0][1]];
    }

    return entries; // [[name, score], ...] sorted descending
  }

  function renderResult(entries) {
    const [topName, topScore] = entries[0];
    const isHealthy = /healthy/i.test(topName);

    resultEl.innerHTML = "";

    const top = document.createElement("div");
    top.className = "result-top";
    top.innerHTML = `
      <div>
        <p class="eyebrow ${isHealthy ? "" : "is-alert"}">Diagnosis</p>
        <h2 class="result-title">${escapeHtml(topName)}</h2>
        <p class="result-confidence"><b>${(topScore * 100).toFixed(1)}%</b> confidence</p>
      </div>
      <span class="badge ${isHealthy ? "" : "is-alert"}">${isHealthy ? "Healthy" : "Disease detected"}</span>
    `;
    resultEl.appendChild(top);

    const label = document.createElement("p");
    label.className = "readings-label";
    label.textContent = "All readings";
    resultEl.appendChild(label);

    entries.forEach(([name, score], i) => {
      const row = document.createElement("div");
      row.className = `reading-row ${i === 0 ? "is-top" : ""} ${i === 0 && !isHealthy ? "is-alert" : ""}`;
      row.innerHTML = `
        <span class="reading-name">${escapeHtml(name)}</span>
        <span class="reading-value">${(score * 100).toFixed(1)}%</span>
        <div class="reading-bar-track"><div class="reading-bar-fill"></div></div>
      `;
      resultEl.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".reading-bar-fill").style.width =
          `${Math.max(score * 100, 2)}%`;
      });
    });

    resultEl.classList.remove("hidden");
    errorBox.classList.add("hidden");
    resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  async function analyze() {
    if (!currentFile) return;
    setLoading(true);
    errorBox.classList.add("hidden");

    try {
      const formData = new FormData();
      formData.append(CONFIG.FIELD_NAME, currentFile);

      const res = await fetch(CONFIG.API_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}.`);
      }

      const data = await res.json();
      const entries = normalizeResponse(data);
      renderResult(entries);
    } catch (err) {
      console.error(err);
      const isNetwork = err instanceof TypeError;
      showError(
        isNetwork
          ? "Couldn't reach the backend. Check that the Render service is running and CONFIG.API_URL in app.js is correct."
          : `Something went wrong: ${err.message}`,
      );
    } finally {
      setLoading(false);
    }
  }

  // ---------- events ----------

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) showPreview(file);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-drag");
    }),
  );

  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-drag");
    }),
  );

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) {
      fileInput.files = e.dataTransfer.files;
      showPreview(file);
    }
  });

  analyzeBtn.addEventListener("click", analyze);
  resetBtn.addEventListener("click", resetToIdle);

  resetToIdle();
});
