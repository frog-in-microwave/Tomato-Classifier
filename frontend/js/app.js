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
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "leaf_yellow_curl_virus",
    "septoria_leaf_spot",
  ],

  // Path to your manifest of sample image URLs, grouped by category
  SAMPLES_URL: "data/image_urls.json",
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
  const sampleTrigger = document.getElementById("sample-trigger");
  const sampleGallery = document.getElementById("sample-gallery");
  const sampleGrid = document.getElementById("sample-grid");
  const galleryBack = document.getElementById("gallery-back");
  const galleryRefetch = document.getElementById("gallery-refetch");
  const actualTag = document.getElementById("actual-tag");

  let currentFile = null;
  let objectUrl = null;
  let slowHintTimer = null;
  let samplesManifest = null;
  let isSampleMode = false;
  let currentTrueLabel = null;

  // ---------- helpers ----------

  function resetToIdle() {
    currentFile = null;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    fileInput.value = "";
    isSampleMode = false;
    currentTrueLabel = null;
    previewFrame.classList.add("hidden");
    dropzone.classList.remove("hidden");
    sampleTrigger.classList.remove("hidden");
    sampleGallery.classList.add("hidden");
    actualTag.classList.add("hidden");
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
    sampleTrigger.classList.add("hidden");
    sampleGallery.classList.add("hidden");
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

  // ---------- sample gallery ----------

  async function loadSamplesManifest() {
    if (samplesManifest) return samplesManifest;
    const res = await fetch(CONFIG.SAMPLES_URL);
    if (!res.ok) throw new Error(`Couldn't load sample list (${res.status}).`);
    samplesManifest = await res.json(); // { "Category": ["url", ...], ... }
    return samplesManifest;
  }

  function pickOnePerCategory(manifest) {
    return Object.entries(manifest)
      .filter(([, urls]) => Array.isArray(urls) && urls.length > 0)
      .map(([label, urls]) => ({
        label,
        url: urls[Math.floor(Math.random() * urls.length)],
      }));
  }

  function renderGallery(samples) {
    sampleGrid.innerHTML = "";
    samples.forEach((sample) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-thumb";
      btn.innerHTML = `
        <img src="${sample.url}" alt="${escapeHtml(sample.label)} sample leaf" loading="lazy"  crossorigin="anonymous"  />
        <span class="sample-label">${escapeHtml(sample.label)}</span>
      `;
      btn.addEventListener("click", () => selectSample(sample, btn));
      sampleGrid.appendChild(btn);
    });
  }

  // Fetches an ImageKit URL and turns it into a File so it can flow through
  // the same FormData/analyze path as a manual upload.
  async function urlToFile(url) {
    const res = await fetch(url);
    if (!res.ok)
      throw new Error(`Couldn't load that sample image (${res.status}).`);
    const blob = await res.blob();
    const filename = url.split("/").pop().split("?")[0] || "sample.jpg";
    return new File([blob], filename, { type: blob.type || "image/jpeg" });
  }

  async function selectSample(sample, btnEl) {
    btnEl.classList.add("is-loading");
    try {
      const file = await urlToFile(sample.url);
      isSampleMode = true;
      currentTrueLabel = sample.label;
      showPreview(file);
      actualTag.textContent = `Actual: ${sample.label} — test sample`;
      actualTag.classList.remove("hidden");
    } catch (err) {
      console.error(err);
      showError(
        "Couldn't load that sample image. This is usually a CORS setting on the ImageKit " +
          "delivery domain — check that cross-origin GET is allowed, or try another sample.",
      );
    } finally {
      btnEl.classList.remove("is-loading");
    }
  }

  // Matches your backend's actual response shape:
  //   { "predictions": [<6 probabilities, softmaxed server-side>], "class": "<top class name>" }
  // Also falls back to a couple of other common shapes so this keeps
  // working if the backend response ever changes.
  function normalizeResponse(data) {
    let entries = null;
    let topOverrideName = null;

    const rawArr = Array.isArray(data?.probabilities)
      ? data.probabilities
      : Array.isArray(data?.scores)
        ? data.scores
        : Array.isArray(data)
          ? data
          : null;

    if (rawArr) {
      // Backend applies softmax before returning, so these already sum to 1.
      entries = rawArr.map((score, i) => [
        CONFIG.CLASS_NAMES[i] || `Class ${i + 1}`,
        Number(score),
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

    let matchBadgeHtml = "";
    if (isSampleMode && currentTrueLabel) {
      const normalize = (s) =>
        s
          .toLowerCase()
          .replace(/[_\-]+/g, " ")
          .trim();
      const isMatch = normalize(topName) === normalize(currentTrueLabel);
      matchBadgeHtml = `
        <span class="badge match-badge ${isMatch ? "" : "is-alert"}">
          ${isMatch ? "✓ Model agrees" : "✗ Model disagrees"}
        </span>`;
    }

    resultEl.innerHTML = "";

    const top = document.createElement("div");
    top.className = "result-top";
    top.innerHTML = `
      <div>
        <p class="eyebrow ${isHealthy ? "" : "is-alert"}">Diagnosis</p>
        <h2 class="result-title">${escapeHtml(topName)}</h2>
        <p class="result-confidence"><b>${(topScore * 100).toFixed(1)}%</b> confidence</p>
        ${matchBadgeHtml}
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

  function resizeImage(file, maxSize = 1200) {
    return new Promise((resolve) => {
      const image = new Image();

      image.onload = () => {
        let width = image.width;
        let height = image.height;

        // Keep the aspect ratio
        if (width > height) {
          if (width > maxSize) {
            height = height * (maxSize / width);
            width = maxSize;
          }
        } else {
          if (height > maxSize) {
            width = width * (maxSize / height);
            height = maxSize;
          }
        }

        const canvas = document.createElement("canvas");

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");

        ctx.drawImage(image, 0, 0, width, height);

        canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.8);
      };

      image.src = URL.createObjectURL(file);
    });
  }







  async function analyze() {
    if (!currentFile) return;
    setLoading(true);
    errorBox.classList.add("hidden");

    try {
      const formData = new FormData();
      formData.append(CONFIG.FIELD_NAME, currentFile);
      currentFile = await resizeImage(currentFile, 1200);

      const res = await fetch(CONFIG.API_URL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Server responded with ${res.status}.`);
      }

      const data = await res.json();
      console.log("Backend response:", data);
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
    if (file) {
      isSampleMode = false;
      currentTrueLabel = null;
      showPreview(file);
    }
  });

  sampleTrigger.addEventListener("click", async () => {
    sampleTrigger.disabled = true;
    try {
      const manifest = await loadSamplesManifest();
      const samples = pickOnePerCategory(manifest);
      renderGallery(samples);
      dropzone.classList.add("hidden");
      sampleTrigger.classList.add("hidden");
      sampleGallery.classList.remove("hidden");
    } catch (err) {
      console.error(err);
      showError(
        "Couldn't load the test samples. Check CONFIG.SAMPLES_URL in app.js.",
      );
    } finally {
      sampleTrigger.disabled = false;
    }
  });

  galleryBack.addEventListener("click", () => {
    sampleGallery.classList.add("hidden");
    dropzone.classList.remove("hidden");
    sampleTrigger.classList.remove("hidden");
  });

  galleryRefetch.addEventListener("click", async () => {
    galleryRefetch.disabled = true;
    try {
      const manifest = await loadSamplesManifest();
      const samples = pickOnePerCategory(manifest);
      renderGallery(samples);
    } catch (err) {
      console.error(err);
      showError(
        "Couldn't refetch the test samples. Check CONFIG.SAMPLES_URL in app.js.",
      );
    } finally {
      galleryRefetch.disabled = false;
    }
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
      isSampleMode = false;
      currentTrueLabel = null;
      showPreview(file);
    }
  });

  analyzeBtn.addEventListener("click", analyze);
  resetBtn.addEventListener("click", resetToIdle);

  resetToIdle();
});
