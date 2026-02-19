/**
 * Panoptic Segmentation – Frontend JavaScript
 *
 * Handles:
 *  - Tab switching
 *  - Video file drag-and-drop + upload
 *  - Processing progress polling
 *  - Live camera MJPEG stream start/stop
 *  - FPS display
 *  - Device/model status display
 */

"use strict";

/* ================================================================
   Utility helpers
   ================================================================ */

/** Format bytes as a human-readable string. */
function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/** Show/hide an element. */
function setVisible(el, visible) {
  if (visible) el.classList.remove("hidden");
  else el.classList.add("hidden");
}

/** Set status message with optional style class. */
function setStatus(el, msg, cls = "") {
  el.textContent = msg;
  el.className = "status-msg" + (cls ? " " + cls : "");
}

/* ================================================================
   Tab switching
   ================================================================ */

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + target).classList.add("active");
  });
});

/* ================================================================
   Health / status bar
   ================================================================ */

async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();

    const deviceBadge = document.getElementById("device-badge");
    const modelBadge = document.getElementById("model-badge");

    if (data.device === "cuda") {
      deviceBadge.textContent = "GPU: " + (data.gpu || "CUDA");
      deviceBadge.className = "badge badge-gpu";
    } else {
      deviceBadge.textContent = "CPU";
      deviceBadge.className = "badge badge-cpu";
    }

    modelBadge.textContent = "Model: " + (data.model_size || "—");
    modelBadge.className = "badge badge-model";
  } catch (_) {
    // Server not yet available; silently ignore
  }
}

fetchHealth();

/* ================================================================
   Video Upload + Processing
   ================================================================ */

const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const fileInfo       = document.getElementById("file-info");
const fileName       = document.getElementById("file-name");
const fileSize       = document.getElementById("file-size");
const uploadBtn      = document.getElementById("upload-btn");
const videoStatus    = document.getElementById("video-status");
const outputCard     = document.getElementById("output-card");
const outputVideo    = document.getElementById("output-video");
const downloadLink   = document.getElementById("download-link");

const uploadWrap     = document.getElementById("upload-progress-wrap");
const uploadBar      = document.getElementById("upload-progress-bar");
const uploadPct      = document.getElementById("upload-progress-pct");

const processWrap    = document.getElementById("process-progress-wrap");
const processBar     = document.getElementById("process-progress-bar");
const processPct     = document.getElementById("process-progress-pct");
const processFps     = document.getElementById("process-fps");

let selectedFile = null;
let pollInterval = null;

/* ---- Drop zone events ---- */

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });

dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileSelected(fileInput.files[0]);
});

function handleFileSelected(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const allowed = ["mp4", "avi", "mov", "mkv"];
  if (!allowed.includes(ext)) {
    setStatus(videoStatus, `Unsupported file type ".${ext}". Allowed: .mp4, .avi, .mov, .mkv`, "error");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  setVisible(fileInfo, true);
  uploadBtn.disabled = false;
  setStatus(videoStatus, "");
}

/* ---- Upload + process ---- */

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // Reset UI
  uploadBtn.disabled = true;
  setVisible(outputCard, false);
  setStatus(videoStatus, "Uploading…");
  setVisible(uploadWrap, true);
  setVisible(processWrap, false);
  uploadBar.style.width = "0%";
  uploadPct.textContent = "0%";

  // --- Upload via XHR (supports progress) ---
  let jobId;
  try {
    jobId = await uploadFileWithProgress(selectedFile);
  } catch (err) {
    setStatus(videoStatus, "Upload failed: " + err.message, "error");
    uploadBtn.disabled = false;
    return;
  }

  setVisible(uploadWrap, false);
  setStatus(videoStatus, "Starting segmentation…");
  setVisible(processWrap, true);

  // --- Trigger processing ---
  try {
    const res = await fetch(`/api/process-video/${jobId}`, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || res.statusText);
    }
  } catch (err) {
    setStatus(videoStatus, "Failed to start processing: " + err.message, "error");
    uploadBtn.disabled = false;
    return;
  }

  // --- Poll for progress ---
  setStatus(videoStatus, "Processing…");
  startPolling(jobId);
});

function uploadFileWithProgress(file) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload-video");

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        uploadBar.style.width = pct + "%";
        uploadPct.textContent = pct + "%";
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data.job_id);
        } catch {
          reject(new Error("Invalid server response"));
        }
      } else {
        try {
          const err = JSON.parse(xhr.responseText);
          reject(new Error(err.detail || xhr.statusText));
        } catch {
          reject(new Error(xhr.statusText));
        }
      }
    });

    xhr.addEventListener("error", () => reject(new Error("Network error")));
    xhr.send(formData);
  });
}

function startPolling(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/job-status/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();

      const done = data.progress || 0;
      const total = data.total_frames || 0;
      const pct = total > 0 ? Math.round((done / total) * 100) : 0;

      processBar.style.width = pct + "%";
      processPct.textContent = `${done} / ${total}`;
      processFps.textContent = data.fps ? `${data.fps} fps` : "";

      if (data.status === "done") {
        clearInterval(pollInterval);
        processBar.style.width = "100%";
        setStatus(videoStatus, "✅ Segmentation complete!", "success");

        const downloadUrl = data.download_url;
        outputVideo.src = downloadUrl;
        downloadLink.href = downloadUrl;
        setVisible(outputCard, true);
        uploadBtn.disabled = false;
      } else if (data.status === "failed") {
        clearInterval(pollInterval);
        setStatus(videoStatus, "❌ Processing failed: " + (data.error || "unknown error"), "error");
        uploadBtn.disabled = false;
      }
    } catch (_) {
      // network hiccup – keep polling
    }
  }, 1000);
}

/* ================================================================
   Live Camera Streaming
   ================================================================ */

const startCamBtn  = document.getElementById("start-cam-btn");
const stopCamBtn   = document.getElementById("stop-cam-btn");
const camFps       = document.getElementById("cam-fps");
const camStream    = document.getElementById("cam-stream");
const camOverlay   = document.getElementById("cam-overlay");
const cameraStatus = document.getElementById("camera-status");
const alphaSlider  = document.getElementById("alpha-slider");
const alphaValue   = document.getElementById("alpha-value");

let fpsPollInterval = null;

/* ---- Slider ---- */
alphaSlider.addEventListener("input", () => {
  alphaValue.textContent = alphaSlider.value + "%";
  // TODO: send alpha preference to backend config endpoint (future enhancement)
});

/* ---- Start camera ---- */
startCamBtn.addEventListener("click", async () => {
  startCamBtn.disabled = true;
  setStatus(cameraStatus, "Starting camera…");

  try {
    const res = await fetch("/api/camera-stream/start", { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || res.statusText);
    }

    // Point the img tag at the MJPEG stream
    camStream.src = "/api/camera-stream?" + Date.now();
    camOverlay.classList.add("hidden");
    stopCamBtn.disabled = false;
    setStatus(cameraStatus, "Camera running");

    // Poll FPS
    fpsPollInterval = setInterval(updateCamFps, 1500);
    updateCamFps();
  } catch (err) {
    setStatus(cameraStatus, "❌ " + err.message, "error");
    startCamBtn.disabled = false;
  }
});

/* ---- Stop camera ---- */
stopCamBtn.addEventListener("click", async () => {
  stopCamBtn.disabled = true;
  clearInterval(fpsPollInterval);

  try {
    await fetch("/api/camera-stream/stop", { method: "POST" });
  } catch (_) {}

  camStream.src = "";
  camOverlay.classList.remove("hidden");
  camFps.textContent = "—";
  startCamBtn.disabled = false;
  setStatus(cameraStatus, "Camera stopped");
});

async function updateCamFps() {
  try {
    const res = await fetch("/api/camera-stream/status");
    if (!res.ok) return;
    const data = await res.json();
    if (data.running) {
      camFps.textContent = data.fps || "0";
    } else {
      camFps.textContent = "—";
    }
  } catch (_) {}
}
