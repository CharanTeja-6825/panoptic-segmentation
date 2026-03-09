/**
 * Real-Time Intelligent Scene Understanding Platform – Frontend JavaScript
 *
 * Handles:
 *  - Tab switching
 *  - Video file drag-and-drop + upload
 *  - Processing progress polling
 *  - Live camera MJPEG stream start/stop
 *  - FPS display
 *  - Device/model status display
 *  - Analytics dashboard (live stats, events, benchmark)
 *  - Heatmap + Depth toggles
 *  - Multi-camera management
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
    const depthBadge = document.getElementById("depth-badge");

    if (data.device === "cuda") {
      deviceBadge.textContent = "GPU: " + (data.gpu || "CUDA");
      deviceBadge.className = "badge badge-gpu";
    } else {
      deviceBadge.textContent = "CPU";
      deviceBadge.className = "badge badge-cpu";
    }

    modelBadge.textContent = "Model: " + (data.model_size || "—");
    modelBadge.className = "badge badge-model";

    if (data.depth_estimation === "enabled") {
      depthBadge.textContent = "Depth: on";
      depthBadge.className = "badge badge-gpu";
    } else {
      depthBadge.textContent = "Depth: off";
      depthBadge.className = "badge badge-neutral";
    }
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
const heatmapBtn   = document.getElementById("heatmap-btn");
const depthBtn     = document.getElementById("depth-btn");
const camFps       = document.getElementById("cam-fps");
const camStream    = document.getElementById("cam-stream");
const camOverlay   = document.getElementById("cam-overlay");
const cameraStatus = document.getElementById("camera-status");
const alphaSlider  = document.getElementById("alpha-slider");
const alphaValue   = document.getElementById("alpha-value");
const camStats     = document.getElementById("cam-stats");

let fpsPollInterval = null;
let analyticsPollInterval = null;

/* ---- Slider ---- */
alphaSlider.addEventListener("input", () => {
  alphaValue.textContent = alphaSlider.value + "%";
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
    setVisible(camStats, true);
    setStatus(cameraStatus, "Camera running");

    // Poll FPS + analytics
    fpsPollInterval = setInterval(updateCamFps, 1500);
    analyticsPollInterval = setInterval(updateCamAnalytics, 2000);
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
  clearInterval(analyticsPollInterval);

  try {
    await fetch("/api/camera-stream/stop", { method: "POST" });
  } catch (_) {}

  camStream.src = "";
  camOverlay.classList.remove("hidden");
  camFps.textContent = "—";
  setVisible(camStats, false);
  startCamBtn.disabled = false;
  setStatus(cameraStatus, "Camera stopped");
});

/* ---- Heatmap toggle ---- */
heatmapBtn.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/camera-stream/toggle-heatmap", { method: "POST" });
    const data = await res.json();
    heatmapBtn.textContent = data.heatmap ? "🔥 Heatmap ON" : "🔥 Heatmap";
    heatmapBtn.classList.toggle("btn-active", data.heatmap);
  } catch (_) {}
});

/* ---- Depth toggle ---- */
depthBtn.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/toggle-depth", { method: "POST" });
    const data = await res.json();
    depthBtn.textContent = data.depth_enabled ? "🔭 Depth ON" : "🔭 Depth";
    depthBtn.classList.toggle("btn-active", data.depth_enabled);
    fetchHealth();
  } catch (_) {}
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

async function updateCamAnalytics() {
  try {
    const res = await fetch("/api/analytics/live");
    if (!res.ok) return;
    const data = await res.json();
    const objCount = document.getElementById("cam-obj-count");
    const personCount = document.getElementById("cam-person-count");
    const vehicleCount = document.getElementById("cam-vehicle-count");
    if (objCount) objCount.textContent = data.total_objects || 0;
    if (personCount) personCount.textContent = (data.current_counts || {}).person || 0;
    if (vehicleCount) {
      const counts = data.current_counts || {};
      vehicleCount.textContent = (counts.car || 0) + (counts.truck || 0) + (counts.bus || 0);
    }
  } catch (_) {}
}

/* ================================================================
   Analytics Tab
   ================================================================ */

let analyticsTabInterval = null;

// Start analytics polling when tab is active
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.dataset.tab === "analytics") {
      startAnalyticsPolling();
    } else {
      stopAnalyticsPolling();
    }
    if (btn.dataset.tab === "multicam") {
      refreshMulticamList();
    }
  });
});

function startAnalyticsPolling() {
  stopAnalyticsPolling();
  updateAnalyticsTab();
  analyticsTabInterval = setInterval(updateAnalyticsTab, 2000);
}

function stopAnalyticsPolling() {
  clearInterval(analyticsTabInterval);
  analyticsTabInterval = null;
}

async function updateAnalyticsTab() {
  try {
    const res = await fetch("/api/analytics/live");
    if (!res.ok) return;
    const data = await res.json();

    const totalEl = document.getElementById("analytics-total");
    const fpsEl = document.getElementById("analytics-fps");
    const framesEl = document.getElementById("analytics-frames");
    if (totalEl) totalEl.textContent = data.total_objects || 0;
    if (fpsEl) fpsEl.textContent = data.fps || 0;
    if (framesEl) framesEl.textContent = data.total_frames || 0;

    // Counts by class
    const countsDiv = document.getElementById("analytics-counts");
    const counts = data.current_counts || {};
    if (Object.keys(counts).length > 0) {
      countsDiv.innerHTML = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([cls, cnt]) => `<div class="count-row"><span>${cls}</span><span class="count-badge">${cnt}</span></div>`)
        .join("");
    }

    // Recent events
    const eventsDiv = document.getElementById("analytics-events");
    const events = data.recent_events || [];
    if (events.length > 0) {
      eventsDiv.innerHTML = events
        .map((e) => `<div class="event-row ${e.event_type}"><span class="event-type">${e.event_type === "entry" ? "➡️" : "⬅️"} ${e.event_type}</span><span>${e.class_name}</span><span class="event-id">ID:${e.track_id}</span></div>`)
        .join("");
    }
  } catch (_) {}
}

// Benchmark refresh
const refreshBenchBtn = document.getElementById("refresh-bench-btn");
if (refreshBenchBtn) {
  refreshBenchBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/benchmark");
      if (!res.ok) return;
      const data = await res.json();
      const fpsEl = document.getElementById("bench-fps");
      const gpuEl = document.getElementById("bench-gpu-mem");
      const cpuEl = document.getElementById("bench-cpu");
      if (fpsEl) fpsEl.textContent = data.avg_fps || "—";
      if (gpuEl) gpuEl.textContent = data.gpu_memory_mb ? `${data.gpu_memory_mb} MB` : "N/A";
      if (cpuEl) cpuEl.textContent = data.cpu_load_1m || "—";
    } catch (_) {}
  });
}

/* ================================================================
   Multi-Camera Tab
   ================================================================ */

const mcStartBtn = document.getElementById("mc-start-btn");

if (mcStartBtn) {
  mcStartBtn.addEventListener("click", async () => {
    const camIndex = parseInt(document.getElementById("mc-cam-index").value) || 0;
    const label = document.getElementById("mc-cam-label").value || "";

    try {
      const res = await fetch("/api/camera/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: camIndex, label: label }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        alert("Failed: " + (body.detail || res.statusText));
        return;
      }
      refreshMulticamList();
    } catch (err) {
      alert("Error: " + err.message);
    }
  });
}

async function refreshMulticamList() {
  try {
    const res = await fetch("/api/camera/list");
    if (!res.ok) return;
    const data = await res.json();
    const listDiv = document.getElementById("mc-streams-list");
    if (!listDiv) return;

    const streams = data.streams || [];
    if (streams.length === 0) {
      listDiv.innerHTML = '<p class="hint">No active camera streams</p>';
      return;
    }

    listDiv.innerHTML = streams.map((s) => `
      <div class="stream-card">
        <div class="stream-info">
          <strong>${s.label || "Camera " + s.camera_index}</strong>
          <span class="hint">ID: ${s.stream_id} · Camera: ${s.camera_index}</span>
        </div>
        <div class="stream-stats">
          <span class="fps-badge">${s.fps} fps</span>
          <span class="count-badge">${s.object_count} objects</span>
        </div>
        <button class="btn btn-danger btn-sm" onclick="stopMulticamStream('${s.stream_id}')">✕</button>
      </div>
    `).join("");
  } catch (_) {}
}

async function stopMulticamStream(streamId) {
  try {
    await fetch("/api/camera/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_id: streamId }),
    });
    refreshMulticamList();
  } catch (_) {}
}
