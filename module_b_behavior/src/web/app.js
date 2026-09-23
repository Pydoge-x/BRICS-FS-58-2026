const API = "/api/v1";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let selectedFile = null;
let selectedMediaType = null;
let previewObjectUrl = null;
let pollTimer = null;
let currentJobId = null;
let jobStartedAt = null;
let toastTimer = null;
let demoSamples = [];

const BEHAVIOR_LABELS_ZH = {
  sit_listen: "听讲",
  raise_hand: "举手",
  write: "书写",
  bow_head: "低头",
  stand: "站立",
  sport: "运动",
  play: "玩耍",
  chat: "交谈",
  walk: "行走",
  unknown: "未知",
};

const SCENE_LABELS = {
  classroom: "课堂",
  extracurricular: "课外（MMAction2）",
};

const SCENE_HINTS = {
  classroom: "课堂场景支持图片与视频，基于 YOLO-Pose 姿态规则推断行为。",
  extracurricular:
    "课外 MMAction2 TSN（Kinetics-400 预训练），仅视频，建议最大帧数 60。",
};

function getSelectedScene() {
  const checked = document.querySelector('input[name="scene"]:checked');
  return checked ? checked.value : "classroom";
}

function isExtracurricularScene(scene) {
  return scene === "extracurricular" || scene === "extracurricular_kinetics";
}

function updateSceneUi() {
  const scene = getSelectedScene();
  const hint = $("#sceneHint");
  const isExtra = isExtracurricularScene(scene);
  hint.textContent = SCENE_HINTS[scene] || SCENE_HINTS.classroom;
  if (isExtra && !$("#maxFrames").value) {
    $("#maxFrames").placeholder = "建议 60";
  }
  if (isExtra && selectedMediaType === "image") {
    $("#submitBtn").disabled = true;
    hint.textContent += " 当前为图片，请切换课堂或上传视频。";
  } else if (selectedFile) {
    $("#submitBtn").disabled = false;
  }
  $$(".video-only").forEach((el) => {
    const isKf = el.querySelector("#exportKeyframes");
    if (isKf) {
      el.classList.toggle("hidden", selectedMediaType !== "video" || isExtra);
    } else {
      el.classList.toggle("hidden", selectedMediaType !== "video");
    }
  });
}

function isImage(file) {
  return file.type.startsWith("image/") || /\.(jpe?g|png|bmp|webp)$/i.test(file.name);
}

function isVideo(file) {
  return file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);
}

function revokePreviewUrl() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

function setMediaSrc(el, src, isVid) {
  if (!el) return;
  el.classList.add("hidden");
  el.removeAttribute("src");
  if (!src) return;
  if (isVid) {
    el.src = src;
    el.classList.remove("hidden");
    el.load();
  } else {
    el.src = src;
    el.classList.remove("hidden");
  }
}

function hideAll(...els) {
  els.forEach((el) => {
    if (el) {
      el.classList.add("hidden");
      el.removeAttribute("src");
    }
  });
}

async function checkHealth() {
  const el = $("#apiStatus");
  try {
    const res = await fetch(`${API}/health`);
    if (!res.ok) throw new Error("unhealthy");
    const data = await res.json();
    el.className = "header-status ok";
    const scenes = (data.scenes || []).join(" / ") || "classroom / extracurricular";
    el.innerHTML = `<span class="dot"></span><span>服务正常 · ${scenes}</span>`;
  } catch {
    el.className = "header-status err";
    el.innerHTML = `<span class="dot"></span><span>服务离线</span>`;
  }
}

function setupUpload() {
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  $("#pickFile").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });

  $("#clearFile").addEventListener("click", clearFile);
  $("#submitBtn").addEventListener("click", submitAnalysis);
  $$('input[name="scene"]').forEach((el) => {
    el.addEventListener("change", updateSceneUi);
  });
  updateSceneUi();

  $$(".clickable-media").forEach((el) => {
    el.addEventListener("click", () => openLightbox(el.src));
  });
  $("#lightboxClose").addEventListener("click", closeLightbox);
  $("#lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox") closeLightbox();
  });
}

function setFile(file) {
  if (!isImage(file) && !isVideo(file)) {
    alert("请选择图片或视频文件");
    return;
  }
  selectedFile = file;
  selectedMediaType = isImage(file) ? "image" : "video";

  $("#fileName").textContent = `${file.name} (${formatSize(file.size)})`;
  $("#mediaBadge").textContent = selectedMediaType === "image" ? "图片" : "视频";
  $("#mediaBadge").className = `media-badge ${selectedMediaType}`;

  revokePreviewUrl();
  previewObjectUrl = URL.createObjectURL(file);
  hideAll($("#uploadPreviewImg"), $("#uploadPreviewVideo"));
  if (selectedMediaType === "image") {
    setMediaSrc($("#uploadPreviewImg"), previewObjectUrl, false);
  } else {
    setMediaSrc($("#uploadPreviewVideo"), previewObjectUrl, true);
  }

  $$(".video-only, .video-only-tab").forEach((el) => {
    el.classList.toggle("hidden", selectedMediaType !== "video");
  });

  $("#filePreview").classList.remove("hidden");
  updateSceneUi();
}

function clearFile() {
  selectedFile = null;
  selectedMediaType = null;
  revokePreviewUrl();
  $("#fileInput").value = "";
  hideAll($("#uploadPreviewImg"), $("#uploadPreviewVideo"));
  $("#filePreview").classList.add("hidden");
  $("#submitBtn").disabled = true;
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function cacheBust(url) {
  return `${url}?t=${Date.now()}`;
}

function showToast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  if (isError) {
    el.style.borderColor = "var(--danger)";
    el.style.color = "var(--danger)";
  } else {
    el.style.borderColor = "var(--success)";
    el.style.color = "var(--success)";
  }
  el.classList.remove("hidden");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4000);
}

function formatElapsed(ms) {
  const sec = Math.floor(ms / 1000);
  if (sec < 60) return `${sec} 秒`;
  return `${Math.floor(sec / 60)} 分 ${sec % 60} 秒`;
}

function updateElapsed() {
  const el = $("#progressElapsed");
  if (!jobStartedAt) {
    el.classList.add("hidden");
    return;
  }
  el.textContent = `已用时 ${formatElapsed(Date.now() - jobStartedAt)}`;
  el.classList.remove("hidden");
}

function sceneTag(mode) {
  if (mode === "classroom") return "课堂";
  return "MMAction2";
}

async function loadDemoSamples() {
  const grid = $("#demoGrid");
  try {
    const res = await fetch(`${API}/demo-samples`);
    if (!res.ok) throw new Error("无法加载样本");
    const samples = await res.json();
    demoSamples = samples;
    if (!samples.length) {
      grid.innerHTML = "<p class='hint'>暂无内置样本</p>";
      return;
    }
    grid.innerHTML = samples.map((s) => `
      <article class="demo-item ${s.available ? "" : "disabled"}" data-id="${s.id}">
        ${s.preview_url
          ? `<img class="demo-thumb" src="${s.preview_url}" alt="${s.name}" loading="lazy" />`
          : `<div class="demo-thumb"></div>`}
        <div class="demo-body">
          <h3>${s.name}</h3>
          <p>${s.description}</p>
          <div class="demo-tags">
            <span class="demo-tag">${s.media_type === "image" ? "图片" : "视频"}</span>
            <span class="demo-tag">${sceneTag(s.analysis_mode)}</span>
            ${s.max_frames ? `<span class="demo-tag">${s.max_frames} 帧</span>` : ""}
          </div>
          <button type="button" class="demo-run" ${s.available ? "" : "disabled"}>
            ${s.available ? "一键分析" : "文件缺失"}
          </button>
        </div>
      </article>
    `).join("");

    grid.querySelectorAll(".demo-item:not(.disabled)").forEach((card) => {
      const id = card.dataset.id;
      card.querySelector(".demo-run").addEventListener("click", (e) => {
        e.stopPropagation();
        runDemoSample(id);
      });
      card.addEventListener("click", () => selectDemoSample(id, samples));
    });
  } catch {
    grid.innerHTML = "<p class='hint'>内置样本加载失败，请确认 API 已启动</p>";
  }
}

function selectDemoSample(id, samples) {
  const sample = samples.find((s) => s.id === id);
  if (!sample || !sample.available) return;
  const radio = document.querySelector(`input[name="scene"][value="${sample.analysis_mode}"]`);
  if (radio) radio.checked = true;
  if (sample.max_frames) $("#maxFrames").value = sample.max_frames;
  updateSceneUi();
  showToast(`已切换场景：${sample.name}`);
}

async function runDemoSample(sampleId) {
  const sample = demoSamples.find((s) => s.id === sampleId);
  if (sample) {
    const radio = document.querySelector(`input[name="scene"][value="${sample.analysis_mode}"]`);
    if (radio) radio.checked = true;
    if (sample.max_frames) $("#maxFrames").value = sample.max_frames;
    updateSceneUi();
  }

  $$(".demo-run").forEach((btn) => { btn.disabled = true; });
  $("#submitBtn").disabled = true;
  $("#progressCard").classList.remove("hidden");
  $("#resultsCard").classList.add("hidden");
  jobStartedAt = Date.now();
  setProgress(5, "提交内置样本…");
  document.querySelector(".progress-bar")?.classList.add("running");

  try {
    const form = new FormData();
    form.append("sample_id", sampleId);
    form.append("visualize", $("#visualize").checked);
    form.append("export_keyframes", $("#exportKeyframes").checked);
    const maxFrames = $("#maxFrames").value;
    if (maxFrames) form.append("max_frames", maxFrames);

    const res = await fetch(`${API}/behavior/analyze-demo`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    const job = await res.json();
    currentJobId = job.job_id;
    setProgress(15, `样本任务已创建 · ${job.filename}`);
    pollJob(job.job_id);
    loadJobHistory();
  } catch (err) {
    setProgress(0, `提交失败: ${err.message}`);
    finishJobUi(false);
    showToast(err.message, true);
  }
}

function finishJobUi(success) {
  document.querySelector(".progress-bar")?.classList.remove("running");
  $$(".demo-run").forEach((btn) => { btn.disabled = false; });
  $("#submitBtn").disabled = !selectedFile;
  if (success && jobStartedAt) {
    updateElapsed();
  }
}

async function submitAnalysis() {
  if (!selectedFile) return;

  const form = new FormData();
  form.append("media", selectedFile);
  if (selectedMediaType === "video") {
    const maxFrames = $("#maxFrames").value;
    if (maxFrames) form.append("max_frames", maxFrames);
    form.append("export_keyframes", $("#exportKeyframes").checked);
  }
  form.append("visualize", $("#visualize").checked);
  form.append("analysis_mode", getSelectedScene());

  $("#submitBtn").disabled = true;
  $("#progressCard").classList.remove("hidden");
  $("#resultsCard").classList.add("hidden");
  jobStartedAt = Date.now();
  setProgress(5, "提交任务…");
  document.querySelector(".progress-bar")?.classList.add("running");

  try {
    const res = await fetch(`${API}/behavior/analyze`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    const job = await res.json();
    currentJobId = job.job_id;
    setProgress(15, "任务已创建，排队中…");
    pollJob(job.job_id);
    loadJobHistory();
  } catch (err) {
    setProgress(0, `提交失败: ${err.message}`);
    finishJobUi(false);
    showToast(err.message, true);
  }
}

function setProgress(pct, text) {
  $("#progressFill").style.width = `${pct}%`;
  $("#progressText").textContent = text;
  updateElapsed();
}

function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  updateElapsed();
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API}/jobs/${jobId}`);
      const job = await res.json();
      updateJobMeta(job);
      updateElapsed();

      if (job.status === "pending") setProgress(20, job.progress_message || "排队中…");
      else if (job.status === "running") setProgress(55, job.progress_message || "分析中，请稍候…");
      else if (job.status === "completed") {
        clearInterval(pollTimer);
        setProgress(100, "分析完成 ✓");
        finishJobUi(true);
        showResults(job);
        loadJobHistory();
        showToast(`分析完成 · ${job.filename}`);
        $("#resultsCard").scrollIntoView({ behavior: "smooth", block: "start" });
      } else if (job.status === "failed") {
        clearInterval(pollTimer);
        setProgress(0, `失败: ${job.error || "未知错误"}`);
        finishJobUi(false);
        showToast(job.error || "分析失败", true);
      }
    } catch (err) {
      clearInterval(pollTimer);
      setProgress(0, `轮询失败: ${err.message}`);
      finishJobUi(false);
      showToast(err.message, true);
    }
  }, 1500);
}

function sceneDisplayLabel(job) {
  if (job?.scene === "extracurricular" || isExtracurricularScene(job?.analysis_mode)) {
    return SCENE_LABELS.extracurricular;
  }
  const am = job?.analysis_mode;
  if (am && SCENE_LABELS[am]) return SCENE_LABELS[am];
  return SCENE_LABELS.classroom;
}

function updateJobMeta(job) {
  const typeLabel = job.media_type === "image" ? "图片" : "视频";
  const sceneLabel = sceneDisplayLabel(job);
  $("#jobMeta").innerHTML = `
    <div><dt>任务 ID</dt><dd>${job.job_id}</dd></div>
    <div><dt>文件</dt><dd>${job.filename}</dd></div>
    <div><dt>场景</dt><dd>${sceneLabel}</dd></div>
    <div><dt>类型</dt><dd>${typeLabel}</dd></div>
    <div><dt>状态</dt><dd><span class="status-badge ${job.status}">${job.status}</span></dd></div>
  `;
}

function showAnnotatedMedia(job, urls) {
  const isImageJob = job.media_type === "image" || job.result_summary?.media_type === "image";
  const annotatedUrl = urls.annotated ? cacheBust(urls.annotated) : null;
  const originalUrl = urls.original ? cacheBust(urls.original) : null;

  hideAll($("#resultImage"), $("#resultVideo"));
  const hint = $("#resultMediaHint");
  const download = $("#downloadAnnotated");

  if (annotatedUrl) {
    download.href = annotatedUrl;
    download.classList.remove("hidden");
    if (isImageJob) {
      setMediaSrc($("#resultImage"), annotatedUrl, false);
      hint.textContent = "点击图片可放大查看";
      hint.classList.remove("hidden");
    } else {
      const vid = $("#resultVideo");
      setMediaSrc(vid, annotatedUrl, true);
      vid.onerror = () => {
        hint.textContent = "浏览器无法直接播放该编码，请点击下载后在本地播放";
        hint.classList.remove("hidden");
      };
      vid.onloadeddata = () => hint.classList.add("hidden");
    }
  } else {
    hint.textContent = "未生成标注结果";
    hint.classList.remove("hidden");
    download.classList.add("hidden");
  }

  hideAll(
    $("#compareOriginalImg"), $("#compareOriginalVideo"),
    $("#compareAnnotatedImg"), $("#compareAnnotatedVideo"),
  );
  if (originalUrl) {
    if (isImageJob) {
      setMediaSrc($("#compareOriginalImg"), originalUrl, false);
      setMediaSrc($("#compareAnnotatedImg"), annotatedUrl, false);
    } else {
      setMediaSrc($("#compareOriginalVideo"), originalUrl, true);
      setMediaSrc($("#compareAnnotatedVideo"), annotatedUrl, true);
    }
  }

  $$(".video-only-tab").forEach((el) => {
    el.classList.toggle("hidden", isImageJob);
  });
}

async function showResults(job) {
  $("#resultsCard").classList.remove("hidden");
  const summary = job.result_summary || {};
  const isImage = job.media_type === "image" || summary.media_type === "image";

  const isExtra = job.scene === "extracurricular" || isExtracurricularScene(job.analysis_mode)
    || summary.scene === "extracurricular";
  const sceneBeh = summary.scene_behavior || {};
  const labelMode = summary.label_mode || sceneBeh.label_mode || job.label_mode;
  const stats = isImage
    ? [
        ["检测人数", summary.person_count],
        ["分辨率", summary.resolution ? `${summary.resolution[0]}×${summary.resolution[1]}` : "—"],
        ["推理耗时", summary.performance?.avg_pipeline_ms ? `${summary.performance.avg_pipeline_ms} ms` : "—"],
        ["行为种类", Object.keys(summary.behavior_segment_counts || {}).length],
      ]
    : isExtra
      ? [
          ["Kinetics 行为", sceneBeh.behavior || sceneBeh.raw_label || "—"],
          ["置信度", sceneBeh.score != null ? `${(sceneBeh.score * 100).toFixed(0)}%` : "—"],
          ["模型", "MMAction2 预训练"],
          ["帧数", summary.total_frames ?? "—"],
        ]
      : [
          ["总帧数", summary.total_frames],
          ["Track 数", summary.track_count],
          ["FPS", summary.fps?.toFixed?.(1) || summary.fps],
          ["流水线 FPS", summary.performance?.avg_pipeline_fps],
          ["关键帧", summary.keyframes_count],
        ];

  $("#statsGrid").innerHTML = stats.map(([label, val]) => `
    <div class="stat-box">
      <div class="value">${val ?? "—"}</div>
      <div class="label">${label}</div>
    </div>
  `).join("");

  renderBehaviorChart(summary.behavior_segment_counts || {}, job);

  const res = await fetch(`${API}/jobs/${job.job_id}/result`);
  const data = await res.json();
  renderTracksTable(data.tracks || [], isImage, job);

  const urls = job.urls || {};
  showAnnotatedMedia(job, urls);

  if (!isImage && urls.keyframes_manifest) {
    const kfRes = await fetch(urls.keyframes_manifest);
    const kf = await kfRes.json();
    renderKeyframes(kf);
  } else {
    $("#keyframesGrid").innerHTML = "";
  }

  $$("#tabBar .tab").forEach((t) => t.classList.remove("active"));
  $$(".tab-panel").forEach((p) => p.classList.remove("active"));
  $("#tabBar .tab[data-tab='annotated']").classList.add("active");
  $("#panel-annotated").classList.add("active");
}

function behaviorLabel(behavior, job) {
  const isExtra = job?.scene === "extracurricular" || isExtracurricularScene(job?.analysis_mode)
    || job?.result_summary?.scene === "extracurricular";
  if (isExtra) return behavior;
  return BEHAVIOR_LABELS_ZH[behavior] || behavior;
}

function renderBehaviorChart(counts, job) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map((e) => e[1]), 1);
  $("#behaviorChart").innerHTML = entries.length
    ? entries.map(([b, c]) => `
        <div class="behavior-row">
          <span class="name">${behaviorLabel(b, job)}</span>
          <div class="bar-bg"><div class="bar-fill" style="width:${(c / max) * 100}%"></div></div>
          <span class="count">${c}</span>
        </div>
      `).join("")
    : "<p style='color:var(--muted)'>暂无行为片段</p>";
}

function renderTracksTable(tracks, isImage, job) {
  const rows = [];
  for (const t of tracks) {
    for (const seg of t.segments || []) {
      rows.push(`
        <tr>
          <td>#${t.track_id}</td>
          <td><span class="behavior-tag">${behaviorLabel(seg.behavior, job)}</span></td>
          <td>${(seg.behavior_confidence * 100).toFixed(0)}%</td>
          <td>${isImage ? "—" : (seg.start_timestamp || "—")}</td>
          <td>${isImage ? "—" : (seg.end_timestamp || "—")}</td>
          <td>${isImage ? "—" : (seg.duration_sec ?? "—")}</td>
        </tr>
      `);
    }
  }
  $("#tracksTable tbody").innerHTML = rows.length
    ? rows.join("")
    : "<tr><td colspan='6' style='color:var(--muted)'>无轨迹数据</td></tr>";
}

function renderKeyframes(items) {
  $("#keyframesGrid").innerHTML = items.map((kf) => `
    <div class="kf-card">
      <img src="${cacheBust(kf.annotated_url)}" alt="frame ${kf.frame}" loading="lazy"
           class="clickable-media" title="点击放大" />
      <div class="kf-meta">
        帧 ${kf.frame} · ${kf.timestamp_sec}s · ${kf.num_persons} 人
        ${kf.behaviors?.includes("raise_hand") ? " · 举手" : ""}
      </div>
    </div>
  `).join("");
  $("#keyframesGrid").querySelectorAll(".clickable-media").forEach((el) => {
    el.addEventListener("click", () => openLightbox(el.src));
  });
}

function openLightbox(src) {
  if (!src) return;
  $("#lightboxImg").src = src;
  $("#lightbox").classList.remove("hidden");
}

function closeLightbox() {
  $("#lightbox").classList.add("hidden");
  $("#lightboxImg").removeAttribute("src");
}

function setupTabs() {
  $$("#tabBar .tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$("#tabBar .tab").forEach((t) => t.classList.remove("active"));
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#panel-${tab.dataset.tab}`).classList.add("active");
    });
  });
}

async function loadJobHistory() {
  try {
    const res = await fetch(`${API}/jobs?limit=10`);
    const jobs = await res.json();
    const list = $("#jobList");
    if (!jobs.length) {
      list.innerHTML = '<li class="empty">暂无任务</li>';
      return;
    }
    list.innerHTML = jobs.map((j) => `
      <li data-job="${j.job_id}">
        <span>${j.media_type === "image" ? "🖼" : "🎬"} ${j.filename}</span>
        <span class="status-badge ${j.status}">${j.status}</span>
      </li>
    `).join("");

    list.querySelectorAll("li[data-job]").forEach((li) => {
      li.addEventListener("click", async () => {
        const jobId = li.dataset.job;
        const res = await fetch(`${API}/jobs/${jobId}`);
        const job = await res.json();
        currentJobId = jobId;
        $("#progressCard").classList.remove("hidden");
        updateJobMeta(job);
        if (job.status === "completed") {
          setProgress(100, "已完成");
          showResults(job);
        } else if (job.status === "failed") {
          setProgress(0, job.error || "失败");
          $("#resultsCard").classList.add("hidden");
        } else {
          setProgress(30, job.progress_message || job.status);
          pollJob(jobId);
        }
      });
    });
  } catch { /* ignore */ }
}

checkHealth();
setupUpload();
setupTabs();
loadJobHistory();
loadDemoSamples();
setInterval(checkHealth, 30000);
