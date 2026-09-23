/**
 * 表情识别推理测试平台 - 前端逻辑
 * FER (Face Expression Recognition) Testing Platform
 */

// ============================================
// 常量
// ============================================
const EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
const EMOTION_EMOJI = {
    angry: '😠', disgust: '🤢', fear: '😨', happy: '😊',
    sad: '😢', surprise: '😲', neutral: '😐'
};
const EMOTION_COLORS = {
    angry: '#ef476f', disgust: '#a06cd5', fear: '#7b68ee', happy: '#ffd166',
    sad: '#4361ee', surprise: '#06d6a0', neutral: '#8d99ae'
};

// ============================================
// 全局状态
// ============================================
let currentImage = null;
let currentResult = null;

// ============================================
// DOM 引用
// ============================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    modelSelect: $('#model-select'),
    modelStatus: $('#model-status'),
    modelInfo: $('#model-info'),
    detectFace: $('#detect-face'),
    detectorSelect: $('#detector-select'),
    detectorStatus: $('#detector-status'),
    apiStatusDot: $('#api-status-dot'),
    apiStatusText: $('#api-status-text'),
    apiDetail: $('#api-detail'),
    uploadArea: $('#upload-area'),
    fileInput: $('#file-input'),
    resultsArea: $('#results-area'),
    previewCanvas: $('#preview-canvas'),
    faceCount: $('#face-count'),
    resultsContainer: $('#results-container'),
    toastContainer: $('#toast-container'),
    btnRecognize: $('#btn-recognize'),
    btnChangeImage: $('#btn-change-image'),
    btnPreviewClose: $('#btn-preview-close'),
    inferenceMeta: $('#inference-meta'),
};

// ============================================
// API 调用
// ============================================
async function apiCall(path, options = {}) {
    const merged = {
        headers: { ...options.headers || {} },
        ...options
    };
    // GET 请求不设 Content-Type，POST JSON 才设
    if (merged.method && merged.method !== 'GET' && !merged.headers['Content-Type']) {
        merged.headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(path, merged);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

async function loadModels() {
    const data = await apiCall('/api/models');
    if (!data.success) throw new Error(data.message);
    return data.data;
}

async function switchModel(modelName) {
    const data = await apiCall('/api/model/switch', {
        method: 'POST',
        body: JSON.stringify({ model_name: modelName })
    });
    if (!data.success) throw new Error(data.message);
    return data;
}

async function switchDetector(detectorType) {
    const data = await apiCall('/api/detector/switch', {
        method: 'POST',
        body: JSON.stringify({ detector_type: detectorType })
    });
    if (!data.success) throw new Error(data.message);
    return data;
}

async function getDetectorInfo() {
    const data = await apiCall('/api/detector');
    return data;
}

async function recognizeImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    const res = await fetch('/api/recognize', {
        method: 'POST',
        body: formData
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

async function checkHealth() {
    const data = await apiCall('/api/health');
    return data;
}

// ============================================
// Toast 通知
// ============================================
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================
// 模型管理
// ============================================
async function refreshModels() {
    try {
        const data = await loadModels();
        const { models, current_model } = data;
        
        dom.modelSelect.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = `${m.name.toUpperCase()} (${m.pth_size_mb}MB)`;
            if (!m.pth_exists) opt.disabled = true;
            dom.modelSelect.appendChild(opt);
        });
        
        if (current_model) {
            dom.modelSelect.value = current_model;
        }
        
        updateModelInfo(current_model);
        showToast(`已加载 ${models.length} 个模型`, 'success');
    } catch (e) {
        showToast('加载模型列表失败: ' + e.message, 'error');
    }
}

function updateModelInfo(modelName) {
    if (modelName) {
        dom.modelStatus.textContent = '已加载';
        dom.modelStatus.className = 'tag tag-green';
    } else {
        dom.modelStatus.textContent = '未加载';
        dom.modelStatus.className = 'tag tag-warning';
    }
}

dom.modelSelect.addEventListener('change', async () => {
    const modelName = dom.modelSelect.value;
    if (!modelName) return;
    
    dom.modelStatus.textContent = '切换中...';
    dom.modelStatus.className = 'tag';
    
    try {
        await switchModel(modelName);
        showToast(`已切换到 ${modelName.toUpperCase()}，请点击"识别"按钮重新推理`, 'success');
        updateModelInfo(modelName);
    } catch (e) {
        showToast('切换模型失败: ' + e.message, 'error');
        updateModelInfo(dom.modelSelect.value);
    }
});

$('#btn-reload-models').addEventListener('click', refreshModels);

// 检测器切换
dom.detectorSelect.addEventListener('change', async () => {
    const detType = dom.detectorSelect.value;
    dom.detectorStatus.textContent = '切换中...';
    dom.detectorStatus.className = 'tag';
    try {
        const data = await switchDetector(detType);
        dom.detectorStatus.textContent = data.detector_type === 'insightface' ? 'InsightFace' : 'OpenCV';
        dom.detectorStatus.className = 'tag tag-green';
        showToast(`检测器已切换为 ${dom.detectorStatus.textContent}`, 'success');
    } catch (e) {
        showToast('切换检测器失败: ' + e.message, 'error');
        dom.detectorStatus.textContent = 'InsightFace';
        dom.detectorStatus.className = 'tag tag-green';
        dom.detectorSelect.value = 'insightface';
    }
});

// ============================================
// 健康检查
// ============================================
async function checkServiceHealth() {
    try {
        const data = await checkHealth();
        if (data.model_loaded) {
            dom.apiStatusDot.className = 'status-dot online';
            dom.apiStatusText.textContent = '服务正常';
            dom.apiDetail.textContent = `当前模型: ${data.current_model || '-'}`;
        } else {
            dom.apiStatusDot.className = 'status-dot online';
            dom.apiStatusText.textContent = '等待模型加载';
            dom.apiDetail.textContent = '模型未加载';
        }
    } catch (e) {
        dom.apiStatusDot.className = 'status-dot offline';
        dom.apiStatusText.textContent = '服务离线';
        dom.apiDetail.textContent = e.message;
    }
}

// ============================================
// 图片上传
// ============================================
function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        showToast('请选择图片文件', 'warning');
        return;
    }
    
    currentImage = file;
    
    // 显示结果区
    dom.resultsArea.style.display = 'block';
    dom.uploadArea.style.display = 'none';
    
    // 预览
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            drawPreview(img);
            // 重置结果，等待手动点击识别
            dom.resultsContainer.innerHTML = '<div class="empty-state">点击上方"识别"按钮开始推理</div>';
            dom.inferenceMeta.style.display = 'none';
            dom.faceCount.textContent = '';
            dom.btnRecognize.disabled = false;
            dom.btnRecognize.textContent = '🔍 识别';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

async function runInference(file) {
    dom.resultsContainer.innerHTML = `
        <div class="empty-state">
            <div class="loading-spinner" style="border-color: var(--primary); border-top-color: transparent; margin: 0 auto 12px;"></div>
            <p>推理中...</p>
        </div>`;
    
    try {
        const data = await recognizeImage(file);
        currentResult = data;
        
        if (data.success) {
            renderResults(data.data);
            // 重绘预览（加上人脸框）
            if (data.data.faces && data.data.faces.length > 0) {
                redrawWithFaces(data.data.faces);
            }
        } else {
            dom.resultsContainer.innerHTML = `
                <div class="error-card">
                    <strong>推理失败:</strong> ${data.message}
                </div>`;
            showToast(data.message, 'error');
        }
    } catch (e) {
        dom.resultsContainer.innerHTML = `
            <div class="error-card">
                <strong>请求失败:</strong> ${e.message}
            </div>`;
        showToast('推理请求失败: ' + e.message, 'error');
    } finally {
        dom.btnRecognize.disabled = false;
        dom.btnRecognize.textContent = '🔍 识别';
    }
}

// ============================================
// 预览
// ============================================
let previewImage = null;

function drawPreview(img) {
    previewImage = img;
    const canvas = dom.previewCanvas;
    const maxW = 800, maxH = 480;
    let w = img.width, h = img.height;
    
    if (w > maxW) { h = h * maxW / w; w = maxW; }
    if (h > maxH) { w = w * maxH / h; h = maxH; }
    
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
}

function redrawWithFaces(faces) {
    if (!previewImage) return;
    const canvas = dom.previewCanvas;
    const maxW = 800, maxH = 480;
    let w = previewImage.width, h = previewImage.height;
    
    if (w > maxW) { h = h * maxW / w; w = maxW; }
    if (h > maxH) { w = w * maxH / h; h = maxH; }
    
    const scaleX = w / previewImage.width;
    const scaleY = h / previewImage.height;
    
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(previewImage, 0, 0, w, h);
    
    // 绘制人脸框
    faces.forEach((face, i) => {
        const [bx, by, bw, bh] = face.bbox;
        const x = bx * scaleX, y = by * scaleY;
        const fw = bw * scaleX, fh = bh * scaleY;
        
        // 人脸框
        ctx.strokeStyle = EMOTION_COLORS[face.expression] || '#4361ee';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, fw, fh);
        
        // 标签背景
        const label = `${EMOTION_EMOJI[face.expression]} ${face.expression} ${(face.confidence_emotion*100).toFixed(0)}%`;
        ctx.font = 'bold 14px sans-serif';
        const metrics = ctx.measureText(label);
        const labelW = metrics.width + 12;
        const labelH = 24;
        const labelY = y - labelH - 4 > 0 ? y - labelH - 4 : y + fh + 4;
        
        ctx.fillStyle = EMOTION_COLORS[face.expression] || '#4361ee';
        ctx.beginPath();
        ctx.roundRect(x, labelY, labelW, labelH, 6);
        ctx.fill();
        
        ctx.fillStyle = (face.expression === 'happy' || face.expression === 'surprise') ? '#333' : '#fff';
        ctx.fillText(label, x + 6, labelY + 17);
        
        // 关键点
        if (face.keypoints) {
            Object.values(face.keypoints).forEach(pt => {
                ctx.fillStyle = '#fff';
                ctx.beginPath();
                ctx.arc(pt[0] * scaleX, pt[1] * scaleY, 3, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = EMOTION_COLORS[face.expression] || '#4361ee';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            });
        }
    });
    
    dom.faceCount.textContent = `${faces.length} 张人脸`;
}

// Canvas roundRect polyfill
if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        this.beginPath();
        this.moveTo(x + r, y);
        this.lineTo(x + w - r, y);
        this.arcTo(x + w, y, x + w, y + r, r);
        this.lineTo(x + w, y + h - r);
        this.arcTo(x + w, y + h, x + w - r, y + h, r);
        this.lineTo(x + r, y + h);
        this.arcTo(x, y + h, x, y + h - r, r);
        this.lineTo(x, y + r);
        this.arcTo(x, y, x + r, y, r);
        this.closePath();
    };
}

// ============================================
// 结果渲染
// ============================================
function renderResults(data) {
    const { faces, faces_detected, model, process_time } = data;
    
    // 显示模型和推理时长
    dom.inferenceMeta.style.display = 'flex';
    dom.inferenceMeta.innerHTML = `
        <span>模型: <strong>${model?.toUpperCase() || '-'}</strong></span>
        <span class="meta-sep">|</span>
        <span>推理时长: <strong>${(process_time * 1000).toFixed(1)} ms</strong></span>
    `;
    
    if (faces_detected === 0) {
        // 整体推理结果
        dom.resultsContainer.innerHTML = `
            <div class="face-result-card">
                <div class="face-result-header">
                    <div class="face-result-emotion">
                        <span class="emotion-emoji">${EMOTION_EMOJI[data.expression]}</span>
                        ${data.expression}
                    </div>
                    <div class="face-result-conf">
                        置信度: ${(data.confidence*100).toFixed(1)}%
                    </div>
                </div>
                <div class="face-result-bbox">
                    未检测到人脸 · 使用全局图像推理
                </div>
                ${renderProbBars(data.probabilities, data.expression)}
            </div>`;
        return;
    }
    
    // 多人脸结果
    dom.resultsContainer.innerHTML = faces.map((face, i) => `
        <div class="face-result-card">
            <div class="face-result-header">
                <div class="face-result-emotion">
                    <span class="emotion-emoji">${EMOTION_EMOJI[face.expression]}</span>
                    ${face.expression}
                </div>
                <div class="face-result-conf">
                    置信度: ${(face.confidence_emotion*100).toFixed(1)}%
                </div>
            </div>
            <div class="face-result-bbox">
                人脸 #${i+1} · 
                bbox: [${face.bbox.map(v=>Math.round(v)).join(', ')}] · 
                检测置信度: ${(face.confidence*100).toFixed(0)}%
            </div>
            ${renderProbBars(face.probabilities, face.expression)}
        </div>
    `).join('');
}

function renderProbBars(probs, topExpression) {
    const entries = EMOTION_LABELS.map(label => ({
        label,
        prob: probs[label] || 0
    }));
    entries.sort((a, b) => b.prob - a.prob);
    
    return `
        <div class="prob-bars">
            ${entries.map(e => `
                <div class="prob-row">
                    <span class="prob-label">${EMOTION_EMOJI[e.label]} ${e.label}</span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill bar-${e.label} ${e.label === topExpression ? 'highlight' : ''}"
                             style="width:${Math.max(e.prob*100, 1)}%">
                        </div>
                    </div>
                    <span class="prob-value">${(e.prob*100).toFixed(1)}%</span>
                </div>
            `).join('')}
        </div>`;
}

// ============================================
// 拖拽上传
// ============================================
dom.uploadArea.addEventListener('click', () => dom.fileInput.click());

dom.uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.uploadArea.classList.add('drag-over');
});

dom.uploadArea.addEventListener('dragleave', () => {
    dom.uploadArea.classList.remove('drag-over');
});

dom.uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleImageUpload(file);
});

dom.fileInput.addEventListener('change', () => {
    const file = dom.fileInput.files[0];
    if (file) handleImageUpload(file);
});

// 识别按钮 - 人工控制推理
dom.btnRecognize.addEventListener('click', () => {
    if (!currentImage) {
        showToast('请先上传图片', 'warning');
        return;
    }
    dom.btnRecognize.disabled = true;
    dom.btnRecognize.textContent = '推理中...';
    runInference(currentImage);
});

// 粘贴上传
document.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    
    for (const item of items) {
        if (item.type.startsWith('image/')) {
            e.preventDefault();
            const file = item.getAsFile();
            handleImageUpload(file);
            break;
        }
    }
});

// 清空
$('#btn-clear').addEventListener('click', resetToUpload);

// 更换图片
dom.btnChangeImage.addEventListener('click', resetToUpload);

// 预览区右上角 X 关闭
dom.btnPreviewClose.addEventListener('click', resetToUpload);

function resetToUpload() {
    currentImage = null;
    currentResult = null;
    previewImage = null;
    dom.previewCanvas.width = 0;
    dom.previewCanvas.height = 0;
    dom.resultsArea.style.display = 'none';
    dom.uploadArea.style.display = '';
    dom.resultsContainer.innerHTML = '<div class="empty-state">点击上方"识别"按钮开始推理</div>';
    dom.inferenceMeta.style.display = 'none';
    dom.faceCount.textContent = '';
    dom.fileInput.value = '';
    dom.btnRecognize.disabled = false;
    dom.btnRecognize.textContent = '🔍 识别';
}

// ============================================
// 初始化
// ============================================
async function init() {
    await refreshModels();
    await checkServiceHealth();
    // 同步检测器状态
    try {
        const detInfo = await getDetectorInfo();
        if (detInfo.success) {
            dom.detectorSelect.value = detInfo.data.detector_type;
            dom.detectorStatus.textContent = detInfo.data.detector_type === 'insightface' ? 'InsightFace' : 'OpenCV';
        }
    } catch (e) {}
    setInterval(checkServiceHealth, 30000);
}

init();
