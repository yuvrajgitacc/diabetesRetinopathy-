/* ============================================
   OptiGemma v2.0 — Clinical Suite JavaScript
   SPA navigation, Patient management, Batch queue
   ============================================ */

// === State ===
let currentPage = 'dashboard';
let selectedPatientId = '';
let selectedPatientName = '';
let currentScanFile = null;
let batchQueue = [];
let batchFile = null;
let currentDetailPatientId = null;
let currentReport = null;

// === Theme Toggle ===
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    document.getElementById('theme-icon').textContent = next === 'dark' ? '🌙' : '☀️';
    localStorage.setItem('theme', next);
}

// Load saved theme
(function() {
    const saved = localStorage.getItem('theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
        const icon = document.getElementById('theme-icon');
        if (icon) icon.textContent = saved === 'dark' ? '🌙' : '☀️';
    }
})();

// === SPA Navigation ===
function navigateTo(page, data) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    // Show target page
    const target = document.getElementById('page-' + page);
    if (target) target.classList.add('active');
    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navItem = document.querySelector(`[data-page="${page}"]`);
    if (navItem) navItem.classList.add('active');
    currentPage = page;

    // Load page data
    if (page === 'dashboard') loadDashboard();
    else if (page === 'patients') loadPatients();
    else if (page === 'patient-detail' && data) loadPatientDetail(data);
    else if (page === 'scan') resetScan();
}

// === DASHBOARD ===
async function loadDashboard() {
    try {
        const res = await fetch('/api/dashboard');
        const data = await res.json();
        if (!data.success) return;

        document.getElementById('stat-patients').textContent = data.total_patients;
        document.getElementById('stat-scans').textContent = data.total_scans;

        const dist = data.stage_distribution || {};
        const healthy = (dist[0] || {}).count || 0;
        const atRisk = data.total_scans - healthy;
        document.getElementById('stat-healthy').textContent = healthy;
        document.getElementById('stat-atrisk').textContent = atRisk;

        // Recent scans table
        const container = document.getElementById('recent-scans-list');
        if (data.recent_scans && data.recent_scans.length > 0) {
            let html = '<table class="data-table"><thead><tr><th>Patient</th><th>Stage</th><th>Confidence</th><th>Date</th></tr></thead><tbody>';
            data.recent_scans.forEach(s => {
                html += `<tr onclick="navigateTo('patient-detail','${s.patient_id}')">
                    <td><strong>${s.patient_name || 'Quick Scan'}</strong></td>
                    <td><span class="stage-badge stage-${s.stage}">${s.stage_name}</span></td>
                    <td>${s.confidence}%</td>
                    <td>${formatDate(s.created_at)}</td>
                </tr>`;
            });
            html += '</tbody></table>';
            container.innerHTML = html;
        } else {
            container.innerHTML = '<p class="empty-state">No scans yet. Click "New Scan" to get started.</p>';
        }
    } catch (e) {
        console.error('Dashboard load error:', e);
    }
}

// === PATIENTS ===
async function loadPatients(search) {
    try {
        const q = search ? `?search=${encodeURIComponent(search)}` : '';
        const res = await fetch('/api/patients' + q);
        const data = await res.json();
        if (!data.success) return;

        const container = document.getElementById('patients-table-container');
        if (data.patients.length === 0) {
            container.innerHTML = '<p class="empty-state">No patients found. Create one from New Scan.</p>';
            return;
        }

        let html = '<table class="data-table"><thead><tr><th>ID</th><th>Name</th><th>Age</th><th>Scans</th><th>Last Stage</th><th>Last Scan</th></tr></thead><tbody>';
        data.patients.forEach(p => {
            const lastStage = p.latest_scan ? `<span class="stage-badge stage-${p.latest_scan.stage}">${p.latest_scan.stage_name}</span>` : '—';
            const lastDate = p.latest_scan ? formatDate(p.latest_scan.created_at) : '—';
            html += `<tr onclick="navigateTo('patient-detail','${p.id}')">
                <td style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--text-muted)">${p.id}</td>
                <td><strong>${p.name}</strong></td>
                <td>${p.age || '—'}</td>
                <td>${p.scan_count}</td>
                <td>${lastStage}</td>
                <td>${lastDate}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch (e) {
        console.error('Patients load error:', e);
    }
}

// === PATIENT DETAIL ===
async function loadPatientDetail(patientId) {
    currentDetailPatientId = patientId;
    try {
        const res = await fetch(`/api/patients/${patientId}`);
        const data = await res.json();
        if (!data.success) { alert('Patient not found.'); navigateTo('patients'); return; }

        const p = data.patient;
        // Avatar initials
        const initials = p.name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
        document.getElementById('pd-avatar').textContent = initials;
        document.getElementById('pd-name').textContent = p.name;
        document.getElementById('pd-subtitle').textContent = `${p.id} • ${p.gender || ''} • ${p.age ? p.age + ' years' : ''}`;

        // Hero stats
        const statsEl = document.getElementById('pd-hero-stats');
        statsEl.innerHTML = [
            heroStat(p.age || '—', 'Age'),
            heroStat(p.diabetes_duration ? p.diabetes_duration + 'y' : '—', 'Diabetes'),
            heroStat(p.sugar_level ? p.sugar_level : '—', 'Sugar mg/dL'),
            heroStat(p.hba1c ? p.hba1c + '%' : '—', 'HbA1c'),
        ].join('');

        // Risk bar + summary
        const scans = p.scans || [];
        const lastScan = scans.length > 0 ? scans[0] : null;
        document.getElementById('pd-scan-count').textContent = scans.length;
        document.getElementById('pd-last-scan-date').textContent = lastScan ? formatDate(lastScan.created_at) : '—';

        if (lastScan) {
            const stageBadge = `<span class="stage-badge stage-${lastScan.stage}">${lastScan.stage_name}</span>`;
            document.getElementById('pd-last-stage').innerHTML = stageBadge;
            const riskPct = Math.min(10 + lastScan.stage * 22, 95);
            document.getElementById('pd-risk-fill').style.width = riskPct + '%';
        } else {
            document.getElementById('pd-last-stage').textContent = '—';
            document.getElementById('pd-risk-fill').style.width = '5%';
        }

        // Scans table
        const scansContainer = document.getElementById('pd-scans-list');
        if (scans.length > 0) {
            let html = '<table class="data-table"><thead><tr><th>Date</th><th>Stage</th><th>Confidence</th><th>Model</th><th>Time</th></tr></thead><tbody>';
            scans.forEach(s => {
                html += `<tr onclick="viewScanResult('${s.id}')">
                    <td>${formatDate(s.created_at)}</td>
                    <td><span class="stage-badge stage-${s.stage}">${s.stage_name}</span></td>
                    <td>${s.confidence}%</td>
                    <td style="font-size:0.78rem;color:var(--text-muted)">${s.model_used || '—'}</td>
                    <td>${s.processing_time}s</td>
                </tr>`;
            });
            html += '</tbody></table>';
            scansContainer.innerHTML = html;

            if (scans.length >= 2) {
                const progCard = document.getElementById('pd-progression-card');
                progCard.style.display = 'block';
                const progDiv = document.getElementById('pd-progression');
                const colors = ['#34d399', '#fbbf24', '#fb923c', '#f87171', '#a855f7'];
                let progHtml = '';
                scans.slice().reverse().forEach(s => {
                    const h = 20 + s.stage * 22;
                    progHtml += `<div class="pt-item">
                        <div class="pt-bar" style="height:${h}px;background:${colors[s.stage]}"></div>
                        <div class="pt-stage" style="color:${colors[s.stage]}">Stage ${s.stage}</div>
                        <div class="pt-date">${formatDate(s.created_at)}</div>
                    </div>`;
                });
                progDiv.innerHTML = progHtml;
            } else {
                document.getElementById('pd-progression-card').style.display = 'none';
            }
        } else {
            scansContainer.innerHTML = '<p class="empty-state">No scans yet for this patient.</p>';
            document.getElementById('pd-progression-card').style.display = 'none';
        }
    } catch (e) { console.error('Patient detail error:', e); }
}

function heroStat(val, label) {
    return `<div class="pd-hero-stat"><span class="phs-val">${val}</span><span class="phs-lbl">${label}</span></div>`;
}

// Edit Patient — Modal-based
async function editPatient(patientId) {
    if (!patientId) { alert('No patient selected.'); return; }
    try {
        const res = await fetch(`/api/patients/${patientId}`);
        const data = await res.json();
        if (!data.success) { alert('Patient not found.'); return; }
        const p = data.patient;
        document.getElementById('edit-pid').value = patientId;
        document.getElementById('edit-name').value = p.name || '';
        document.getElementById('edit-age').value = p.age || '';
        document.getElementById('edit-gender').value = p.gender || '';
        document.getElementById('edit-diabetes').value = p.diabetes_duration || '';
        document.getElementById('edit-sugar').value = p.sugar_level || '';
        document.getElementById('edit-hba1c').value = p.hba1c || '';
        document.getElementById('edit-modal').style.display = 'flex';
    } catch (e) { alert('Error loading patient: ' + e.message); }
}

function closeEditModal() {
    document.getElementById('edit-modal').style.display = 'none';
}

async function saveEditPatient() {
    const pid = document.getElementById('edit-pid').value;
    const body = {
        name: document.getElementById('edit-name').value.trim(),
        age: parseInt(document.getElementById('edit-age').value) || null,
        gender: document.getElementById('edit-gender').value,
        diabetes_duration: parseInt(document.getElementById('edit-diabetes').value) || null,
        sugar_level: parseFloat(document.getElementById('edit-sugar').value) || null,
        hba1c: parseFloat(document.getElementById('edit-hba1c').value) || null
    };
    if (!body.name) { alert('Name is required.'); return; }
    try {
        const res = await fetch(`/api/patients/${pid}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.success) {
            closeEditModal();
            loadPatientDetail(pid);
        } else {
            alert('Failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) { alert('Error: ' + e.message); }
}

function scanForPatient() {
    selectedPatientId = currentDetailPatientId;
    navigateTo('scan');
    // Pre-select patient
    const banner = document.getElementById('selected-patient-banner');
    banner.style.display = 'flex';
    document.getElementById('selected-patient-info').textContent = `Patient: ${selectedPatientId}`;
    document.getElementById('scan-step-upload').style.opacity = '1';
    document.getElementById('scan-step-upload').style.pointerEvents = 'auto';
}

async function viewScanResult(scanId) {
    try {
        const res = await fetch(`/api/scans/${scanId}`);
        const data = await res.json();
        if (!data.success) return;
        // Navigate to scan page and show results
        navigateTo('scan');
        displayResults({
            success: true,
            detection: { stage: data.scan.stage, stage_name: data.scan.stage_name, confidence: data.scan.confidence, all_probabilities: data.scan.all_probabilities, severity: data.scan.severity, color: data.scan.color },
            heatmap_analysis: data.scan.heatmap_analysis,
            vessel_stats: data.scan.vessel_stats,
            report: data.scan.report,
            images: { original: data.scan.image_original, heatmap: data.scan.image_heatmap, vessels: data.scan.image_vessels },
            processing_time: data.scan.processing_time
        });
    } catch (e) {
        console.error('View scan error:', e);
    }
}

// === NEW SCAN ===
function switchScanTab(tab) {
    document.querySelectorAll('.scan-tab').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('scan-tab-' + tab).style.display = 'block';
    event.target.classList.add('active');
}

async function searchPatientsForScan(query) {
    if (!query || query.length < 1) {
        document.getElementById('scan-patient-results').innerHTML = '';
        return;
    }
    try {
        const res = await fetch(`/api/patients?search=${encodeURIComponent(query)}`);
        const data = await res.json();
        if (!data.success) return;

        const container = document.getElementById('scan-patient-results');
        if (data.patients.length === 0) {
            container.innerHTML = '<p class="info-text">No patients found. Try "New Patient" tab.</p>';
            return;
        }
        container.innerHTML = data.patients.map(p =>
            `<div class="patient-result-item" onclick="selectPatient('${p.id}','${p.name.replace(/'/g, "\\'")}')">
                <span class="pr-name">${p.name}</span>
                <span class="pr-id">${p.id} • ${p.age || '—'} yrs</span>
            </div>`
        ).join('');
    } catch (e) {
        console.error(e);
    }
}

function selectPatient(id, name) {
    selectedPatientId = id;
    selectedPatientName = name;
    document.getElementById('selected-patient-banner').style.display = 'flex';
    document.getElementById('selected-patient-info').textContent = `✓ ${name} (${id})`;
    document.getElementById('scan-step-upload').style.opacity = '1';
    document.getElementById('scan-step-upload').style.pointerEvents = 'auto';
}

async function createPatientAndSelect() {
    const name = document.getElementById('new-patient-name').value.trim();
    if (!name) { alert('Please enter patient name.'); return; }

    try {
        const body = {
            name: name,
            age: parseInt(document.getElementById('new-patient-age').value) || null,
            gender: document.getElementById('new-patient-gender').value,
            diabetes_duration: parseInt(document.getElementById('new-patient-diabetes').value) || null,
            sugar_level: parseFloat(document.getElementById('new-patient-sugar').value) || null,
            hba1c: parseFloat(document.getElementById('new-patient-hba1c').value) || null,
        };
        const res = await fetch('/api/patients', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json();
        if (!data.success) { alert(data.error || 'Failed to create patient.'); return; }
        selectPatient(data.patient.id, data.patient.name);
    } catch (e) {
        alert('Error creating patient: ' + e.message);
    }
}

function selectQuickScan() {
    selectedPatientId = '';
    selectedPatientName = '';
    document.getElementById('selected-patient-banner').style.display = 'flex';
    document.getElementById('selected-patient-info').textContent = '⚡ Quick Scan (no patient record)';
    document.getElementById('scan-step-upload').style.opacity = '1';
    document.getElementById('scan-step-upload').style.pointerEvents = 'auto';
}

function clearSelectedPatient() {
    selectedPatientId = '';
    selectedPatientName = '';
    document.getElementById('selected-patient-banner').style.display = 'none';
    document.getElementById('scan-step-upload').style.opacity = '0.5';
    document.getElementById('scan-step-upload').style.pointerEvents = 'none';
}

// File upload handling
document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
    setupUploadZone('scan-drop-zone', 'scan-file-input', handleScanFile);
    setupUploadZone('batch-drop-zone', 'batch-file-input', handleBatchFile);
});

function setupUploadZone(zoneId, inputId, handler) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handler(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => { if (input.files.length) handler(input.files[0]); });
}

function handleScanFile(file) {
    if (!file.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
    if (file.size > 16 * 1024 * 1024) { alert('File too large. Max 16MB.'); return; }
    currentScanFile = file;
    document.getElementById('scan-drop-zone').style.display = 'none';
    document.getElementById('scan-preview').style.display = 'block';
    document.getElementById('scan-preview-name').textContent = file.name;
    document.getElementById('run-analysis-btn').style.display = 'flex';
    const reader = new FileReader();
    reader.onload = e => document.getElementById('scan-preview-img').src = e.target.result;
    reader.readAsDataURL(file);
}

function clearScanImage() {
    currentScanFile = null;
    document.getElementById('scan-drop-zone').style.display = 'block';
    document.getElementById('scan-preview').style.display = 'none';
    document.getElementById('run-analysis-btn').style.display = 'none';
    document.getElementById('scan-file-input').value = '';
}

function resetScan() {
    clearScanImage();
    document.getElementById('scan-results').style.display = 'none';
    document.getElementById('scan-loading').style.display = 'none';
    document.getElementById('scan-step-patient').style.display = 'block';
    document.getElementById('scan-step-upload').style.display = 'block';
    document.querySelectorAll('.lstep').forEach(s => { s.classList.remove('active', 'done'); });
}

// === Run Analysis ===
async function runAnalysis() {
    if (!currentScanFile) { alert('Please upload a fundus image first.'); return; }

    // Hide form, show loading
    document.getElementById('scan-step-patient').style.display = 'none';
    document.getElementById('scan-step-upload').style.display = 'none';
    document.getElementById('scan-loading').style.display = 'block';
    document.getElementById('scan-results').style.display = 'none';

    // Animate loading steps
    const steps = ['ls-preprocess', 'ls-detect', 'ls-segment', 'ls-heatmap', 'ls-report'];
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
        if (stepIdx > 0) document.getElementById(steps[stepIdx - 1]).classList.replace('active', 'done');
        if (stepIdx < steps.length) document.getElementById(steps[stepIdx]).classList.add('active');
        stepIdx++;
        if (stepIdx > steps.length) clearInterval(stepTimer);
    }, 2500);

    const formData = new FormData();
    formData.append('image', currentScanFile);
    if (selectedPatientId) formData.append('patient_id', selectedPatientId);

    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        clearInterval(stepTimer);
        steps.forEach(s => { document.getElementById(s).classList.remove('active'); document.getElementById(s).classList.add('done'); });

        if (!data.success && data.error) {
            alert('Analysis failed: ' + data.error);
            resetScan();
            return;
        }
        setTimeout(() => displayResults(data), 600);
    } catch (e) {
        clearInterval(stepTimer);
        alert('Connection error: ' + e.message);
        resetScan();
    }
}

function displayResults(data) {
    document.getElementById('scan-loading').style.display = 'none';
    document.getElementById('scan-results').style.display = 'block';

    const d = data.detection;
    currentReport = data.report;

    // Gauge
    document.getElementById('result-stage').textContent = d.stage;
    document.getElementById('result-name').textContent = d.stage_name;
    document.getElementById('result-confidence').textContent = `Confidence: ${d.confidence}%`;
    document.getElementById('result-time').textContent = `${data.processing_time}s`;
    document.getElementById('result-conf-pill').textContent = `${d.confidence}%`;
    const riskLabels = ['Low', 'Low', 'Medium', 'High', 'Critical'];
    document.getElementById('result-risk-level').textContent = riskLabels[d.stage] || '—';

    const circle = document.getElementById('result-gauge-circle');
    const pct = d.confidence / 100;
    circle.style.strokeDashoffset = 327 - (327 * pct);
    const colors = ['#34d399', '#fbbf24', '#fb923c', '#f87171', '#a855f7'];
    circle.style.stroke = colors[d.stage] || '#4f8ff7';

    // Images
    if (data.images) {
        document.getElementById('r-original').src = data.images.original;
        document.getElementById('r-vessels').src = data.images.vessels;
        document.getElementById('r-heatmap').src = data.images.heatmap;
    }

    // Vessel & Heatmap captions
    const vs = data.vessel_stats || {};
    document.getElementById('r-vessel-cap').textContent = `Density: ${vs.vessel_density_percent || '—'}%`;
    const ha = data.heatmap_analysis || {};
    document.getElementById('r-heatmap-cap').textContent = `Coverage: ${ha.activation_coverage || '—'}%`;

    // Probability bars
    const probContainer = document.getElementById('result-prob-bars');
    const stageNames = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'];
    let probHtml = '';
    for (let i = 0; i < 5; i++) {
        const val = d.all_probabilities ? (d.all_probabilities[i] || 0) : 0;
        probHtml += `<div class="prob-bar-row">
            <span class="prob-label">${stageNames[i]}</span>
            <div class="prob-track"><div class="prob-fill" style="width:${val}%;background:${colors[i]}"></div></div>
            <span class="prob-val">${val.toFixed(1)}%</span>
        </div>`;
    }
    probContainer.innerHTML = probHtml;

    // Report
    const r = data.report || {};
    document.getElementById('rpt-diagnosis').textContent = r.diagnosis || 'Analysis complete.';
    document.getElementById('rpt-heatmap').textContent = r.heatmap_analysis || '—';
    document.getElementById('rpt-vessels').textContent = r.vessel_analysis || '—';

    // Risk prediction
    const risk = r.risk_prediction || {};
    const r6 = risk['6_month'] || {};
    const r12 = risk['12_month'] || {};
    document.getElementById('rpt-6m-pct').textContent = r6.progression_probability || '—';
    document.getElementById('rpt-6m-bad').textContent = r6.if_untreated || '';
    document.getElementById('rpt-6m-good').textContent = r6.if_managed || '';
    document.getElementById('rpt-12m-pct').textContent = r12.progression_probability || '—';
    document.getElementById('rpt-12m-bad').textContent = r12.if_untreated || '';
    document.getElementById('rpt-12m-good').textContent = r12.if_managed || '';

    // Actions
    const actionsList = document.getElementById('rpt-actions');
    const actions = r.action_plan || [];
    actionsList.innerHTML = (Array.isArray(actions) ? actions : [actions]).map(a => `<li>${a}</li>`).join('');

    // Diet
    const dietList = document.getElementById('rpt-diet');
    const diet = r.diet_recommendations || [];
    dietList.innerHTML = (Array.isArray(diet) ? diet : [diet]).map(d => `<li>${d}</li>`).join('');

    // Follow-up
    document.getElementById('rpt-followup').textContent = r.follow_up || '—';
}

// === Translate Report ===
async function translateReport() {
    const langSelect = document.getElementById('report-language');
    const lang = langSelect.value;
    if (lang === 'english') {
        if (currentReport) displayReportFields(currentReport);
        return;
    }
    if (!currentReport) { alert('No report to translate.'); return; }

    // Show loading
    const diagEl = document.getElementById('rpt-diagnosis');
    const origText = diagEl.textContent;
    diagEl.textContent = 'Translating to ' + (lang === 'hindi' ? 'Hindi' : 'Gujarati') + '...';
    langSelect.disabled = true;

    try {
        const res = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ report: currentReport, language: lang })
        });
        const data = await res.json();
        langSelect.disabled = false;
        if (data.success && data.report) {
            displayReportFields(data.report);
        } else {
            diagEl.textContent = origText;
            alert('Translation failed: ' + (data.error || 'Unknown error'));
            langSelect.value = 'english';
        }
    } catch (e) {
        langSelect.disabled = false;
        diagEl.textContent = origText;
        alert('Translation error: ' + e.message);
        langSelect.value = 'english';
    }
}

function displayReportFields(r) {
    if (r.diagnosis) document.getElementById('rpt-diagnosis').textContent = r.diagnosis;
    if (r.heatmap_analysis) document.getElementById('rpt-heatmap').textContent = r.heatmap_analysis;
    if (r.vessel_analysis) document.getElementById('rpt-vessels').textContent = r.vessel_analysis;
    const risk = r.risk_prediction || {};
    const r6 = risk['6_month'] || {};
    const r12 = risk['12_month'] || {};
    if (r6.progression_probability) document.getElementById('rpt-6m-pct').textContent = r6.progression_probability;
    if (r6.if_untreated) document.getElementById('rpt-6m-bad').textContent = r6.if_untreated;
    if (r6.if_managed) document.getElementById('rpt-6m-good').textContent = r6.if_managed;
    if (r12.progression_probability) document.getElementById('rpt-12m-pct').textContent = r12.progression_probability;
    if (r12.if_untreated) document.getElementById('rpt-12m-bad').textContent = r12.if_untreated;
    if (r12.if_managed) document.getElementById('rpt-12m-good').textContent = r12.if_managed;
    const actionsList = document.getElementById('rpt-actions');
    const actions = r.action_plan || [];
    if (actions.length) actionsList.innerHTML = (Array.isArray(actions) ? actions : [actions]).map(a => `<li>${a}</li>`).join('');
    const dietList = document.getElementById('rpt-diet');
    const diet = r.diet_recommendations || [];
    if (diet.length) dietList.innerHTML = (Array.isArray(diet) ? diet : [diet]).map(d => `<li>${d}</li>`).join('');
    if (r.follow_up) document.getElementById('rpt-followup').textContent = r.follow_up;
}

// === BATCH UPLOAD ===
function handleBatchFile(file) {
    if (!file.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
    batchFile = file;
    document.getElementById('batch-drop-zone').style.display = 'none';
    document.getElementById('batch-preview').style.display = 'flex';
    document.getElementById('batch-preview-name').textContent = file.name;
    const reader = new FileReader();
    reader.onload = e => document.getElementById('batch-preview-img').src = e.target.result;
    reader.readAsDataURL(file);
}

async function addToQueue() {
    const name = document.getElementById('batch-name').value.trim();
    if (!name) { alert('Please enter patient name.'); return; }
    if (!batchFile) { alert('Please upload a fundus image.'); return; }

    const age = parseInt(document.getElementById('batch-age').value) || null;
    const gender = document.getElementById('batch-gender') ? document.getElementById('batch-gender').value : '';
    const diabetes = parseInt(document.getElementById('batch-diabetes') ? document.getElementById('batch-diabetes').value : '') || null;
    const sugar = parseFloat(document.getElementById('batch-sugar') ? document.getElementById('batch-sugar').value : '') || null;
    const hba1c = parseFloat(document.getElementById('batch-hba1c') ? document.getElementById('batch-hba1c').value : '') || null;

    batchQueue.push({ name, age, gender, diabetes_duration: diabetes, sugar_level: sugar, hba1c, file: batchFile, status: 'pending', result: null });
    renderBatchQueue();

    // Reset form
    document.getElementById('batch-name').value = '';
    document.getElementById('batch-age').value = '';
    if (document.getElementById('batch-gender')) document.getElementById('batch-gender').value = '';
    if (document.getElementById('batch-diabetes')) document.getElementById('batch-diabetes').value = '';
    if (document.getElementById('batch-sugar')) document.getElementById('batch-sugar').value = '';
    if (document.getElementById('batch-hba1c')) document.getElementById('batch-hba1c').value = '';
    batchFile = null;
    document.getElementById('batch-drop-zone').style.display = 'block';
    document.getElementById('batch-preview').style.display = 'none';
    document.getElementById('batch-file-input').value = '';

    // Show start button
    document.getElementById('start-batch-btn').style.display = 'inline-flex';
}

function renderBatchQueue() {
    const container = document.getElementById('batch-queue');
    if (batchQueue.length === 0) {
        container.innerHTML = '<p class="empty-state">Queue is empty.</p>';
        document.getElementById('start-batch-btn').style.display = 'none';
        return;
    }

    container.innerHTML = batchQueue.map((item, i) => {
        const statusClass = 'qi-' + item.status;
        const statusText = item.status === 'pending' ? '⏳ Pending' :
                          item.status === 'processing' ? '🔄 Processing...' :
                          item.status === 'done' ? `✅ Stage ${item.result?.detection?.stage || '?'}` :
                          '❌ Error';
        return `<div class="queue-item">
            <div><span class="qi-name">${item.name}</span> ${item.age ? '<span style="color:var(--text-muted);font-size:0.8rem"> • ' + item.age + ' yrs</span>' : ''}</div>
            <span class="qi-status ${statusClass}">${statusText}</span>
        </div>`;
    }).join('');
}

async function startBatchProcessing() {
    document.getElementById('start-batch-btn').style.display = 'none';

    for (let i = 0; i < batchQueue.length; i++) {
        if (batchQueue[i].status !== 'pending') continue;

        batchQueue[i].status = 'processing';
        renderBatchQueue();

        try {
            // Create patient first
            const patientRes = await fetch('/api/patients', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: batchQueue[i].name, age: batchQueue[i].age, gender: batchQueue[i].gender, diabetes_duration: batchQueue[i].diabetes_duration, sugar_level: batchQueue[i].sugar_level, hba1c: batchQueue[i].hba1c })
            });
            const patientData = await patientRes.json();

            if (patientData.success) {
                // Run analysis
                const formData = new FormData();
                formData.append('image', batchQueue[i].file);
                formData.append('patient_id', patientData.patient.id);

                const analysisRes = await fetch('/analyze', { method: 'POST', body: formData });
                const analysisData = await analysisRes.json();

                batchQueue[i].status = 'done';
                batchQueue[i].result = analysisData;
            } else {
                batchQueue[i].status = 'error';
            }
        } catch (e) {
            batchQueue[i].status = 'error';
            console.error('Batch item error:', e);
        }
        renderBatchQueue();
    }

    // All done - show completion
    const allDone = batchQueue.every(q => q.status === 'done');
    if (allDone) {
        alert(`✅ Batch complete! ${batchQueue.length} patients processed.`);
        batchQueue = [];
        renderBatchQueue();
    }
}

// === Helpers ===
function formatDate(dateStr) {
    if (!dateStr) return '—';
    try {
        const d = new Date(dateStr + 'Z');
        return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch { return dateStr; }
}

function downloadReport() {
    if (!currentReport) { alert('No report available.'); return; }
    const r = currentReport;
    const stage = document.getElementById('result-stage').textContent;
    const stageName = document.getElementById('result-name').textContent;
    const conf = document.getElementById('result-confidence').textContent;
    const risk = r.risk_prediction || {};
    const r6 = risk['6_month'] || {};
    const r12 = risk['12_month'] || {};
    const date = new Date().toLocaleDateString('en-IN', {day:'numeric',month:'long',year:'numeric'});

    // Grab retinal images from the page
    const imgOrig = document.getElementById('r-original')?.src || '';
    const imgVessel = document.getElementById('r-vessels')?.src || '';
    const imgHeatmap = document.getElementById('r-heatmap')?.src || '';

    const actionsHtml = (r.action_plan || []).map(a => `<li>${a}</li>`).join('');
    const dietHtml = (r.diet_recommendations || []).map(d => `<li>${d}</li>`).join('');

    const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>OptiGemma Report - ${date}</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',sans-serif;color:#1a1a2e;padding:40px;max-width:800px;margin:auto;line-height:1.6}
.header{text-align:center;border-bottom:3px solid #4f8ff7;padding-bottom:20px;margin-bottom:24px}
.header h1{font-size:22px;color:#4f8ff7;margin-bottom:4px}.header p{font-size:12px;color:#666}
.stage-box{display:inline-flex;align-items:center;gap:12px;background:#f0f7ff;border:2px solid #4f8ff7;border-radius:12px;padding:12px 24px;margin:16px 0}
.stage-num{font-size:36px;font-weight:800;color:#4f8ff7}.stage-info h2{font-size:16px}.stage-info p{font-size:12px;color:#666}
.images-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:20px 0}
.img-box{text-align:center;border:1px solid #e0e3e8;border-radius:8px;overflow:hidden}
.img-box img{width:100%;display:block;background:#000}.img-box .img-label{padding:6px;font-size:11px;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:0.5px;background:#f5f7fa}
.section{margin:20px 0;page-break-inside:avoid}.section h3{font-size:14px;color:#4f8ff7;border-left:4px solid #4f8ff7;padding-left:10px;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px}
.section p{font-size:13px;color:#333;margin-bottom:4px}
.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}
.risk-card{border:1px solid #e0e3e8;border-radius:8px;padding:12px;border-left:4px solid #4f8ff7}
.risk-card h4{font-size:11px;color:#666;text-transform:uppercase;margin-bottom:4px}.risk-card .pct{font-size:20px;font-weight:800;color:#1a1a2e;margin-bottom:4px}
.risk-card .bad{font-size:11px;color:#d93025}.risk-card .good{font-size:11px;color:#0f9d58}
ul{list-style:none;display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px}
li{font-size:12px;padding:6px 10px;background:#f5f7fa;border-radius:6px;border-left:3px solid #0f9d58}
.disclaimer{margin-top:24px;padding:12px;background:#fff8e1;border:1px solid #e0c800;border-radius:8px;font-size:11px;color:#8a6d00;text-align:center}
.footer{text-align:center;margin-top:20px;font-size:10px;color:#999}
@media print{body{padding:20px}@page{margin:15mm}}
</style></head><body>
<div class="header"><h1>🔬 OptiGemma Diagnostic Report</h1><p>AI-Powered Retinal Analysis • ${date}</p></div>
<div style="text-align:center"><div class="stage-box"><span class="stage-num">${stage}</span><div class="stage-info"><h2>${stageName}</h2><p>${conf}</p></div></div></div>
<div class="images-row">
<div class="img-box"><img src="${imgOrig}"><div class="img-label">Original Scan</div></div>
<div class="img-box"><img src="${imgVessel}"><div class="img-label">Vessel Map</div></div>
<div class="img-box"><img src="${imgHeatmap}"><div class="img-label">AI Heatmap</div></div>
</div>
<div class="section"><h3>🔍 Current Status</h3><p>${r.diagnosis || '—'}</p></div>
<div class="section"><h3>👁️ Visual Findings</h3><p>${r.heatmap_analysis || '—'}</p><p>${r.vessel_analysis || ''}</p></div>
<div class="section"><h3>📊 Risk Prediction</h3><div class="risk-grid">
<div class="risk-card"><h4>6-Month Outlook</h4><div class="pct">${r6.progression_probability || '—'}</div><div class="bad">▼ ${r6.if_untreated || ''}</div><div class="good">▲ ${r6.if_managed || ''}</div></div>
<div class="risk-card"><h4>12-Month Outlook</h4><div class="pct">${r12.progression_probability || '—'}</div><div class="bad">▼ ${r12.if_untreated || ''}</div><div class="good">▲ ${r12.if_managed || ''}</div></div>
</div></div>
<div class="section"><h3>✅ Action Plan</h3><ul>${actionsHtml}</ul></div>
<div class="section"><h3>🥗 Diet Recommendations</h3><ul>${dietHtml}</ul></div>
<div class="section"><h3>📅 Follow-Up</h3><p>${r.follow_up || '—'}</p></div>
<div class="disclaimer">⚠️ This is an AI-assisted screening tool. Please consult a qualified ophthalmologist for final diagnosis and treatment.</div>
<div class="footer">OptiGemma v2.0 — Gemma-4 Hackathon • Generated on ${date}</div>
<script>window.onload=function(){window.print()}</script>
</body></html>`;

    const w = window.open('', '_blank');
    w.document.write(html);
    w.document.close();
}
