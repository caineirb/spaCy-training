/**
 * spaCy NER Evaluation Scanner - Application Logic
 * Comprehensive human-readable viewer for held-out tests, unseen benchmarks,
 * and dictionary overrides with 3-way model comparison.
 */

// Application State
const state = {
  activeFileType: 'held_out_test',
  activeModel: 'trtr',          // 'trtr' | 'trstr_paraphrase' | 'trstr_llm' | 'compare'
  activeScope: 'all',           // 'all' | 'errors_only'
  searchQuery: '',
  activeStatusFilter: 'all',    // 'all' | 'correct' | 'boundary_error' | 'false_negative' | 'false_positive' | 'label_error'
  activeLabelFilter: 'all',     // 'all' | 'IT_TERM' | 'CLERICAL_TERM'
  activeSourceFilter: 'all',    // 'all' | 'ML' | 'dictionary'
  sbsFilter: 'all',             // 'all' | 'llm_win' | 'disagreement' | 'all_failed' | 'unanimous'
  page: 1,
  pageSize: 15,
  theme: localStorage.getItem('ner_eval_theme') || 'dark',
  data: window.EVAL_DATA || null,
  activeTooltipSpan: null
};

// Model display metadata
const MODEL_CONFIG = {
  trtr: {
    name: 'TRTR Baseline',
    shortName: 'TRTR',
    color: '#3b82f6',
    desc: 'Trained on 923 authentic real records only (No synthetic data)'
  },
  trstr_paraphrase: {
    name: 'TRSTR-Paraphrase',
    shortName: 'Paraphrase',
    color: '#a855f7',
    desc: 'Trained with real records + heuristic sentence paraphrasing'
  },
  trstr_llm: {
    name: 'TRSTR-LLM (High Recall)',
    shortName: 'TRSTR-LLM',
    color: '#10b981',
    desc: 'Trained with real records + Gemini 2.5 Flash LLM synthetic examples'
  }
};

// File type display metadata
const FILE_TYPE_CONFIG = {
  held_out_test: {
    name: 'Held-Out Test Set',
    icon: '📄',
    desc: '199 human-annotated authentic test documents evaluating in-distribution NER generalization.'
  },
  unseen_benchmark_hybrid: {
    name: 'Unseen Benchmark (Hybrid)',
    icon: '⚡',
    desc: '85 out-of-domain sentences with modern tech stacks evaluated via Hybrid (EntityRuler + Transformer).'
  },
  unseen_benchmark_transformer_only: {
    name: 'Unseen Benchmark (Transformer Only)',
    icon: '🤖',
    desc: '85 out-of-domain sentences evaluated purely with ML Transformer NER model.'
  },
  unseen_benchmark_entity_ruler_only: {
    name: 'Unseen Benchmark (Entity Ruler Only)',
    icon: '📖',
    desc: '85 out-of-domain sentences evaluated purely with Dictionary exact-match EntityRuler.'
  },
  dictionary_overrides: {
    name: 'Dictionary Overrides',
    icon: '🔄',
    desc: 'Audit trail of runtime cases where EntityRuler dictionary took precedence over conflicting ML predictions.'
  },
  matrix_summary: {
    name: '3-Way Benchmark Matrix',
    icon: '📊',
    desc: 'Executive performance comparison across TRTR, TRSTR-Paraphrase, and TRSTR-LLM.'
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  applyTheme(state.theme);
  setupEventListeners();

  if (!state.data) {
    await fetchEvaluationData();
  } else {
    renderApp();
  }
});

/**
 * Fetch evaluation data from local server if not already embedded
 */
async function fetchEvaluationData() {
  try {
    const res = await fetch('/api/data');
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    state.data = await res.json();
    renderApp();
  } catch (err) {
    console.error('Failed to fetch data from /api/data:', err);
    showToast('Failed to load evaluation data from local server.');
  }
}

/**
 * Setup Global Event Handlers
 */
function setupEventListeners() {
  // Theme Toggle
  const themeBtn = document.getElementById('theme-toggle-btn');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      state.theme = state.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('ner_eval_theme', state.theme);
      applyTheme(state.theme);
    });
  }

  // Search Input
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', debounce((e) => {
      state.searchQuery = e.target.value.trim().toLowerCase();
      state.page = 1;
      renderCurrentView();
    }, 200));
  }

  // Label filter dropdown
  const labelFilter = document.getElementById('label-filter-select');
  if (labelFilter) {
    labelFilter.addEventListener('change', (e) => {
      state.activeLabelFilter = e.target.value;
      state.page = 1;
      renderCurrentView();
    });
  }

  // Source filter dropdown
  const sourceFilter = document.getElementById('source-filter-select');
  if (sourceFilter) {
    sourceFilter.addEventListener('change', (e) => {
      state.activeSourceFilter = e.target.value;
      state.page = 1;
      renderCurrentView();
    });
  }

  // Page size dropdown
  const pageSizeSelect = document.getElementById('page-size-select');
  if (pageSizeSelect) {
    pageSizeSelect.addEventListener('change', (e) => {
      state.pageSize = e.target.value === 'all' ? 999999 : parseInt(e.target.value, 10);
      state.page = 1;
      renderCurrentView();
    });
  }

  // Tooltip dismissal on outside click/hover
  document.addEventListener('scroll', hideTooltip, true);
  window.addEventListener('resize', hideTooltip);
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const themeIcon = document.getElementById('theme-icon');
  if (themeIcon) {
    themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
  }
}

/**
 * Main Render Dispatcher
 */
function renderApp() {
  if (!state.data) return;

  renderHeaderStats();
  renderPrimaryTabs();
  renderSubNav();
  renderCurrentView();
}

/**
 * Render Header Quick Stats
 */
function renderHeaderStats() {
  const container = document.getElementById('quick-stats-container');
  if (!container || !state.data) return;

  const dictCount = state.data.dictionary_overrides ? state.data.dictionary_overrides.length : 0;
  const heldOutCount = state.data.datasets?.held_out_test?.trtr?.all?.length || 199;
  const unseenCount = state.data.datasets?.unseen_benchmark_hybrid?.trtr?.all?.length || 85;

  container.innerHTML = `
    <div class="quick-stats-pill">
      <span>Held-Out: <strong>${heldOutCount}</strong></span>
    </div>
    <div class="quick-stats-pill">
      <span>Unseen Bench: <strong>${unseenCount}</strong></span>
    </div>
    <div class="quick-stats-pill">
      <span>Dict Overrides: <strong>${dictCount}</strong></span>
    </div>
  `;
}

/**
 * Render Primary Navigation Tabs (File Types)
 */
function renderPrimaryTabs() {
  const nav = document.getElementById('primary-tabs-nav');
  if (!nav) return;

  const tabs = [
    { id: 'held_out_test', label: 'Held-Out Test Set', count: 199, icon: '📄' },
    { id: 'unseen_benchmark_hybrid', label: 'Unseen (Hybrid)', count: 85, icon: '⚡' },
    { id: 'unseen_benchmark_transformer_only', label: 'Unseen (Transformer)', count: 85, icon: '🤖' },
    { id: 'unseen_benchmark_entity_ruler_only', label: 'Unseen (Entity Ruler)', count: 85, icon: '📖' },
    { id: 'dictionary_overrides', label: 'Dictionary Overrides', count: state.data?.dictionary_overrides?.length || 18, icon: '🔄' },
    { id: 'matrix_summary', label: '3-Way Benchmark Matrix', count: 'Report', icon: '📊' }
  ];

  nav.innerHTML = tabs.map(tab => `
    <button class="file-tab-btn ${state.activeFileType === tab.id ? 'active' : ''}" 
            data-tab-id="${tab.id}"
            onclick="switchFileType('${tab.id}')">
      <span>${tab.icon}</span>
      <span>${tab.label}</span>
      <span class="tab-badge">${tab.count}</span>
    </button>
  `).join('');
}

/**
 * Switch Active Primary File Type Tab
 */
function switchFileType(fileTypeId) {
  state.activeFileType = fileTypeId;
  state.page = 1;
  state.activeStatusFilter = 'all';

  renderPrimaryTabs();
  renderSubNav();
  renderCurrentView();
}

/**
 * Render Sub-Navigation Bar (Models, Scope Toggle, and Filter Chips)
 */
function renderSubNav() {
  const container = document.getElementById('subnav-wrapper');
  if (!container) return;

  // If dictionary overrides or summary matrix, show custom controls
  if (state.activeFileType === 'dictionary_overrides' || state.activeFileType === 'matrix_summary') {
    container.style.display = state.activeFileType === 'dictionary_overrides' ? 'flex' : 'none';
    if (state.activeFileType === 'dictionary_overrides') {
      container.innerHTML = `
        <div class="filter-toolbar" style="border:none; padding:0;">
          <div class="search-input-wrapper">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <input type="text" class="search-input" id="search-input" placeholder="Search terms, sentences..." value="${escapeHtml(state.searchQuery)}">
          </div>
          <div class="quick-stats-pill">
            <span>Showing all <strong>${state.data?.dictionary_overrides?.length || 0}</strong> runtime dictionary precedence overrides</span>
          </div>
        </div>
      `;
      setupSearchInputListener();
    }
    return;
  }

  container.style.display = 'flex';

  // Get counts for current file type
  const counts = getFileTypeCounts(state.activeFileType);

  container.innerHTML = `
    <div class="model-selection-bar">
      <!-- Model Pills -->
      <div class="model-pills-group">
        <button class="model-pill-btn ${state.activeModel === 'trtr' ? 'active' : ''}" 
                data-model="trtr" onclick="switchModel('trtr')">
          <span class="indicator-dot trtr"></span>
          <span>TRTR Baseline</span>
        </button>
        <button class="model-pill-btn ${state.activeModel === 'trstr_paraphrase' ? 'active' : ''}" 
                data-model="trstr_paraphrase" onclick="switchModel('trstr_paraphrase')">
          <span class="indicator-dot para"></span>
          <span>TRSTR-Paraphrase</span>
        </button>
        <button class="model-pill-btn ${state.activeModel === 'trstr_llm' ? 'active' : ''}" 
                data-model="trstr_llm" onclick="switchModel('trstr_llm')">
          <span class="indicator-dot llm"></span>
          <span>TRSTR-LLM (High Recall)</span>
        </button>
        <button class="model-pill-btn ${state.activeModel === 'compare' ? 'active' : ''}" 
                data-model="compare" onclick="switchModel('compare')">
          <span class="indicator-dot compare"></span>
          <span>⚖️ Side-by-Side (3-Way)</span>
        </button>
      </div>

      <!-- Scope Toggle: All vs Errors Only (hide in side-by-side mode) -->
      ${state.activeModel !== 'compare' ? `
        <div class="scope-toggle-group">
          <button class="scope-btn ${state.activeScope === 'all' ? 'active' : ''}" 
                  onclick="switchScope('all')">
            All Records (${counts[state.activeModel]?.all || 0})
          </button>
          <button class="scope-btn errors-only ${state.activeScope === 'errors_only' ? 'active' : ''}" 
                  onclick="switchScope('errors_only')">
            ⚠️ Errors Only (${counts[state.activeModel]?.errors || 0})
          </button>
        </div>
      ` : `
        <!-- Side-by-Side Filter Pills -->
        <div class="scope-toggle-group">
          <button class="scope-btn ${state.sbsFilter === 'all' ? 'active' : ''}" onclick="switchSbsFilter('all')">All Sentences</button>
          <button class="scope-btn ${state.sbsFilter === 'llm_win' ? 'active' : ''}" onclick="switchSbsFilter('llm_win')">🟢 LLM Fixes Error</button>
          <button class="scope-btn ${state.sbsFilter === 'disagreement' ? 'active' : ''}" onclick="switchSbsFilter('disagreement')">🟡 Disagreements</button>
          <button class="scope-btn ${state.sbsFilter === 'all_failed' ? 'active' : ''}" onclick="switchSbsFilter('all_failed')">🔴 All Failed</button>
        </div>
      `}
    </div>

    <!-- Filter Toolbar -->
    <div class="filter-toolbar">
      <div class="search-input-wrapper">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        <input type="text" class="search-input" id="search-input" placeholder="Search sentence text, gold entities, or predictions..." value="${escapeHtml(state.searchQuery)}">
      </div>

      ${state.activeModel !== 'compare' ? `
        <div class="status-filter-pills" id="status-chips-container">
          <!-- Populated by updateStatusChips() -->
        </div>

        <div class="secondary-filters">
          <select class="select-custom" id="label-filter-select" onchange="state.activeLabelFilter = this.value; state.page = 1; renderCurrentView();">
            <option value="all" ${state.activeLabelFilter === 'all' ? 'selected' : ''}>All Labels</option>
            <option value="IT_TERM" ${state.activeLabelFilter === 'IT_TERM' ? 'selected' : ''}>IT_TERM</option>
            <option value="CLERICAL_TERM" ${state.activeLabelFilter === 'CLERICAL_TERM' ? 'selected' : ''}>CLERICAL_TERM</option>
          </select>
          <select class="select-custom" id="source-filter-select" onchange="state.activeSourceFilter = this.value; state.page = 1; renderCurrentView();">
            <option value="all" ${state.activeSourceFilter === 'all' ? 'selected' : ''}>All Sources</option>
            <option value="ML" ${state.activeSourceFilter === 'ML' ? 'selected' : ''}>ML Model Only</option>
            <option value="dictionary" ${state.activeSourceFilter === 'dictionary' ? 'selected' : ''}>Dictionary Only</option>
          </select>
        </div>
      ` : ''}
    </div>
  `;

  setupSearchInputListener();
  if (state.activeModel !== 'compare') {
    updateStatusChips();
  }
}

function setupSearchInputListener() {
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', debounce((e) => {
      state.searchQuery = e.target.value.trim().toLowerCase();
      state.page = 1;
      renderCurrentView();
    }, 200));
  }
}

function switchModel(modelId) {
  state.activeModel = modelId;
  state.page = 1;
  renderSubNav();
  renderCurrentView();
}

function switchScope(scopeId) {
  state.activeScope = scopeId;
  state.page = 1;
  renderSubNav();
  renderCurrentView();
}

function switchSbsFilter(filterId) {
  state.sbsFilter = filterId;
  state.page = 1;
  renderSubNav();
  renderCurrentView();
}

function switchStatusFilter(status) {
  state.activeStatusFilter = status;
  state.page = 1;
  updateStatusChips();
  renderCurrentView();
}

/**
 * Compute counts for file types and models
 */
function getFileTypeCounts(fileTypeId) {
  const out = {};
  ['trtr', 'trstr_paraphrase', 'trstr_llm'].forEach(m => {
    const ds = state.data?.datasets?.[fileTypeId]?.[m];
    out[m] = {
      all: ds?.all?.length || 0,
      errors: ds?.errors_only?.length || 0
    };
  });
  return out;
}

/**
 * Update Status Filter Chips with live counts
 */
function updateStatusChips() {
  const container = document.getElementById('status-chips-container');
  if (!container) return;

  const dataset = getActiveDataset();
  const counts = {
    all: dataset.length,
    correct: 0,
    boundary_error: 0,
    false_negative: 0,
    false_positive: 0,
    label_error: 0
  };

  dataset.forEach(row => {
    const statuses = new Set((row.entity_results || []).map(er => er.status));
    statuses.forEach(s => {
      if (counts[s] !== undefined) counts[s]++;
    });
  });

  const chips = [
    { id: 'all', label: 'All', count: counts.all, cls: '' },
    { id: 'correct', label: '✅ Correct', count: counts.correct, cls: 'correct' },
    { id: 'boundary_error', label: '⚠️ Boundary', count: counts.boundary_error, cls: 'boundary' },
    { id: 'false_positive', label: '🚫 Spurious (FP)', count: counts.false_positive, cls: 'fp' },
    { id: 'false_negative', label: '❌ Missed (FN)', count: counts.false_negative, cls: 'fn' },
    { id: 'label_error', label: '🏷️ Label Conf', count: counts.label_error, cls: 'label' }
  ];

  container.innerHTML = chips.map(chip => `
    <button class="chip-btn ${chip.cls} ${state.activeStatusFilter === chip.id ? 'active' : ''}" 
            onclick="switchStatusFilter('${chip.id}')">
      <span>${chip.label}</span>
      <span class="chip-count">${chip.count}</span>
    </button>
  `).join('');
}

/**
 * Retrieve raw list of records for current FileType, Model, and Scope
 */
function getActiveDataset() {
  if (!state.data || !state.data.datasets) return [];
  const ft = state.data.datasets[state.activeFileType];
  if (!ft) return [];
  const modelData = ft[state.activeModel];
  if (!modelData) return [];

  return state.activeScope === 'errors_only' ? (modelData.errors_only || []) : (modelData.all || []);
}

/**
 * Render Current Active View
 */
function renderCurrentView() {
  const main = document.getElementById('main-view-container');
  if (!main) return;

  if (state.activeFileType === 'matrix_summary') {
    renderBenchmarkMatrixView(main);
    return;
  }

  if (state.activeFileType === 'dictionary_overrides') {
    renderDictionaryOverridesView(main);
    return;
  }

  if (state.activeModel === 'compare') {
    renderSideBySideView(main);
    return;
  }

  renderStandardModelView(main);
}

/**
 * Render Standard Model View (Single Model Inspection)
 */
function renderStandardModelView(container) {
  const rawRecords = getActiveDataset();
  const filtered = filterRecords(rawRecords);

  // Pagination slice
  const total = filtered.length;
  const startIdx = (state.page - 1) * state.pageSize;
  const endIdx = Math.min(startIdx + state.pageSize, total);
  const pagedRecords = filtered.slice(startIdx, endIdx);

  // KPIs
  const kpis = computeKPIs(rawRecords, filtered);

  container.innerHTML = `
    <!-- KPI Summary Grid -->
    <div class="kpi-grid">
      <div class="kpi-card highlight">
        <span class="kpi-label">Filtered Documents</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${total}</span>
          <span class="kpi-subtext">of ${rawRecords.length}</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Exact Match Rate</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${kpis.accuracy}%</span>
          <span class="kpi-subtext">${kpis.correctEntities}/${kpis.goldEntities} Gold</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Boundary Errors</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#fbbf24">${kpis.boundaryErrors}</span>
          <span class="kpi-subtext">Span mismatch</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Spurious Predictions (FP)</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#fb923c">${kpis.falsePositives}</span>
          <span class="kpi-subtext">Over-generation</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Missed Gold (FN)</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#fb7185">${kpis.falseNegatives}</span>
          <span class="kpi-subtext">Recall omissions</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Mean Confidence</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#38bdf8">${kpis.avgConfidence}%</span>
          <span class="kpi-subtext">Model certainty</span>
        </div>
      </div>
    </div>

    <!-- Results Header / Pagination -->
    <div class="results-header-info">
      <div>
        Showing records <strong>${total > 0 ? startIdx + 1 : 0}–${endIdx}</strong> of <strong>${total}</strong>
        ${state.searchQuery ? ` matching "${escapeHtml(state.searchQuery)}"` : ''}
      </div>
      ${renderPaginationControls(total, state.page, state.pageSize)}
    </div>

    <!-- Document Sentence Cards -->
    <div class="results-container">
      ${pagedRecords.length > 0 ? pagedRecords.map((doc, idx) => renderDocCard(doc, startIdx + idx)).join('') : `
        <div class="empty-state">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <h3>No records match the active criteria</h3>
          <p>Try resetting the search query or selecting a different status filter.</p>
        </div>
      `}
    </div>

    <!-- Bottom Pagination -->
    ${total > state.pageSize ? `
      <div class="results-header-info" style="margin-top: 1rem;">
        <div></div>
        ${renderPaginationControls(total, state.page, state.pageSize)}
      </div>
    ` : ''}
  `;
}

/**
 * Filter Records based on search, status, label, source
 */
function filterRecords(records) {
  return records.filter(row => {
    // Search query
    if (state.searchQuery) {
      const q = state.searchQuery;
      const textMatch = (row.text || '').toLowerCase().includes(q);
      const goldMatch = (row.gold_entities || []).some(g => (g.term || '').toLowerCase().includes(q));
      const predMatch = (row.pred_entities || []).some(p => (p.term || '').toLowerCase().includes(q));
      if (!textMatch && !goldMatch && !predMatch) return false;
    }

    // Status filter
    if (state.activeStatusFilter !== 'all') {
      const hasStatus = (row.entity_results || []).some(er => er.status === state.activeStatusFilter);
      if (!hasStatus) return false;
    }

    // Label filter
    if (state.activeLabelFilter !== 'all') {
      const hasLabel = (row.entity_results || []).some(er => 
        er.gold_label === state.activeLabelFilter || er.pred_label === state.activeLabelFilter
      );
      if (!hasLabel) return false;
    }

    // Source filter
    if (state.activeSourceFilter !== 'all') {
      const hasSource = (row.pred_entities || []).some(p => p.source === state.activeSourceFilter);
      if (!hasSource) return false;
    }

    return true;
  });
}

/**
 * Compute KPIs across dataset
 */
function computeKPIs(allRecords, filteredRecords) {
  let goldCount = 0;
  let correctCount = 0;
  let boundaryCount = 0;
  let fpCount = 0;
  let fnCount = 0;
  let confSum = 0;
  let confCount = 0;

  filteredRecords.forEach(r => {
    goldCount += (r.gold_entities || []).length;
    (r.pred_entities || []).forEach(p => {
      if (p.confidence !== null && p.confidence !== undefined) {
        confSum += p.confidence;
        confCount++;
      }
    });
    (r.entity_results || []).forEach(er => {
      if (er.status === 'correct') correctCount++;
      else if (er.status === 'boundary_error') boundaryCount++;
      else if (er.status === 'false_positive') fpCount++;
      else if (er.status === 'false_negative') fnCount++;
    });
  });

  const accuracy = goldCount > 0 ? ((correctCount / goldCount) * 100).toFixed(1) : 0;
  const avgConf = confCount > 0 ? ((confSum / confCount) * 100).toFixed(1) : 0;

  return {
    goldEntities: goldCount,
    correctEntities: correctCount,
    boundaryErrors: boundaryCount,
    falsePositives: fpCount,
    falseNegatives: fnCount,
    accuracy,
    avgConfidence: avgConf
  };
}

/**
 * Render an individual Document Sentence Card
 */
function renderDocCard(doc, docIndex) {
  const cardId = `doc-card-${docIndex}`;
  const rawJson = JSON.stringify(doc, null, 2);

  const statuses = Array.from(new Set((doc.entity_results || []).map(er => er.status)));
  const hasErrors = statuses.some(s => s !== 'correct');
  const cardClass = hasErrors ? 'has-error' : 'all-correct';

  const textHighlighted = renderHighlightedSentence(doc.text, doc.entity_results, doc.pred_entities, doc.gold_entities);

  return `
    <div class="doc-card ${cardClass}" id="${cardId}">
      <!-- Header -->
      <div class="doc-card-header">
        <div style="display:flex; align-items:center; gap:0.6rem;">
          <span class="doc-index-badge">#${docIndex + 1}</span>
          <span style="font-size:0.8rem; color:var(--text-muted);">${(doc.text || '').length} chars</span>
        </div>
        <div class="doc-status-tags">
          ${statuses.map(s => `<span class="status-badge ${s}">${formatStatusLabel(s)}</span>`).join('')}
        </div>
      </div>

      <!-- Sentence Text with Highlighted Spans -->
      <div class="sentence-box">
        ${textHighlighted}
      </div>

      <!-- Entity Table Breakdown -->
      ${renderEntityTable(doc.entity_results, doc.pred_entities, doc.gold_entities)}

      <!-- Footer & Raw JSON Drawer -->
      <div class="doc-card-footer">
        <button class="text-btn" onclick="toggleRawJson('${cardId}')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
          <span id="${cardId}-json-btn-text">View Raw JSON</span>
        </button>
        <button class="text-btn" onclick="copyToClipboard('${escapeAttr(rawJson)}', this)">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          <span>Copy JSON</span>
        </button>
      </div>
      <pre class="raw-json-drawer" id="${cardId}-json-drawer">${escapeHtml(rawJson)}</pre>
    </div>
  `;
}

/**
 * Render Sentence with Interactive Spans
 */
function renderHighlightedSentence(text, entityResults = [], predEntities = [], goldEntities = []) {
  if (!text) return '';

  // Extract spans to highlight
  // Sort non-overlapping or boundary spans
  const spans = [];

  (entityResults || []).forEach(er => {
    let start = null;
    let end = null;
    let term = '';
    let label = er.pred_label || er.gold_label || '';
    let status = er.status || 'unknown';
    let conf = er.pred_confidence;
    let src = er.pred_source;

    if (er.gold_start !== null && er.gold_end !== null) {
      start = er.gold_start;
      end = er.gold_end;
      term = er.gold_term || text.slice(start, end);
    } else if (er.pred_term) {
      // Find pred_term in text
      const idx = text.indexOf(er.pred_term);
      if (idx !== -1) {
        start = idx;
        end = idx + er.pred_term.length;
        term = er.pred_term;
      }
    }

    if (start !== null && end !== null && start >= 0 && end <= text.length && start < end) {
      spans.push({
        start,
        end,
        term,
        label,
        status,
        conf,
        src,
        goldTerm: er.gold_term,
        predTerm: er.pred_term,
        goldLabel: er.gold_label,
        predLabel: er.pred_label
      });
    }
  });

  // Sort spans by start offset
  spans.sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));

  // Merge or render disjoint spans
  let html = '';
  let cursor = 0;

  spans.forEach(sp => {
    if (sp.start < cursor) return; // avoid overlap breakages

    if (sp.start > cursor) {
      html += escapeHtml(text.slice(cursor, sp.start));
    }

    const spanText = text.slice(sp.start, sp.end);
    const spanClass = getSpanClass(sp.label, sp.status);
    const tooltipData = encodeURIComponent(JSON.stringify(sp));

    html += `
      <mark class="highlight-span ${spanClass}" 
            onmouseenter="showTooltip(event, '${tooltipData}')" 
            onmouseleave="hideTooltip()">
        ${escapeHtml(spanText)}
        <span class="span-tag-label">${escapeHtml(sp.label || '')}</span>
      </mark>
    `;
    cursor = sp.end;
  });

  if (cursor < text.length) {
    html += escapeHtml(text.slice(cursor));
  }

  return html;
}

function getSpanClass(label, status) {
  let cls = '';
  if (label === 'IT_TERM') cls += ' it-term';
  else if (label === 'CLERICAL_TERM') cls += ' clerical-term';

  if (status === 'boundary_error') cls += ' boundary';
  else if (status === 'false_negative') cls += ' missed';
  else if (status === 'false_positive') cls += ' spurious';

  return cls;
}

/**
 * Render Entity Table Breakdown
 */
function renderEntityTable(entityResults = [], predEntities = [], goldEntities = []) {
  if (!entityResults || entityResults.length === 0) {
    return `
      <div style="font-size:0.8rem; color:var(--text-muted); font-style:italic;">
        No entities annotated or detected in this document.
      </div>
    `;
  }

  return `
    <div class="entity-table-wrapper">
      <table class="entity-table">
        <thead>
          <tr>
            <th>Term</th>
            <th>Evaluation Status</th>
            <th>Gold Ground Truth</th>
            <th>Model Prediction</th>
            <th>Confidence</th>
            <th>Source</th>
            <th>Offsets</th>
          </tr>
        </thead>
        <tbody>
          ${entityResults.map(er => {
            const conf = er.pred_confidence !== null && er.pred_confidence !== undefined
              ? (er.pred_confidence * 100).toFixed(1)
              : null;
            const confClass = conf >= 90 ? 'high' : conf >= 75 ? 'medium' : 'low';
            const term = er.gold_term || er.pred_term || '—';
            const offsets = er.gold_start !== null ? `[${er.gold_start}:${er.gold_end}]` : '—';

            return `
              <tr>
                <td><strong>${escapeHtml(term)}</strong></td>
                <td><span class="status-badge ${er.status}">${formatStatusLabel(er.status)}</span></td>
                <td>
                  ${er.gold_label ? `<span class="badge-label ${er.gold_label === 'IT_TERM' ? 'it' : 'clerical'}">${er.gold_label}</span>` : '<span class="badge-label none">—</span>'}
                </td>
                <td>
                  ${er.pred_label ? `<span class="badge-label ${er.pred_label === 'IT_TERM' ? 'it' : 'clerical'}">${er.pred_label}</span>` : '<span class="badge-label none">—</span>'}
                </td>
                <td>
                  ${conf !== null ? `
                    <div class="confidence-bar-wrapper">
                      <div class="confidence-bar"><div class="confidence-fill ${confClass}" style="width:${conf}%"></div></div>
                      <span style="font-size:0.75rem; font-weight:600;">${conf}%</span>
                    </div>
                  ` : '—'}
                </td>
                <td>
                  <span style="font-size:0.75rem; text-transform:uppercase; font-weight:600; color:${er.pred_source === 'dictionary' ? 'var(--accent-purple)' : 'var(--accent-cyan)'};">
                    ${escapeHtml(er.pred_source || '—')}
                  </span>
                </td>
                <td style="font-family:var(--font-mono); font-size:0.72rem;">${offsets}</td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

/**
 * Render Side-by-Side 3-Way Comparison View
 * Aligns TRTR vs TRSTR-Paraphrase vs TRSTR-LLM on identical sentences
 */
function renderSideBySideView(container) {
  const ft = state.data?.datasets?.[state.activeFileType];
  if (!ft) {
    container.innerHTML = `<div class="empty-state">No dataset available for side-by-side mode.</div>`;
    return;
  }

  const trtrDocs = ft.trtr?.all || [];
  const paraDocs = ft.trstr_paraphrase?.all || [];
  const llmDocs = ft.trstr_llm?.all || [];

  const total = trtrDocs.length;
  const alignedRows = [];

  for (let i = 0; i < total; i++) {
    const t = trtrDocs[i] || {};
    const p = paraDocs[i] || {};
    const l = llmDocs[i] || {};

    const consensus = evaluateConsensus(t, p, l);
    alignedRows.push({
      index: i,
      text: t.text || p.text || l.text || '',
      gold_entities: t.gold_entities || [],
      trtr: t,
      para: p,
      llm: l,
      consensus
    });
  }

  // Filter aligned rows
  const filtered = alignedRows.filter(row => {
    if (state.searchQuery) {
      const q = state.searchQuery;
      const textMatch = row.text.toLowerCase().includes(q);
      const goldMatch = row.gold_entities.some(g => (g.term || '').toLowerCase().includes(q));
      if (!textMatch && !goldMatch) return false;
    }

    if (state.sbsFilter === 'llm_win') return row.consensus.type === 'llm_win';
    if (state.sbsFilter === 'disagreement') return row.consensus.type === 'disagreement' || row.consensus.type === 'llm_win';
    if (state.sbsFilter === 'all_failed') return row.consensus.type === 'all_failed';

    return true;
  });

  // Pagination
  const startIdx = (state.page - 1) * state.pageSize;
  const endIdx = Math.min(startIdx + state.pageSize, filtered.length);
  const pagedRows = filtered.slice(startIdx, endIdx);

  // Summary counts
  const llmWins = alignedRows.filter(r => r.consensus.type === 'llm_win').length;
  const disagreements = alignedRows.filter(r => r.consensus.type === 'disagreement' || r.consensus.type === 'llm_win').length;
  const allFailed = alignedRows.filter(r => r.consensus.type === 'all_failed').length;

  container.innerHTML = `
    <!-- SBS Stats Ribbon -->
    <div class="kpi-grid">
      <div class="kpi-card highlight">
        <span class="kpi-label">Aligned Sentences</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${filtered.length}</span>
          <span class="kpi-subtext">of ${total} total</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">🟢 TRSTR-LLM Fixed Error</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#34d399">${llmWins}</span>
          <span class="kpi-subtext">Improved over baseline</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">🟡 Model Disagreements</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#fbbf24">${disagreements}</span>
          <span class="kpi-subtext">Predictions diverged</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">🔴 All Models Failed</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#fb7185">${allFailed}</span>
          <span class="kpi-subtext">Difficult cases</span>
        </div>
      </div>
    </div>

    <!-- Header & Pagination -->
    <div class="results-header-info">
      <div>Showing aligned records <strong>${filtered.length > 0 ? startIdx + 1 : 0}–${endIdx}</strong> of <strong>${filtered.length}</strong></div>
      ${renderPaginationControls(filtered.length, state.page, state.pageSize)}
    </div>

    <!-- Side-by-Side Cards -->
    <div class="results-container">
      ${pagedRows.map(row => renderSbsCard(row)).join('')}
    </div>

    ${filtered.length > state.pageSize ? `
      <div class="results-header-info" style="margin-top:1rem;">
        <div></div>
        ${renderPaginationControls(filtered.length, state.page, state.pageSize)}
      </div>
    ` : ''}
  `;
}

function evaluateConsensus(trtr, para, llm) {
  const trtrErrors = (trtr.entity_results || []).filter(e => e.status !== 'correct').length;
  const paraErrors = (para.entity_results || []).filter(e => e.status !== 'correct').length;
  const llmErrors = (llm.entity_results || []).filter(e => e.status !== 'correct').length;

  if (llmErrors === 0 && trtrErrors > 0) {
    return { type: 'llm_win', label: '🟢 TRSTR-LLM Fixed Error', desc: 'LLM achieved full precision/recall where baseline failed' };
  }
  if (trtrErrors === 0 && paraErrors === 0 && llmErrors === 0) {
    return { type: 'unanimous', label: '🔵 Unanimous Consensus', desc: 'All 3 models predicted accurately' };
  }
  if (trtrErrors > 0 && paraErrors > 0 && llmErrors > 0) {
    return { type: 'all_failed', label: '🔴 All 3 Erred', desc: 'Complex span or unseen domain entity' };
  }
  return { type: 'disagreement', label: '🟡 Prediction Divergence', desc: 'Models made different boundary or category decisions' };
}

function renderSbsCard(row) {
  const goldEntitiesPills = (row.gold_entities || []).length > 0
    ? (row.gold_entities || []).map(g => `<span class="badge-label ${g.label === 'IT_TERM' ? 'it' : 'clerical'}">${escapeHtml(g.term)} (${g.label})</span>`).join(' ')
    : '<span style="color:var(--text-muted); font-size:0.75rem;">None</span>';

  return `
    <div class="side-by-side-card">
      <div class="sbs-header">
        <div style="display:flex; align-items:center; gap:0.6rem;">
          <span class="doc-index-badge">Sentence #${row.index + 1}</span>
          <span class="sbs-consensus-badge ${row.consensus.type}">${row.consensus.label}</span>
        </div>
        <div style="font-size:0.8rem; color:var(--text-secondary);">
          <strong>Gold Truth:</strong> ${goldEntitiesPills}
        </div>
      </div>

      <!-- Prominent Sentence Display -->
      <div class="sentence-box" style="font-weight:500;">
        ${escapeHtml(row.text)}
      </div>

      <!-- 3 Columns for 3 Models -->
      <div class="sbs-columns-grid">
        <!-- TRTR Column -->
        <div class="sbs-column trtr-col">
          <div class="sbs-col-header">
            <span style="color:#93c5fd;">TRTR Baseline</span>
            <span style="font-size:0.75rem; color:var(--text-muted);">${(row.trtr.pred_entities || []).length} preds</span>
          </div>
          <div class="sbs-col-entities">
            ${renderModelPredPills(row.trtr)}
          </div>
        </div>

        <!-- Paraphrase Column -->
        <div class="sbs-column para-col">
          <div class="sbs-col-header">
            <span style="color:#d8b4fe;">TRSTR-Paraphrase</span>
            <span style="font-size:0.75rem; color:var(--text-muted);">${(row.para.pred_entities || []).length} preds</span>
          </div>
          <div class="sbs-col-entities">
            ${renderModelPredPills(row.para)}
          </div>
        </div>

        <!-- LLM Column -->
        <div class="sbs-column llm-col">
          <div class="sbs-col-header">
            <span style="color:#6ee7b7;">TRSTR-LLM (High Recall)</span>
            <span style="font-size:0.75rem; color:var(--text-muted);">${(row.llm.pred_entities || []).length} preds</span>
          </div>
          <div class="sbs-col-entities">
            ${renderModelPredPills(row.llm)}
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderModelPredPills(modelDoc) {
  const preds = modelDoc.pred_entities || [];
  const results = modelDoc.entity_results || [];

  if (preds.length === 0) {
    return `<span style="font-size:0.75rem; color:var(--text-muted); font-style:italic;">No predictions</span>`;
  }

  return preds.map(p => {
    const res = results.find(r => r.pred_term === p.term) || {};
    const status = res.status || 'unknown';
    const conf = p.confidence ? (p.confidence * 100).toFixed(1) + '%' : '';

    return `
      <div class="sbs-entity-chip">
        <div style="display:flex; align-items:center; gap:0.4rem;">
          <span class="badge-label ${p.label === 'IT_TERM' ? 'it' : 'clerical'}">${p.label}</span>
          <span style="font-weight:600;">${escapeHtml(p.term)}</span>
        </div>
        <div style="display:flex; align-items:center; gap:0.4rem;">
          <span style="font-size:0.72rem; color:var(--text-muted);">${conf}</span>
          <span class="status-badge ${status}" style="font-size:0.65rem; padding:0.1rem 0.35rem;">${formatStatusLabel(status)}</span>
        </div>
      </div>
    `;
  }).join('');
}

/**
 * Render Dictionary Overrides Audit View
 */
function renderDictionaryOverridesView(container) {
  const overrides = state.data?.dictionary_overrides || [];
  const filtered = overrides.filter(o => {
    if (!state.searchQuery) return true;
    const q = state.searchQuery;
    return (o.term || '').toLowerCase().includes(q) ||
           (o.sentence || '').toLowerCase().includes(q) ||
           (o.ml_label || '').toLowerCase().includes(q) ||
           (o.dict_label || '').toLowerCase().includes(q);
  });

  container.innerHTML = `
    <!-- Intro Card -->
    <div class="kpi-grid">
      <div class="kpi-card highlight">
        <span class="kpi-label">Runtime Precedence Overrides</span>
        <div class="kpi-value-row">
          <span class="kpi-value">${filtered.length}</span>
          <span class="kpi-subtext">of ${overrides.length} logged events</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Winning Dict Label: CLERICAL_TERM</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#34d399">${overrides.filter(o => o.winning_label === 'CLERICAL_TERM').length}</span>
          <span class="kpi-subtext">Protected by terms.csv</span>
        </div>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Winning Dict Label: IT_TERM</span>
        <div class="kpi-value-row">
          <span class="kpi-value" style="color:#38bdf8">${overrides.filter(o => o.winning_label === 'IT_TERM').length}</span>
          <span class="kpi-subtext">Protected by terms.csv</span>
        </div>
      </div>
    </div>

    <!-- Overrides Grid -->
    <div class="overrides-grid">
      ${filtered.map((item, idx) => `
        <div class="override-card">
          <div class="override-header">
            <span>Override Event #${idx + 1}</span>
            <span>${item.timestamp ? item.timestamp.split('T')[0] : '—'}</span>
          </div>

          <div style="font-size:0.95rem; font-weight:500; line-height:1.6;">
            ${renderSentenceWithTermHighlight(item.sentence, item.term, item.winning_label)}
          </div>

          <div class="override-comparison-row">
            <div class="override-label-box">
              <span class="override-sublabel">ML Model Output</span>
              <span class="badge-label ${item.ml_label === 'IT_TERM' ? 'it' : 'clerical'}" style="opacity:0.6; text-decoration:line-through;">
                ${item.ml_label}
              </span>
            </div>
            <div class="arrow-divider">➔</div>
            <div class="override-label-box">
              <span class="override-sublabel">Dictionary Precedence (Winner)</span>
              <span class="badge-label ${item.winning_label === 'IT_TERM' ? 'it' : 'clerical'}">
                🛡️ ${item.winning_label}
              </span>
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderSentenceWithTermHighlight(sentence, term, winningLabel) {
  if (!sentence || !term) return escapeHtml(sentence || '');
  const idx = sentence.toLowerCase().indexOf(term.toLowerCase());
  if (idx === -1) return escapeHtml(sentence);

  const before = sentence.slice(0, idx);
  const match = sentence.slice(idx, idx + term.length);
  const after = sentence.slice(idx + term.length);

  return `
    ${escapeHtml(before)}
    <mark class="highlight-span ${winningLabel === 'IT_TERM' ? 'it-term' : 'clerical-term'}" style="box-shadow:0 0 10px rgba(6,182,212,0.3)">
      ${escapeHtml(match)}
      <span class="span-tag-label">${winningLabel}</span>
    </mark>
    ${escapeHtml(after)}
  `;
}

/**
 * Render 3-Way Benchmark Summary Matrix View
 */
function renderBenchmarkMatrixView(container) {
  const summary = state.data?.report_summary;

  container.innerHTML = `
    <div class="matrix-container">
      <!-- Hero Banner -->
      <div class="matrix-hero">
        <div class="matrix-hero-text">
          <h2>3-Way Model Evaluation & Generalization Matrix</h2>
          <p>
            Comparative analysis of baseline real training (<strong>TRTR</strong>), heuristic paraphrase augmentation (<strong>TRSTR-Paraphrase</strong>), 
            and synthetic LLM training (<strong>TRSTR-LLM</strong>). TRSTR-LLM delivers substantial generalization gains on unseen modern tech stack benchmarks.
          </p>
        </div>
        <div style="display:flex; flex-direction:column; gap:0.5rem; text-align:right;">
          <div style="font-size:0.78rem; color:var(--text-muted);">Key Generalization Delta:</div>
          <div style="font-size:1.6rem; font-weight:800; color:#34d399;">+12.31% Recall</div>
          <div style="font-size:0.75rem; color:#6ee7b7;">TRSTR-LLM vs Baseline on Unseen Tech</div>
        </div>
      </div>

      <!-- Comparative Model Cards -->
      <div class="matrix-cards-grid">
        <!-- TRTR Baseline Card -->
        <div class="model-benchmark-card">
          <div class="model-card-title">
            <span class="model-title-text" style="color:#93c5fd;">TRTR Baseline</span>
            <span class="badge-version">Baseline</span>
          </div>
          <p style="font-size:0.82rem; color:var(--text-secondary);">
            Trained exclusively on 923 authentic OJT records. Strong in-domain memorization but vulnerable to domain shift.
          </p>
          <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:0.75rem; display:flex; flex-direction:column; gap:0.4rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Held-Out Precision:</span>
              <strong>65.96%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Held-Out Recall:</span>
              <strong>67.69%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Unseen Benchmark Recall:</span>
              <strong>61.54%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Unseen Spurious Errors:</span>
              <strong style="color:#fb923c;">91 FP</strong>
            </div>
          </div>
        </div>

        <!-- Paraphrase Card -->
        <div class="model-benchmark-card">
          <div class="model-card-title">
            <span class="model-title-text" style="color:#d8b4fe;">TRSTR-Paraphrase</span>
            <span class="badge-version">Augmented</span>
          </div>
          <p style="font-size:0.82rem; color:var(--text-secondary);">
            Augmented with rule-based sentence transformations. Moderately reduces false positives but limited lexicon diversity.
          </p>
          <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:0.75rem; display:flex; flex-direction:column; gap:0.4rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Held-Out Precision:</span>
              <strong>68.12%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Held-Out Recall:</span>
              <strong>67.69%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Unseen Benchmark Recall:</span>
              <strong>63.08%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Unseen Spurious Errors:</span>
              <strong style="color:#fb923c;">82 FP</strong>
            </div>
          </div>
        </div>

        <!-- LLM Winner Card -->
        <div class="model-benchmark-card winner">
          <div class="model-card-title">
            <span class="model-title-text" style="color:#6ee7b7;">TRSTR-LLM</span>
            <span class="badge-version" style="background:#10b981; color:#0f172a; font-weight:700;">Top Performer</span>
          </div>
          <p style="font-size:0.82rem; color:var(--text-secondary);">
            Augmented with 200 synthetic Gemini 2.5 Flash curriculum records. Outstanding recall on modern tech terminology.
          </p>
          <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:0.75rem; display:flex; flex-direction:column; gap:0.4rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Held-Out Precision:</span>
              <strong>67.57%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Held-Out Recall:</span>
              <strong style="color:#34d399;">69.23%</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Unseen Benchmark Recall:</span>
              <strong style="color:#34d399;">73.85% (+12.3%)</strong>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span style="color:var(--text-muted);">Unseen Spurious Errors:</span>
              <strong style="color:#34d399;">52 FP (-43%)</strong>
            </div>
          </div>
        </div>
      </div>

      <!-- Comparative Performance Table -->
      <div class="comparison-table-wrapper">
        <table class="matrix-table">
          <thead>
            <tr>
              <th>Evaluation Test Mode</th>
              <th>Metric</th>
              <th>TRTR Baseline</th>
              <th>TRSTR-Paraphrase</th>
              <th>TRSTR-LLM (High Recall)</th>
              <th>LLM vs Baseline Delta</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td rowspan="3" style="font-weight:600; color:var(--text-primary); border-right:1px solid rgba(255,255,255,0.05);">
                Held-Out Test Set<br><span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">199 authentic documents</span>
              </td>
              <td>Overall Precision</td>
              <td>65.96%</td>
              <td>68.12%</td>
              <td>67.57%</td>
              <td><span class="delta-badge pos">+1.61%</span></td>
            </tr>
            <tr>
              <td>Overall Recall</td>
              <td>67.69%</td>
              <td>67.69%</td>
              <td><strong>69.23%</strong></td>
              <td><span class="delta-badge pos">+1.54%</span></td>
            </tr>
            <tr>
              <td>Overall F1-Score</td>
              <td>66.81%</td>
              <td>67.90%</td>
              <td><strong>68.39%</strong></td>
              <td><span class="delta-badge pos">+1.58%</span></td>
            </tr>

            <tr class="highlight-row">
              <td rowspan="4" style="font-weight:600; color:var(--text-primary); border-right:1px solid rgba(255,255,255,0.05);">
                Unseen Benchmark (Hybrid Pipeline)<br><span style="font-size:0.75rem; color:var(--text-muted); font-weight:normal;">85 modern tech stack sentences</span>
              </td>
              <td>Precision</td>
              <td>30.53%</td>
              <td>32.03%</td>
              <td><strong>37.88%</strong></td>
              <td><span class="delta-badge pos">+7.35%</span></td>
            </tr>
            <tr class="highlight-row">
              <td>Recall</td>
              <td>61.54%</td>
              <td>63.08%</td>
              <td><strong>73.85%</strong></td>
              <td><span class="delta-badge pos">+12.31%</span></td>
            </tr>
            <tr class="highlight-row">
              <td>F1-Score</td>
              <td>40.82%</td>
              <td>42.49%</td>
              <td><strong>50.00%</strong></td>
              <td><span class="delta-badge pos">+9.18%</span></td>
            </tr>
            <tr class="highlight-row">
              <td>False Positives (Spurious)</td>
              <td style="color:#fb923c;">91 errors</td>
              <td style="color:#fb923c;">82 errors</td>
              <td style="color:#34d399;"><strong>52 errors</strong></td>
              <td><span class="delta-badge pos">-39 errors (-42.8%)</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

/**
 * Render Pagination Controls
 */
function renderPaginationControls(totalRecords, currentPage, pageSize) {
  const totalPages = Math.ceil(totalRecords / pageSize);
  if (totalPages <= 1) return '';

  return `
    <div class="pagination-controls">
      <button class="page-btn" ${currentPage <= 1 ? 'disabled' : ''} onclick="changePage(${currentPage - 1})">
        ◀ Prev
      </button>
      <span style="font-size:0.78rem; color:var(--text-muted); margin:0 0.35rem;">
        Page <strong>${currentPage}</strong> of ${totalPages}
      </span>
      <button class="page-btn" ${currentPage >= totalPages ? 'disabled' : ''} onclick="changePage(${currentPage + 1})">
        Next ▶
      </button>
    </div>
  `;
}

function changePage(newPage) {
  state.page = newPage;
  renderCurrentView();
  window.scrollTo({ top: 180, behavior: 'smooth' });
}

/**
 * Interactive Tooltip Logic
 */
function showTooltip(event, encodedData) {
  try {
    const data = JSON.parse(decodeURIComponent(encodedData));
    let tooltip = document.getElementById('global-tooltip');
    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.id = 'global-tooltip';
      tooltip.className = 'tooltip-popover';
      document.body.appendChild(tooltip);
    }

    const conf = data.conf ? (data.conf * 100).toFixed(1) + '%' : '—';
    const statusLabel = formatStatusLabel(data.status);

    tooltip.innerHTML = `
      <div class="tooltip-header">
        <span>${escapeHtml(data.term || '')}</span>
        <span class="badge-label ${data.label === 'IT_TERM' ? 'it' : 'clerical'}">${data.label}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-key">Status:</span>
        <span class="tooltip-val">${statusLabel}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-key">Confidence:</span>
        <span class="tooltip-val">${conf}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-key">Source:</span>
        <span class="tooltip-val">${data.src || 'ML'}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-key">Span:</span>
        <span class="tooltip-val" style="font-family:var(--font-mono)">[${data.start}:${data.end}]</span>
      </div>
    `;

    tooltip.style.display = 'flex';
    const rect = event.target.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();

    let top = rect.top - tooltipRect.height - 8;
    let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);

    if (top < 10) top = rect.bottom + 8;
    if (left < 10) left = 10;
    if (left + tooltipRect.width > window.innerWidth - 10) {
      left = window.innerWidth - tooltipRect.width - 10;
    }

    tooltip.style.top = `${top}px`;
    tooltip.style.left = `${left}px`;
  } catch (err) {
    console.error('Error rendering tooltip:', err);
  }
}

function hideTooltip() {
  const tooltip = document.getElementById('global-tooltip');
  if (tooltip) {
    tooltip.style.display = 'none';
  }
}

/**
 * Toggle Raw JSON Drawer in Document Card
 */
function toggleRawJson(cardId) {
  const drawer = document.getElementById(`${cardId}-json-drawer`);
  const btnText = document.getElementById(`${cardId}-json-btn-text`);
  if (!drawer) return;

  const isOpen = drawer.classList.contains('open');
  if (isOpen) {
    drawer.classList.remove('open');
    if (btnText) btnText.textContent = 'View Raw JSON';
  } else {
    drawer.classList.add('open');
    if (btnText) btnText.textContent = 'Hide Raw JSON';
  }
}

/**
 * Clipboard Copy with Animated Feedback
 */
function copyToClipboard(text, btnElement) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied JSON to clipboard!');
    if (btnElement) {
      const origHtml = btnElement.innerHTML;
      btnElement.innerHTML = `<span>✓ Copied!</span>`;
      setTimeout(() => {
        btnElement.innerHTML = origHtml;
      }, 1500);
    }
  }).catch(err => {
    console.error('Failed to copy to clipboard:', err);
  });
}

/**
 * Toast Notice
 */
function showToast(message) {
  let toast = document.getElementById('global-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'global-toast';
    toast.className = 'toast-notice';
    document.body.appendChild(toast);
  }

  toast.innerHTML = `<span>ℹ️</span> <span>${escapeHtml(message)}</span>`;
  toast.classList.add('show');
  setTimeout(() => {
    toast.classList.remove('show');
  }, 2500);
}

/**
 * Utility Helpers
 */
function formatStatusLabel(status) {
  if (!status) return '—';
  const map = {
    correct: '✅ Correct',
    boundary_error: '⚠️ Boundary Error',
    false_negative: '❌ Missed Gold',
    false_positive: '🚫 Spurious Pred',
    label_error: '🏷️ Label Error'
  };
  return map[status] || status.replace(/_/g, ' ');
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeAttr(str) {
  if (!str) return '';
  return String(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
