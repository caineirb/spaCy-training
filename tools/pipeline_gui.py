#!/usr/bin/env python3
"""
OJT Journal Entity Extraction Studio (GUI)
===========================================
Desktop GUI interface for the Hybrid NER + Classification Inference Pipeline,
mirroring the functionality and JSON response contract of the FastAPI service (/api).

Features:
- Dual-engine Hybrid inference: EntityRuler (dictionary) + Transformer NER (ML).
- Live entity highlighting in journal text with IT_TERM and CLERICAL_TERM coloring.
- Interactive entity inspection table with filtering, sorting, and text-jump navigation.
- Summary analytics cards (counts, percentages, review alerts, inference latency).
- Exact FastAPI JSON contract viewer with 1-click clipboard copy and cURL generator.
- Multi-model switching (TRSTR-LLM, Baseline, TRTR, Custom) and threshold tuning.
- Batch file processing (.jsonl / .txt) with progress tracking and JSONL/CSV export.
- Asynchronous model loading & background inference to prevent UI freezing.
"""

import os
import sys
import re
import json
import time
import queue
import logging
import threading
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------------------------------------------------------
# Path bootstrap — ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import scripts
from scripts.pipeline import HybridJournalPipeline

logger = logging.getLogger("ojt_pipeline.gui")

# ---------------------------------------------------------------------------
# Configuration & Constants (Matching api/main.py)
# ---------------------------------------------------------------------------
MAX_TEXT_LENGTH = 100_000
_WS_RE = re.compile(r"\s+")

DEFAULT_MODEL_CANDIDATES = [
    ("models/ner_trf_trstr_llm/model-best", "TRSTR-LLM (Direct Synthetic, High Recall)"),
    ("models/ner_trf/model-best", "Production Baseline (Real Data)"),
    ("models/ner_trf_trtr/model-best", "TRTR Baseline (Real-Only Archive)"),
    ("models/hybrid_pipeline", "Packaged Hybrid Pipeline"),
]

COLOR_IT_BG = "#dbeafe"          # Light soft blue
COLOR_IT_FG = "#1e40af"          # Deep blue
COLOR_CLERICAL_BG = "#d1fae5"    # Light emerald green
COLOR_CLERICAL_FG = "#065f46"    # Deep forest green
COLOR_REVIEW_BG = "#fef3c7"      # Soft amber
COLOR_REVIEW_FG = "#92400e"      # Dark amber
COLOR_SELECT_BG = "#fde047"      # Bright highlighter yellow
COLOR_SELECT_FG = "#854d0e"      # Deep golden brown

SAMPLE_PRESETS = [
    (
        "Full-Stack Web Dev (Modern Tech)",
        "Configured modern responsive web styling using Tailwind CSS. "
        "Built dynamic user interfaces in Svelte and connected state stores. "
        "Engineered high-performance REST API endpoints with FastAPI and "
        "managed PostgreSQL migrations using Prisma ORM. Deployed Docker containers."
    ),
    (
        "Clerical & Office Administration",
        "Encoded weekly employee timesheets using Microsoft Excel spreadsheets. "
        "Cross-checked physical paper receipts for financial auditing, "
        "prepared official billing invoices, filed corporate tax documents in cabinet archives, "
        "and transcribed meeting minutes into Google Docs."
    ),
    (
        "Mixed Technical & Project Management",
        "Conducted a morning briefing with project stakeholders in Zoom. "
        "Wrote automated Python data scraping scripts and transferred structured outputs "
        "into SQL Server tables. Drafted the final documentation in Microsoft Word "
        "and scheduled deployment sprints in Jira."
    ),
    (
        "Out-of-Vocabulary Probes (Unseen Benchmark)",
        "Configured continuous integration pipelines using GitHub Actions. "
        "Built microservices with NestJS and managed distributed queues using RabbitMQ. "
        "Applied state management in Redux Toolkit and compiled frontend assets with Vite."
    ),
    (
        "Negative Narrative (Non-Task Baseline)",
        "Arrived at the company building around 8:00 AM, greeted colleagues at the lobby, "
        "and attended the mandatory general orientation seminar in the third-floor auditorium. "
        "Ate lunch at the cafeteria and walked around the commercial park before returning."
    ),
]


def _normalise(text: str) -> str:
    """Lowercase and collapse internal whitespace (identical to api/main.py)."""
    return _WS_RE.sub(" ", text.strip().lower())


def _deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group raw entity records by normalised (term, category), count frequency.
    Matches exact api/main.py logic.
    """
    groups: dict[Tuple[str, str], dict] = {}
    freq: dict[Tuple[str, str], int] = defaultdict(int)

    for ent in entities:
        key = (_normalise(ent["term"]), ent["category"])
        freq[key] += 1
        if key not in groups:
            groups[key] = dict(ent)

    deduped = []
    for key, record in groups.items():
        record_copy = dict(record)
        record_copy["frequency"] = freq[key]
        deduped.append(record_copy)

    deduped.sort(key=lambda e: e["start"])
    return deduped


def _build_summary(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build category breakdown summary matching api/main.py."""
    unique = len(entities)
    total_occurrences = sum(e.get("frequency", 1) for e in entities)

    category_counts: dict[str, int] = defaultdict(int)
    for e in entities:
        category_counts[e["category"]] += 1

    it_count = category_counts.get("IT_TERM", 0)
    clerical_count = category_counts.get("CLERICAL_TERM", 0)

    summary: Dict[str, Any] = {
        "total_occurrences": total_occurrences,
        "unique_entities": unique,
        "IT_TERM": it_count,
        "CLERICAL_TERM": clerical_count,
    }

    if unique > 0:
        summary["IT_TERM_percentage"] = round(it_count / unique * 100, 1)
        summary["CLERICAL_TERM_percentage"] = round(clerical_count / unique * 100, 1)
    else:
        summary["IT_TERM_percentage"] = 0.0
        summary["CLERICAL_TERM_percentage"] = 0.0

    return summary


# ---------------------------------------------------------------------------
# Tooltip Helper
# ---------------------------------------------------------------------------
class Tooltip:
    """Floating tooltip for Tkinter widgets and text tags."""
    def __init__(self, widget: tk.Widget):
        self.widget = widget
        self.tip_window: Optional[tk.Toplevel] = None
        self.text = ""

    def show(self, text: str, x: int, y: int):
        self.hide()
        self.text = text
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x + 15}+{y + 15}")
        tw.attributes("-topmost", True)

        label = tk.Label(
            tw,
            text=text,
            justify=tk.LEFT,
            background="#1e293b",
            foreground="#f8fafc",
            relief=tk.SOLID,
            borderwidth=1,
            font=("TkDefaultFont", 9),
            padx=8,
            pady=4,
        )
        label.pack()

    def hide(self):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


# ---------------------------------------------------------------------------
# Main GUI Application Class
# ---------------------------------------------------------------------------
class PipelineGUIApp:
    """Primary Tkinter GUI application for OJT Entity Extraction."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OJT Journal Entity Extraction Studio — Hybrid Pipeline")
        self.root.geometry("1280x880")
        self.root.minsize(1020, 680)

        # Pipeline runtime state
        self.pipeline: Optional[HybridJournalPipeline] = None
        self.is_loading_model = False
        self.is_inferring = False
        self.current_model_path: str = ""
        self.detected_device: str = "Detecting..."
        self.terms_csv_var = tk.StringVar(value="data/terms.csv")

        # Extraction cache
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_api_response: Optional[Dict[str, Any]] = None
        self.displayed_entities: List[Dict[str, Any]] = []

        # Batch state
        self.batch_records: List[Dict[str, Any]] = []
        self.is_batch_running = False

        # Work queue for thread-safe UI updates
        self.work_queue: queue.Queue = queue.Queue()

        # Build UI & Styles
        self._init_styles()
        self._create_menu()
        self._create_layout()

        # Tooltip
        self.tooltip = Tooltip(self.root)

        # Detect GPU / Device info
        self._detect_environment()

        # Discover models & select initial
        self._populate_model_list()

        # Start periodic queue processor
        self.root.after(50, self._process_queue)

        # Auto-load initial model
        self.root.after(200, self.load_selected_model)

    # -----------------------------------------------------------------------
    # Styling
    # -----------------------------------------------------------------------
    def _init_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Card & button styling
        self.style.configure("Header.TLabel", font=("TkDefaultFont", 11, "bold"))
        self.style.configure("Subheader.TLabel", font=("TkDefaultFont", 9), foreground="#64748b")
        self.style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"))
        self.style.configure("Badge.TLabel", font=("TkDefaultFont", 9, "bold"), padding=4)
        self.style.configure("StatValue.TLabel", font=("TkDefaultFont", 16, "bold"))
        self.style.configure("StatLabel.TLabel", font=("TkDefaultFont", 8), foreground="#475569")

    # -----------------------------------------------------------------------
    # Menu Bar
    # -----------------------------------------------------------------------
    def _create_menu(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Text File...", command=self.open_text_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Extracted JSON...", command=self.save_json_file, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Clear Input Text", command=self.clear_input, accelerator="Ctrl+L")
        edit_menu.add_command(label="Copy API JSON", command=self.copy_json_to_clipboard, accelerator="Ctrl+Shift+C")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        # Inference menu
        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="Extract Entities", command=self.run_inference, accelerator="Ctrl+Return")
        run_menu.add_command(label="Reload Model", command=self.load_selected_model, accelerator="Ctrl+R")
        menubar.add_cascade(label="Inference", menu=run_menu)

        # Samples menu
        samples_menu = tk.Menu(menubar, tearoff=0)
        for name, text in SAMPLE_PRESETS:
            samples_menu.add_command(
                label=name,
                command=lambda t=text: self._load_sample_text(t)
            )
        menubar.add_cascade(label="Presets", menu=samples_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="API Contract Guide", command=self._show_api_guide)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

        # Global Keybindings
        self.root.bind("<Control-o>", lambda e: self.open_text_file())
        self.root.bind("<Control-s>", lambda e: self.save_json_file())
        self.root.bind("<Control-l>", lambda e: self.clear_input())
        self.root.bind("<Control-r>", lambda e: self.load_selected_model())
        self.root.bind("<Control-Return>", lambda e: self.run_inference())

    # -----------------------------------------------------------------------
    # Layout Construction
    # -----------------------------------------------------------------------
    def _create_layout(self):
        # Top toolbar: Model & inference config
        self._build_top_toolbar()

        # Main splitter (Top: input & highlighted view | Bottom: analysis tabs)
        self.paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 0))

        # Input Frame (Top Pane)
        self._build_input_pane()

        # Output / Analysis Notebook (Bottom Pane)
        self._build_output_notebook()

        # Status Bar
        self._build_status_bar()

    def _build_top_toolbar(self):
        toolbar = ttk.LabelFrame(self.root, text="Model & Pipeline Configuration", padding=(10, 6))
        toolbar.pack(fill=tk.X, padx=10, pady=(6, 4))

        # --- Row 0: Model & Terms CSV paths ---
        ttk.Label(toolbar, text="Model:").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)

        self.model_combo_var = tk.StringVar()
        self.model_combo = ttk.Combobox(toolbar, textvariable=self.model_combo_var, state="readonly", width=38)
        self.model_combo.grid(row=0, column=1, sticky="ew", padx=(0, 4), pady=2)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_combo_change)

        self.browse_model_btn = ttk.Button(toolbar, text="Browse...", width=9, command=self.browse_model_path)
        self.browse_model_btn.grid(row=0, column=2, padx=(0, 16), pady=2)

        ttk.Label(toolbar, text="Terms CSV:").grid(row=0, column=3, sticky="w", padx=(0, 4), pady=2)

        self.terms_entry = ttk.Entry(toolbar, textvariable=self.terms_csv_var, width=28)
        self.terms_entry.grid(row=0, column=4, sticky="ew", padx=(0, 4), pady=2)

        self.browse_terms_btn = ttk.Button(toolbar, text="Browse...", width=9, command=self.browse_terms_csv)
        self.browse_terms_btn.grid(row=0, column=5, padx=(0, 4), pady=2)

        # --- Row 1: Mode, Threshold, Load/Reload, Status ---
        row1_frame = ttk.Frame(toolbar)
        row1_frame.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(4, 0))

        ttk.Label(row1_frame, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        self.mode_var = tk.StringVar(value="hybrid")
        self.mode_combo = ttk.Combobox(
            row1_frame,
            textvariable=self.mode_var,
            values=["hybrid", "transformer_only", "entity_ruler_only"],
            state="readonly",
            width=16,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=(0, 14))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(row1_frame, text="Threshold:").pack(side=tk.LEFT, padx=(0, 4))
        self.thresh_var = tk.DoubleVar(value=0.80)
        self.thresh_spin = ttk.Spinbox(
            row1_frame,
            from_=0.50,
            to=0.99,
            increment=0.05,
            textvariable=self.thresh_var,
            width=6,
            format="%.2f",
            command=self._on_threshold_change,
        )
        self.thresh_spin.pack(side=tk.LEFT, padx=(0, 14))

        self.load_model_btn = ttk.Button(
            row1_frame,
            text="Load / Reload Model",
            style="Primary.TButton",
            command=self.load_selected_model,
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=(0, 12))

        self.model_status_label = ttk.Label(
            row1_frame,
            text="● Unloaded",
            foreground="#b91c1c",
            font=("TkDefaultFont", 9, "bold"),
        )
        self.model_status_label.pack(side=tk.LEFT, padx=(0, 12))

        self.device_label = ttk.Label(
            row1_frame,
            text="Hardware: Checking...",
            font=("TkDefaultFont", 8),
            foreground="#475569",
        )
        self.device_label.pack(side=tk.RIGHT)

        toolbar.columnconfigure(1, weight=3)
        toolbar.columnconfigure(4, weight=2)

    def _build_input_pane(self):
        input_frame = ttk.LabelFrame(self.paned, text="Journal Entry Input & Highlighted Visualizer", padding=8)
        self.paned.add(input_frame, weight=3)

        # Control strip above text area
        ctrl_strip = ttk.Frame(input_frame)
        ctrl_strip.pack(fill=tk.X, pady=(0, 6))

        # Presets combobox
        ttk.Label(ctrl_strip, text="Presets:").pack(side=tk.LEFT, padx=(0, 4))
        self.preset_combo_var = tk.StringVar(value="Select Sample...")
        preset_names = ["Select Sample..."] + [p[0] for p in SAMPLE_PRESETS]
        self.preset_combo = ttk.Combobox(
            ctrl_strip,
            textvariable=self.preset_combo_var,
            values=preset_names,
            state="readonly",
            width=28,
        )
        self.preset_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Open file button
        ttk.Button(ctrl_strip, text="Open File...", command=self.open_text_file).pack(side=tk.LEFT, padx=(0, 6))

        # Clear button
        ttk.Button(ctrl_strip, text="Clear", command=self.clear_input).pack(side=tk.LEFT, padx=(0, 6))

        # Primary Extract Button
        self.extract_btn = ttk.Button(
            ctrl_strip,
            text="▶ Extract Entities (Ctrl+Enter)",
            style="Primary.TButton",
            command=self.run_inference,
        )
        self.extract_btn.pack(side=tk.RIGHT, padx=(6, 0))

        # Character / Word count tracker
        self.char_count_label = ttk.Label(
            ctrl_strip,
            text="0 chars | 0 words",
            font=("TkDefaultFont", 8),
            foreground="#64748b",
        )
        self.char_count_label.pack(side=tk.RIGHT, padx=10)

        # Text input box with scrollbar
        text_container = ttk.Frame(input_frame)
        text_container.pack(fill=tk.BOTH, expand=True)

        self.text_box = tk.Text(
            text_container,
            wrap="word",
            font=("DejaVu Sans", 10),
            padx=10,
            pady=8,
            relief=tk.SOLID,
            borderwidth=1,
            undo=True,
        )
        text_scroll = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=self.text_box.yview)
        self.text_box.configure(yscrollcommand=text_scroll.set)

        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_box.bind("<KeyRelease>", self._on_text_change)

        # Configure visual tag styles
        self.text_box.tag_configure(
            "IT_TERM",
            background=COLOR_IT_BG,
            foreground=COLOR_IT_FG,
            font=("DejaVu Sans", 10, "bold"),
            relief=tk.GROOVE,
            borderwidth=1,
        )
        self.text_box.tag_configure(
            "CLERICAL_TERM",
            background=COLOR_CLERICAL_BG,
            foreground=COLOR_CLERICAL_FG,
            font=("DejaVu Sans", 10, "bold"),
            relief=tk.GROOVE,
            borderwidth=1,
        )
        self.text_box.tag_configure(
            "SELECTED_SPAN",
            background=COLOR_SELECT_BG,
            foreground=COLOR_SELECT_FG,
            font=("DejaVu Sans", 10, "bold"),
            underline=True,
        )
        self.text_box.tag_configure(
            "NEEDS_REVIEW_ACCENT",
            background=COLOR_REVIEW_BG,
            foreground=COLOR_REVIEW_FG,
            underline=True,
        )

        # Tag bindings for mouse hover & click
        for tag in ["IT_TERM", "CLERICAL_TERM"]:
            self.text_box.tag_bind(tag, "<Enter>", self._on_tag_hover_enter)
            self.text_box.tag_bind(tag, "<Leave>", self._on_tag_hover_leave)
            self.text_box.tag_bind(tag, "<Button-1>", self._on_tag_click)

        # Legend bar below text area
        legend_frame = ttk.Frame(input_frame)
        legend_frame.pack(fill=tk.X, pady=(4, 0))

        tk.Label(
            legend_frame,
            text="IT_TERM",
            background=COLOR_IT_BG,
            foreground=COLOR_IT_FG,
            font=("TkDefaultFont", 8, "bold"),
            padx=6,
            pady=2,
            relief=tk.SOLID,
            borderwidth=1,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(
            legend_frame,
            text="CLERICAL_TERM",
            background=COLOR_CLERICAL_BG,
            foreground=COLOR_CLERICAL_FG,
            font=("TkDefaultFont", 8, "bold"),
            padx=6,
            pady=2,
            relief=tk.SOLID,
            borderwidth=1,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(
            legend_frame,
            text="NEEDS_REVIEW",
            background=COLOR_REVIEW_BG,
            foreground=COLOR_REVIEW_FG,
            font=("TkDefaultFont", 8, "bold"),
            padx=6,
            pady=2,
            relief=tk.SOLID,
            borderwidth=1,
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(
            legend_frame,
            text="💡 Tip: Click any highlighted span in the text or table row to navigate.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="#64748b",
        ).pack(side=tk.LEFT)

    def _build_output_notebook(self):
        output_frame = ttk.Frame(self.paned)
        self.paned.add(output_frame, weight=4)

        self.notebook = ttk.Notebook(output_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Extracted Entities Table & Analytics
        self.tab_entities = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.tab_entities, text="📊 Extracted Entities & KPI Summary")
        self._build_entities_tab()

        # Tab 2: API Contract JSON Viewer
        self.tab_api_json = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.tab_api_json, text="🔌 API JSON Contract (POST /extract)")
        self._build_api_json_tab()

        # Tab 3: Batch Processing
        self.tab_batch = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(self.tab_batch, text="📁 Batch File Processing")
        self._build_batch_tab()

    def _build_entities_tab(self):
        # KPI Metric Cards Header
        kpi_frame = ttk.LabelFrame(self.tab_entities, text="Extraction Summary & Category Breakdown", padding=6)
        kpi_frame.pack(fill=tk.X, pady=(0, 6))

        # 6 KPI cards in a grid
        self.kpi_total_val = ttk.Label(kpi_frame, text="0", style="StatValue.TLabel")
        self.kpi_total_lbl = ttk.Label(kpi_frame, text="TOTAL OCCURRENCES", style="StatLabel.TLabel")

        self.kpi_unique_val = ttk.Label(kpi_frame, text="0", style="StatValue.TLabel")
        self.kpi_unique_lbl = ttk.Label(kpi_frame, text="UNIQUE ENTITIES", style="StatLabel.TLabel")

        self.kpi_it_val = ttk.Label(kpi_frame, text="0 (0.0%)", style="StatValue.TLabel", foreground=COLOR_IT_FG)
        self.kpi_it_lbl = ttk.Label(kpi_frame, text="IT_TERM COUNT / %", style="StatLabel.TLabel")

        self.kpi_clerical_val = ttk.Label(kpi_frame, text="0 (0.0%)", style="StatValue.TLabel", foreground=COLOR_CLERICAL_FG)
        self.kpi_clerical_lbl = ttk.Label(kpi_frame, text="CLERICAL_TERM COUNT / %", style="StatLabel.TLabel")

        self.kpi_review_val = ttk.Label(kpi_frame, text="0", style="StatValue.TLabel")
        self.kpi_review_lbl = ttk.Label(kpi_frame, text="NEEDS REVIEW (< 0.80)", style="StatLabel.TLabel")

        self.kpi_latency_val = ttk.Label(kpi_frame, text="0 ms", style="StatValue.TLabel", foreground="#0284c7")
        self.kpi_latency_lbl = ttk.Label(kpi_frame, text="INFERENCE LATENCY", style="StatLabel.TLabel")

        cards = [
            (self.kpi_total_val, self.kpi_total_lbl),
            (self.kpi_unique_val, self.kpi_unique_lbl),
            (self.kpi_it_val, self.kpi_it_lbl),
            (self.kpi_clerical_val, self.kpi_clerical_lbl),
            (self.kpi_review_val, self.kpi_review_lbl),
            (self.kpi_latency_val, self.kpi_latency_lbl),
        ]

        for col, (val_lbl, title_lbl) in enumerate(cards):
            card_cell = ttk.Frame(kpi_frame, padding=4)
            card_cell.grid(row=0, column=col, sticky="nsew", padx=4)
            val_lbl.lift()
            val_lbl.pack(in_=card_cell, anchor="center")
            title_lbl.pack(in_=card_cell, anchor="center")
            kpi_frame.columnconfigure(col, weight=1)

        # Filters & Actions bar
        filter_bar = ttk.Frame(self.tab_entities)
        filter_bar.pack(fill=tk.X, pady=(0, 6))

        # Search filter
        ttk.Label(filter_bar, text="Filter:").pack(side=tk.LEFT, padx=(0, 4))
        self.search_filter_var = tk.StringVar()
        self.search_entry = ttk.Entry(filter_bar, textvariable=self.search_filter_var, width=16)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda e: self._apply_table_filters())

        # Category filter
        ttk.Label(filter_bar, text="Category:").pack(side=tk.LEFT, padx=(0, 4))
        self.cat_filter_var = tk.StringVar(value="All")
        self.cat_filter_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.cat_filter_var,
            values=["All", "IT_TERM", "CLERICAL_TERM"],
            state="readonly",
            width=14,
        )
        self.cat_filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.cat_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_table_filters())

        # Status filter
        ttk.Label(filter_bar, text="Status:").pack(side=tk.LEFT, padx=(0, 4))
        self.status_filter_var = tk.StringVar(value="All")
        self.status_filter_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.status_filter_var,
            values=["All", "ACCEPTED", "NEEDS_REVIEW"],
            state="readonly",
            width=14,
        )
        self.status_filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.status_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_table_filters())

        # Source filter
        ttk.Label(filter_bar, text="Source:").pack(side=tk.LEFT, padx=(0, 4))
        self.source_filter_var = tk.StringVar(value="All")
        self.source_filter_combo = ttk.Combobox(
            filter_bar,
            textvariable=self.source_filter_var,
            values=["All", "dictionary", "ML"],
            state="readonly",
            width=12,
        )
        self.source_filter_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.source_filter_combo.bind("<<ComboboxSelected>>", lambda e: self._apply_table_filters())

        # Reset filters button
        ttk.Button(filter_bar, text="Reset Filters", command=self._reset_table_filters).pack(side=tk.LEFT)

        # Table row count
        self.table_count_label = ttk.Label(filter_bar, text="0 entities listed", font=("TkDefaultFont", 8), foreground="#64748b")
        self.table_count_label.pack(side=tk.RIGHT)

        # Entities Treeview Table
        tree_frame = ttk.Frame(self.tab_entities)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("term", "category", "frequency", "confidence", "source", "status", "span")
        self.entity_tree = ttk.Treeview(
            tree_frame,
            columns=cols,
            show="headings",
            selectmode="browse",
        )

        self.entity_tree.heading("term", text="Term / Entity", command=lambda: self._sort_treeview("term", False))
        self.entity_tree.heading("category", text="Category", command=lambda: self._sort_treeview("category", False))
        self.entity_tree.heading("frequency", text="Freq", command=lambda: self._sort_treeview("frequency", True))
        self.entity_tree.heading("confidence", text="Confidence", command=lambda: self._sort_treeview("confidence", True))
        self.entity_tree.heading("source", text="Source", command=lambda: self._sort_treeview("source", False))
        self.entity_tree.heading("status", text="Status", command=lambda: self._sort_treeview("status", False))
        self.entity_tree.heading("span", text="First Span", command=lambda: self._sort_treeview("span", False))

        self.entity_tree.column("term", width=220, anchor="w")
        self.entity_tree.column("category", width=140, anchor="center")
        self.entity_tree.column("frequency", width=60, anchor="center")
        self.entity_tree.column("confidence", width=90, anchor="center")
        self.entity_tree.column("source", width=90, anchor="center")
        self.entity_tree.column("status", width=110, anchor="center")
        self.entity_tree.column("span", width=100, anchor="center")

        tree_scroll_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.entity_tree.yview)
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.entity_tree.xview)
        self.entity_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        self.entity_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.entity_tree.bind("<<TreeviewSelect>>", self._on_tree_row_select)

        # Context menu for table
        self.tree_menu = tk.Menu(self.root, tearoff=0)
        self.tree_menu.add_command(label="Highlight in Text", command=self._jump_to_selected_tree_entity)
        self.tree_menu.add_command(label="Copy Term Text", command=self._copy_selected_term)
        self.tree_menu.add_command(label="Copy Entity JSON", command=self._copy_selected_entity_json)
        self.entity_tree.bind("<Button-3>", self._show_tree_context_menu)

    def _build_api_json_tab(self):
        # Action bar
        act_bar = ttk.Frame(self.tab_api_json)
        act_bar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(
            act_bar,
            text="Exact FastAPI response payload corresponding to POST /extract:",
            font=("TkDefaultFont", 9, "italic"),
            foreground="#475569",
        ).pack(side=tk.LEFT)

        ttk.Button(act_bar, text="📋 Copy JSON", command=self.copy_json_to_clipboard).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(act_bar, text="💾 Save JSON File...", command=self.save_json_file).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(act_bar, text="🔍 Generate cURL", command=self._show_curl_command).pack(side=tk.RIGHT, padx=(4, 0))

        # Text area for JSON view
        json_container = ttk.Frame(self.tab_api_json)
        json_container.pack(fill=tk.BOTH, expand=True)

        self.json_box = tk.Text(
            json_container,
            wrap="none",
            font=("DejaVu Sans Mono", 9),
            padx=8,
            pady=8,
            relief=tk.SOLID,
            borderwidth=1,
            background="#0f172a",
            foreground="#e2e8f0",
            insertbackground="#38bdf8",
        )
        json_scroll_y = ttk.Scrollbar(json_container, orient=tk.VERTICAL, command=self.json_box.yview)
        json_scroll_x = ttk.Scrollbar(json_container, orient=tk.HORIZONTAL, command=self.json_box.xview)
        self.json_box.configure(yscrollcommand=json_scroll_y.set, xscrollcommand=json_scroll_x.set)

        self.json_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        json_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        json_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.json_box.insert("1.0", "{\n  \"status\": \"Run extraction to view API response contract\"\n}")
        self.json_box.config(state="disabled")

    def _build_batch_tab(self):
        batch_ctrl = ttk.Frame(self.tab_batch)
        batch_ctrl.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(batch_ctrl, text="Load JSONL / TXT Batch File...", command=self._load_batch_file).pack(side=tk.LEFT, padx=(0, 6))
        self.run_batch_btn = ttk.Button(batch_ctrl, text="▶ Run Batch Extraction", command=self._run_batch_processing)
        self.run_batch_btn.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(batch_ctrl, text="Export Batch JSONL...", command=self._export_batch_jsonl).pack(side=tk.LEFT, padx=(0, 6))

        self.batch_status_label = ttk.Label(batch_ctrl, text="No batch file loaded.", foreground="#64748b")
        self.batch_status_label.pack(side=tk.RIGHT)

        # Batch progress bar
        self.batch_progress = ttk.Progressbar(self.tab_batch, orient=tk.HORIZONTAL, mode="determinate")
        self.batch_progress.pack(fill=tk.X, pady=(0, 6))

        # Batch records treeview
        batch_tree_frame = ttk.Frame(self.tab_batch)
        batch_tree_frame.pack(fill=tk.BOTH, expand=True)

        batch_cols = ("index", "text_snippet", "entities_count", "it_count", "clerical_count", "has_review")
        self.batch_tree = ttk.Treeview(batch_tree_frame, columns=batch_cols, show="headings", selectmode="browse")

        self.batch_tree.heading("index", text="#")
        self.batch_tree.heading("text_snippet", text="Journal Text Snippet")
        self.batch_tree.heading("entities_count", text="Entities")
        self.batch_tree.heading("it_count", text="IT")
        self.batch_tree.heading("clerical_count", text="Clerical")
        self.batch_tree.heading("has_review", text="Needs Review?")

        self.batch_tree.column("index", width=50, anchor="center")
        self.batch_tree.column("text_snippet", width=480, anchor="w")
        self.batch_tree.column("entities_count", width=70, anchor="center")
        self.batch_tree.column("it_count", width=60, anchor="center")
        self.batch_tree.column("clerical_count", width=70, anchor="center")
        self.batch_tree.column("has_review", width=100, anchor="center")

        b_scroll = ttk.Scrollbar(batch_tree_frame, orient=tk.VERTICAL, command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=b_scroll.set)

        self.batch_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        b_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.batch_tree.bind("<Double-1>", self._on_batch_row_double_click)

    def _build_status_bar(self):
        status_bar = ttk.Frame(self.root, padding=(8, 3))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text_var = tk.StringVar(value="Initializing environment...")
        ttk.Label(status_bar, textvariable=self.status_text_var, relief=tk.SUNKEN, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6)
        )

        self.status_model_var = tk.StringVar(value="Model: None")
        ttk.Label(status_bar, textvariable=self.status_model_var, relief=tk.SUNKEN, width=38, anchor="center").pack(
            side=tk.RIGHT
        )

    # -----------------------------------------------------------------------
    # Environment Detection & Model Population
    # -----------------------------------------------------------------------
    def _detect_environment(self):
        """Check for GPU acceleration."""
        try:
            import spacy
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.detected_device = f"⚡ GPU: {gpu_name}"
            else:
                self.detected_device = "💻 CPU (CUDA unavailable)"
        except Exception:
            self.detected_device = "💻 CPU"

        self.device_label.config(text=f"Hardware: {self.detected_device}")

    def _populate_model_list(self, preferred_path: Optional[str] = None):
        """Discovers models in models/ and populates dropdown."""
        models_found = []
        seen_paths = set()

        # Check default candidates
        for rel_path, desc in DEFAULT_MODEL_CANDIDATES:
            full_path = os.path.join(PROJECT_ROOT, rel_path)
            if os.path.exists(full_path):
                models_found.append((rel_path, f"{desc} [{rel_path}]"))
                seen_paths.add(rel_path)

        # Also scan models/ directory for other trained checkpoints
        models_dir = os.path.join(PROJECT_ROOT, "models")
        if os.path.isdir(models_dir):
            for entry in os.listdir(models_dir):
                candidate_best = os.path.join(models_dir, entry, "model-best")
                rel_candidate = os.path.relpath(candidate_best, PROJECT_ROOT)
                if os.path.exists(candidate_best) and rel_candidate not in seen_paths:
                    models_found.append((rel_candidate, f"Checkpoint: {entry} [{rel_candidate}]"))
                    seen_paths.add(rel_candidate)

        self.model_options = models_found
        display_values = [desc for _, desc in models_found]
        display_values.append("📁 Browse Custom Path...")

        self.model_combo["values"] = display_values

        # Select initial (default to TRSTR-LLM if present, else first available)
        selected_index = 0
        if len(models_found) > 0:
            for idx, (path, _) in enumerate(models_found):
                if "trstr_llm" in path:
                    selected_index = idx
                    break

        if display_values:
            self.model_combo.current(selected_index)
            self._update_model_path_from_combo()

    def _update_model_path_from_combo(self):
        sel = self.model_combo.get()
        for path, desc in self.model_options:
            if desc == sel:
                self.current_model_path = path
                return
        if sel == "📁 Browse Custom Path...":
            self.browse_model_path()

    def _on_model_combo_change(self, event=None):
        sel = self.model_combo.get()
        if sel == "📁 Browse Custom Path...":
            self.browse_model_path()
        else:
            self._update_model_path_from_combo()
            self.load_selected_model()

    def browse_model_path(self):
        chosen_dir = filedialog.askdirectory(
            initialdir=os.path.join(PROJECT_ROOT, "models"),
            title="Select spaCy Model Directory (containing config.cfg or meta.json)"
        )
        if chosen_dir:
            rel = os.path.relpath(chosen_dir, PROJECT_ROOT)
            self.current_model_path = rel
            custom_entry = f"Custom [{rel}]"
            self.model_options.append((rel, custom_entry))
            vals = list(self.model_combo["values"])
            vals.insert(-1, custom_entry)
            self.model_combo["values"] = vals
            self.model_combo.set(custom_entry)
            self.load_selected_model()

    def browse_terms_csv(self):
        chosen_file = filedialog.askopenfilename(
            initialdir=PROJECT_ROOT,
            title="Select Terms Dictionary CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if chosen_file:
            rel = os.path.relpath(chosen_file, PROJECT_ROOT)
            self.terms_csv_var.set(rel)
            self.load_selected_model()

    def _on_mode_change(self, event=None):
        if self.last_result and self.text_box.get("1.0", "end-1c").strip():
            self.run_inference()

    def _on_threshold_change(self):
        if self.pipeline:
            new_thresh = float(self.thresh_var.get())
            self.pipeline.confidence_threshold = new_thresh
            if self.last_result and self.text_box.get("1.0", "end-1c").strip():
                self.run_inference()

    # -----------------------------------------------------------------------
    # Background Model Loader
    # -----------------------------------------------------------------------
    def load_selected_model(self):
        if self.is_loading_model:
            return

        self._update_model_path_from_combo()
        if not self.current_model_path:
            return

        self.is_loading_model = True
        self.model_status_label.config(text="⏳ Loading...", foreground="#d97706")
        self.status_text_var.set(f"Loading model '{self.current_model_path}' in background...")
        self.load_model_btn.config(state="disabled")
        self.extract_btn.config(state="disabled")

        model_path = self.current_model_path
        terms_path = self.terms_csv_var.get().strip() or "data/terms.csv"
        threshold = float(self.thresh_var.get())

        def _worker():
            start_t = time.perf_counter()
            try:
                logger.info(f"Loading HybridJournalPipeline(model_path='{model_path}', terms_csv_path='{terms_path}')")
                new_pipeline = HybridJournalPipeline(
                    model_path=model_path,
                    terms_csv_path=terms_path,
                    confidence_threshold=threshold,
                )
                dur = time.perf_counter() - start_t
                self.work_queue.put(("MODEL_LOAD_SUCCESS", (new_pipeline, model_path, dur)))
            except Exception as e:
                logger.exception("Failed to load model")
                self.work_queue.put(("MODEL_LOAD_FAIL", (model_path, str(e))))

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    # -----------------------------------------------------------------------
    # Queue Dispatcher
    # -----------------------------------------------------------------------
    def _process_queue(self):
        try:
            while True:
                msg_type, payload = self.work_queue.get_nowait()
                if msg_type == "MODEL_LOAD_SUCCESS":
                    new_pipeline, model_path, dur = payload
                    self.pipeline = new_pipeline
                    self.is_loading_model = False
                    self.model_status_label.config(text="● Ready", foreground="#15803d")
                    self.load_model_btn.config(state="normal")
                    self.extract_btn.config(state="normal")
                    self.status_model_var.set(f"Model: {os.path.basename(os.path.dirname(model_path))}")
                    self.status_text_var.set(f"Pipeline ready! Loaded '{model_path}' in {dur:.2f}s ({self.detected_device}).")
                elif msg_type == "MODEL_LOAD_FAIL":
                    model_path, err_msg = payload
                    self.is_loading_model = False
                    self.model_status_label.config(text="● Load Error", foreground="#b91c1c")
                    self.load_model_btn.config(state="normal")
                    self.extract_btn.config(state="normal")
                    self.status_text_var.set(f"Failed to load model '{model_path}'.")
                    messagebox.showerror("Model Load Failure", f"Could not load spaCy pipeline from '{model_path}':\n\n{err_msg}")
                elif msg_type == "INFERENCE_SUCCESS":
                    res, api_payload, dur_ms = payload
                    self.is_inferring = False
                    self.extract_btn.config(state="normal")
                    self._apply_inference_results(res, api_payload, dur_ms)
                elif msg_type == "INFERENCE_FAIL":
                    err_msg = payload
                    self.is_inferring = False
                    self.extract_btn.config(state="normal")
                    self.status_text_var.set("Inference failed.")
                    messagebox.showerror("Extraction Error", f"Inference execution encountered an error:\n\n{err_msg}")
                elif msg_type == "BATCH_PROGRESS":
                    curr, total, snippet, counts = payload
                    self.batch_progress["value"] = (curr / total) * 100
                    self.batch_status_label.config(text=f"Processing {curr}/{total} entries...")
                    self.batch_tree.insert(
                        "",
                        "end",
                        values=(curr, snippet, counts["total"], counts["IT_TERM"], counts["CLERICAL_TERM"], counts["review"])
                    )
                elif msg_type == "BATCH_COMPLETE":
                    total, dur_s = payload
                    self.is_batch_running = False
                    self.run_batch_btn.config(state="normal")
                    self.batch_status_label.config(text=f"Batch complete: {total} entries processed in {dur_s:.2f}s.")
                    messagebox.showinfo("Batch Complete", f"Successfully processed {total} journal records in {dur_s:.2f}s.")
        except queue.Empty:
            pass

        self.root.after(50, self._process_queue)

    # -----------------------------------------------------------------------
    # Inference Execution
    # -----------------------------------------------------------------------
    def run_inference(self):
        """Runs hybrid entity extraction on the current text box content."""
        if self.is_inferring:
            return

        if self.pipeline is None:
            messagebox.showwarning("Model Not Loaded", "Please wait for the model to finish loading before extracting.")
            return

        text = self.text_box.get("1.0", "end-1c")
        if not text or not text.strip():
            messagebox.showinfo("Input Empty", "Please type, paste, or select a sample journal entry first.")
            return

        if len(text) > MAX_TEXT_LENGTH:
            messagebox.showerror(
                "Text Limit Exceeded",
                f"Input has {len(text):,} characters, exceeding the limit of {MAX_TEXT_LENGTH:,} characters."
            )
            return

        mode = self.mode_var.get()
        self.is_inferring = True
        self.extract_btn.config(state="disabled")
        self.status_text_var.set("Running entity extraction inference...")

        def _worker():
            t0 = time.perf_counter()
            try:
                scripts.init_gpu()
                # 1. Pipeline inference (Hybrid / Transformer / EntityRuler)
                raw_result = self.pipeline.predict(text, mode=mode)

                # 2. Deduplicate entities (exact api/main.py logic)
                deduped = _deduplicate_entities(raw_result["entities"])

                # 3. Build category breakdown summary (exact api/main.py logic)
                summary = _build_summary(deduped)

                dur_ms = (time.perf_counter() - t0) * 1000.0

                # Formulate exact API response contract
                api_payload = {
                    "success": True,
                    "content": text,
                    "entities": deduped,
                    "entity_count": len(deduped),
                    "summary": summary,
                }

                self.work_queue.put(("INFERENCE_SUCCESS", (raw_result, api_payload, dur_ms)))
            except Exception as e:
                logger.exception("Inference failed")
                self.work_queue.put(("INFERENCE_FAIL", str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_inference_results(self, raw_result: Dict[str, Any], api_payload: Dict[str, Any], dur_ms: float):
        """Renders inference results across visual text tags, KPI cards, table, and JSON view."""
        self.last_result = raw_result
        self.last_api_response = api_payload
        entities = api_payload["entities"]
        summary = api_payload["summary"]

        # 1. Update KPI Summary Cards
        self.kpi_total_val.config(text=str(summary["total_occurrences"]))
        self.kpi_unique_val.config(text=str(summary["unique_entities"]))
        self.kpi_it_val.config(text=f"{summary['IT_TERM']} ({summary['IT_TERM_percentage']}%)")
        self.kpi_clerical_val.config(text=f"{summary['CLERICAL_TERM']} ({summary['CLERICAL_TERM_percentage']}%)")

        review_count = sum(1 for e in entities if e.get("status") == "NEEDS_REVIEW")
        if review_count > 0:
            self.kpi_review_val.config(text=f"⚠️ {review_count}", foreground="#b45309")
        else:
            self.kpi_review_val.config(text="0", foreground="#15803d")

        self.kpi_latency_val.config(text=f"{dur_ms:.1f} ms")

        # 2. Update Visual Highlighting in Text Box
        self._highlight_text_entities(raw_result["entities"])

        # 3. Update Entities Table
        self.displayed_entities = list(entities)
        self._apply_table_filters()

        # 4. Update JSON Box
        self.json_box.config(state="normal")
        self.json_box.delete("1.0", tk.END)
        self.json_box.insert("1.0", json.dumps(api_payload, indent=2))
        self.json_box.config(state="disabled")

        # 5. Status bar update
        mode_str = raw_result.get("mode", "hybrid")
        self.status_text_var.set(
            f"Extracted {len(entities)} unique entities ({summary['total_occurrences']} total) "
            f"in {dur_ms:.1f}ms [{mode_str} mode]."
        )

    def _highlight_text_entities(self, raw_entities: List[Dict[str, Any]]):
        """Applies colored highlights over extracted spans in the text box."""
        # Clear existing tags
        for tag in ["IT_TERM", "CLERICAL_TERM", "SELECTED_SPAN", "NEEDS_REVIEW_ACCENT"]:
            self.text_box.tag_remove(tag, "1.0", tk.END)

        for ent in raw_entities:
            start = ent["start"]
            end = ent["end"]
            cat = ent["category"]
            status = ent["status"]

            idx_start = f"1.0+{start}c"
            idx_end = f"1.0+{end}c"

            if cat in ["IT_TERM", "CLERICAL_TERM"]:
                self.text_box.tag_add(cat, idx_start, idx_end)

            if status == "NEEDS_REVIEW":
                self.text_box.tag_add("NEEDS_REVIEW_ACCENT", idx_start, idx_end)

    # -----------------------------------------------------------------------
    # Table Filtering & Sorting
    # -----------------------------------------------------------------------
    def _apply_table_filters(self):
        """Filters rows displayed in the Treeview based on user criteria."""
        # Clear treeview
        for item in self.entity_tree.get_children():
            self.entity_tree.delete(item)

        if not self.last_api_response:
            self.table_count_label.config(text="0 entities listed")
            return

        search_q = self.search_filter_var.get().strip().lower()
        cat_q = self.cat_filter_var.get()
        status_q = self.status_filter_var.get()
        source_q = self.source_filter_var.get()

        matching = []
        for ent in self.last_api_response["entities"]:
            # Search filter
            if search_q and search_q not in ent["term"].lower():
                continue
            # Category filter
            if cat_q != "All" and ent["category"] != cat_q:
                continue
            # Status filter
            if status_q != "All" and ent["status"] != status_q:
                continue
            # Source filter
            if source_q != "All" and ent["source"] != source_q:
                continue
            matching.append(ent)

        for ent in matching:
            conf_str = f"{ent['confidence'] * 100.0:.1f}%"
            span_str = f"[{ent['start']}:{ent['end']}]"
            freq_str = f"{ent.get('frequency', 1)}x"

            self.entity_tree.insert(
                "",
                "end",
                values=(
                    ent["term"],
                    ent["category"],
                    freq_str,
                    conf_str,
                    ent["source"],
                    ent["status"],
                    span_str,
                )
            )

        self.table_count_label.config(text=f"{len(matching)} entities listed")

    def _reset_table_filters(self):
        self.search_filter_var.set("")
        self.cat_filter_var.set("All")
        self.status_filter_var.set("All")
        self.source_filter_var.set("All")
        self._apply_table_filters()

    def _sort_treeview(self, col: str, is_numeric: bool):
        """Sorts table rows when clicking on header."""
        items = [
            (self.entity_tree.set(k, col), k)
            for k in self.entity_tree.get_children("")
        ]

        if is_numeric:
            def _num_key(val):
                # Clean up %, x, etc
                clean = re.sub(r"[^\d\.]", "", val[0])
                try:
                    return float(clean)
                except ValueError:
                    return 0.0
            items.sort(key=_num_key, reverse=getattr(self, f"_sort_rev_{col}", False))
        else:
            items.sort(reverse=getattr(self, f"_sort_rev_{col}", False))

        for index, (_, k) in enumerate(items):
            self.entity_tree.move(k, "", index)

        setattr(self, f"_sort_rev_{col}", not getattr(self, f"_sort_rev_{col}", False))

    # -----------------------------------------------------------------------
    # Interactive Navigation & Selection
    # -----------------------------------------------------------------------
    def _on_tree_row_select(self, event):
        selected = self.entity_tree.selection()
        if not selected:
            return
        item_vals = self.entity_tree.item(selected[0], "values")
        term, cat, _, _, _, _, span_str = item_vals

        # Parse span [start:end]
        match = re.match(r"\[(\d+):(\d+)\]", span_str)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            self._select_and_scroll_to_span(start, end)

    def _select_and_scroll_to_span(self, start: int, end: int):
        """Highlights a specific span in the text box and scrolls to it."""
        self.text_box.tag_remove("SELECTED_SPAN", "1.0", tk.END)
        idx_start = f"1.0+{start}c"
        idx_end = f"1.0+{end}c"
        self.text_box.tag_add("SELECTED_SPAN", idx_start, idx_end)
        self.text_box.see(idx_start)
        self.text_box.mark_set(tk.INSERT, idx_start)

    def _on_tag_click(self, event):
        """When clicking an entity tag inside the text box, select its row in the table."""
        idx = self.text_box.index(f"@{event.x},{event.y}")
        # Find offset
        line, char = map(int, idx.split("."))
        text_before = self.text_box.get("1.0", idx)
        offset = len(text_before)

        if not self.last_result:
            return

        for ent in self.last_result["entities"]:
            if ent["start"] <= offset <= ent["end"]:
                # Select in treeview
                for item in self.entity_tree.get_children():
                    vals = self.entity_tree.item(item, "values")
                    if vals[0] == ent["term"] and vals[1] == ent["category"]:
                        self.entity_tree.selection_set(item)
                        self.entity_tree.see(item)
                        break
                break

    def _on_tag_hover_enter(self, event):
        """Displays tooltip showing entity details on hover."""
        idx = self.text_box.index(f"@{event.x},{event.y}")
        text_before = self.text_box.get("1.0", idx)
        offset = len(text_before)

        if not self.last_result:
            return

        for ent in self.last_result["entities"]:
            if ent["start"] <= offset <= ent["end"]:
                conf_pct = f"{ent['confidence'] * 100.0:.1f}%"
                tip_text = (
                    f"Label: {ent['category']}\n"
                    f"Term: {ent['term']}\n"
                    f"Confidence: {conf_pct}\n"
                    f"Source: {ent['source']}\n"
                    f"Status: {ent['status']}\n"
                    f"Span: [{ent['start']}:{ent['end']}]"
                )
                self.tooltip.show(tip_text, event.x_root, event.y_root)
                break

    def _on_tag_hover_leave(self, event):
        self.tooltip.hide()

    def _show_tree_context_menu(self, event):
        row_id = self.entity_tree.identify_row(event.y)
        if row_id:
            self.entity_tree.selection_set(row_id)
            self.tree_menu.tk_popup(event.x_root, event.y_root)

    def _jump_to_selected_tree_entity(self):
        self._on_tree_row_select(None)

    def _copy_selected_term(self):
        selected = self.entity_tree.selection()
        if selected:
            term = self.entity_tree.item(selected[0], "values")[0]
            self.root.clipboard_clear()
            self.root.clipboard_append(term)
            self.status_text_var.set(f"Copied term '{term}' to clipboard.")

    def _copy_selected_entity_json(self):
        selected = self.entity_tree.selection()
        if not selected or not self.last_api_response:
            return
        vals = self.entity_tree.item(selected[0], "values")
        term, cat = vals[0], vals[1]
        for ent in self.last_api_response["entities"]:
            if ent["term"] == term and ent["category"] == cat:
                ent_json = json.dumps(ent, indent=2)
                self.root.clipboard_clear()
                self.root.clipboard_append(ent_json)
                self.status_text_var.set(f"Copied entity JSON for '{term}' to clipboard.")
                break

    # -----------------------------------------------------------------------
    # Preset Samples & Text Editing
    # -----------------------------------------------------------------------
    def _on_preset_selected(self, event=None):
        name = self.preset_combo_var.get()
        for p_name, p_text in SAMPLE_PRESETS:
            if p_name == name:
                self._load_sample_text(p_text)
                break

    def _load_sample_text(self, text: str):
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", text)
        self._on_text_change()
        self.run_inference()

    def _on_text_change(self, event=None):
        content = self.text_box.get("1.0", "end-1c")
        chars = len(content)
        words = len(content.split())
        self.char_count_label.config(text=f"{chars:,} chars | {words:,} words")

    def clear_input(self):
        self.text_box.delete("1.0", tk.END)
        self._on_text_change()
        self.kpi_total_val.config(text="0")
        self.kpi_unique_val.config(text="0")
        self.kpi_it_val.config(text="0 (0.0%)")
        self.kpi_clerical_val.config(text="0 (0.0%)")
        self.kpi_review_val.config(text="0", foreground="#0f172a")
        self.kpi_latency_val.config(text="0 ms")
        self._highlight_text_entities([])
        for item in self.entity_tree.get_children():
            self.entity_tree.delete(item)
        self.table_count_label.config(text="0 entities listed")
        self.json_box.config(state="normal")
        self.json_box.delete("1.0", tk.END)
        self.json_box.insert("1.0", "{\n  \"status\": \"Run extraction to view API response contract\"\n}")
        self.json_box.config(state="disabled")
        self.status_text_var.set("Cleared input text.")

    # -----------------------------------------------------------------------
    # File Operations & Clipboard
    # -----------------------------------------------------------------------
    def open_text_file(self):
        file_path = filedialog.askopenfilename(
            title="Open Journal Text File",
            filetypes=[("Text & JSON Files", "*.txt *.json *.jsonl"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):
                    first_line = f.readline()
                    data = json.loads(first_line)
                    text = data.get("text", "")
                elif file_path.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, dict):
                        text = data.get("text", data.get("content", str(data)))
                    else:
                        text = str(data)
                else:
                    text = f.read()

            self.text_box.delete("1.0", tk.END)
            self.text_box.insert("1.0", text)
            self._on_text_change()
            self.status_text_var.set(f"Loaded file '{os.path.basename(file_path)}' ({len(text):,} chars).")
            self.run_inference()
        except Exception as e:
            messagebox.showerror("File Open Error", f"Could not read file:\n{e}")

    def save_json_file(self):
        if not self.last_api_response:
            messagebox.showwarning("No Data", "Run entity extraction first before saving.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Extracted JSON Result",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.last_api_response, f, indent=2, ensure_ascii=False)
            self.status_text_var.set(f"Saved extraction result to '{os.path.basename(save_path)}'.")
            messagebox.showinfo("Saved Successfully", f"JSON payload saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save file:\n{e}")

    def copy_json_to_clipboard(self):
        if not self.last_api_response:
            messagebox.showwarning("No Data", "Run extraction first.")
            return
        payload_str = json.dumps(self.last_api_response, indent=2)
        self.root.clipboard_clear()
        self.root.clipboard_append(payload_str)
        self.status_text_var.set("Copied full API JSON contract to clipboard.")
        messagebox.showinfo("Copied", "FastAPI-compatible JSON payload copied to clipboard.")

    def _show_curl_command(self):
        text = self.text_box.get("1.0", "end-1c").strip()
        if not text:
            text = "Used Microsoft Excel for data encoding. Configured a REST API using FastAPI."

        escaped_text = json.dumps({"text": text})
        curl_cmd = (
            f"curl -X POST http://localhost:8000/extract \\\n"
            f"  -H \"Content-Type: application/json\" \\\n"
            f"  -d '{escaped_text}'"
        )

        top = tk.Toplevel(self.root)
        top.title("Equivalent cURL Command for /extract API")
        top.geometry("640x300")

        frame = ttk.Frame(top, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="You can run this command directly against the running FastAPI service:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 6))

        t = tk.Text(frame, wrap="word", font=("DejaVu Sans Mono", 9), height=8)
        t.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        t.insert("1.0", curl_cmd)
        t.config(state="disabled")

        btn_bar = ttk.Frame(frame)
        btn_bar.pack(fill=tk.X)

        def _copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(curl_cmd)
            messagebox.showinfo("Copied", "cURL command copied to clipboard.", parent=top)

        ttk.Button(btn_bar, text="Copy cURL Command", command=_copy).pack(side=tk.LEFT)
        ttk.Button(btn_bar, text="Close", command=top.destroy).pack(side=tk.RIGHT)

    # -----------------------------------------------------------------------
    # Batch Processing Implementation
    # -----------------------------------------------------------------------
    def _load_batch_file(self):
        path = filedialog.askopenfilename(
            title="Open Batch File (JSONL or TXT)",
            filetypes=[("Batch Data Files", "*.jsonl *.txt"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.batch_records = []
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if path.endswith(".jsonl"):
                        try:
                            d = json.loads(line)
                            t = d.get("text", "")
                            if t:
                                self.batch_records.append({"text": t, "meta": d})
                        except Exception:
                            pass
                    else:
                        self.batch_records.append({"text": line, "meta": {}})

            self.batch_status_label.config(text=f"Loaded {len(self.batch_records):,} records from '{os.path.basename(path)}'.")
            self.batch_progress["value"] = 0
        except Exception as e:
            messagebox.showerror("Batch Load Error", f"Could not read batch file:\n{e}")

    def _run_batch_processing(self):
        if not self.batch_records:
            messagebox.showwarning("No Batch Loaded", "Please load a batch file first.")
            return

        if self.pipeline is None:
            messagebox.showwarning("Model Not Loaded", "Model is still loading. Please wait.")
            return

        if self.is_batch_running:
            return

        self.is_batch_running = True
        self.run_batch_btn.config(state="disabled")
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        mode = self.mode_var.get()
        records = list(self.batch_records)
        total = len(records)

        def _batch_worker():
            t0 = time.perf_counter()
            scripts.init_gpu()
            for idx, item in enumerate(records):
                text = item["text"]
                try:
                    res = self.pipeline.predict(text, mode=mode)
                    deduped = _deduplicate_entities(res["entities"])
                    summary = _build_summary(deduped)

                    item["extracted"] = deduped
                    item["summary"] = summary

                    snippet = text[:70] + "..." if len(text) > 70 else text
                    has_rev = "⚠️ Yes" if res.get("has_review_items") else "No"

                    counts = {
                        "total": summary["total_occurrences"],
                        "IT_TERM": summary["IT_TERM"],
                        "CLERICAL_TERM": summary["CLERICAL_TERM"],
                        "review": has_rev,
                    }
                    self.work_queue.put(("BATCH_PROGRESS", (idx + 1, total, snippet, counts)))
                except Exception as e:
                    logger.warning(f"Batch item {idx+1} failed: {e}")

            dur_s = time.perf_counter() - t0
            self.work_queue.put(("BATCH_COMPLETE", (total, dur_s)))

        threading.Thread(target=_batch_worker, daemon=True).start()

    def _on_batch_row_double_click(self, event):
        sel = self.batch_tree.selection()
        if not sel:
            return
        vals = self.batch_tree.item(sel[0], "values")
        idx = int(vals[0]) - 1
        if 0 <= idx < len(self.batch_records):
            record = self.batch_records[idx]
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert("1.0", record["text"])
            self._on_text_change()
            self.notebook.select(self.tab_entities)
            self.run_inference()

    def _export_batch_jsonl(self):
        if not self.batch_records or "extracted" not in self.batch_records[0]:
            messagebox.showwarning("No Results", "Run batch extraction first before exporting.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Export Batch Extraction Results",
            defaultextension=".jsonl",
            filetypes=[("JSON Lines", "*.jsonl"), ("All Files", "*.*")]
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for rec in self.batch_records:
                    out = {
                        "text": rec["text"],
                        "entities": rec.get("extracted", []),
                        "summary": rec.get("summary", {}),
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            messagebox.showinfo("Export Successful", f"Exported {len(self.batch_records):,} batch records to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not write file:\n{e}")

    # -----------------------------------------------------------------------
    # Help & Documentation
    # -----------------------------------------------------------------------
    def _show_api_guide(self):
        doc = (
            "FastAPI Service Compatibility\n"
            "------------------------------\n"
            "This GUI tool implements the exact inference, deduplication, and summary logic\n"
            "as the FastAPI service in api/main.py.\n\n"
            "Endpoint: POST /extract\n"
            "Payload:\n"
            "  {\n"
            "    \"text\": \"journal entry text...\"\n"
            "  }\n\n"
            "Response Schema:\n"
            "  - success: bool\n"
            "  - content: string\n"
            "  - entities: list of {\n"
            "      term, category, start, end, confidence, source, status, frequency\n"
            "    }\n"
            "  - entity_count: number of unique entities\n"
            "  - summary: {\n"
            "      total_occurrences, unique_entities,\n"
            "      IT_TERM, CLERICAL_TERM, IT_TERM_percentage, CLERICAL_TERM_percentage\n"
            "    }\n"
        )
        messagebox.showinfo("API Contract Guide", doc)

    def _show_about(self):
        info = (
            "OJT Journal Entity Extraction Studio\n"
            "Version 1.0.0\n\n"
            "Hybrid NER & Classification pipeline combining deterministic dictionary\n"
            "matching with fine-tuned RoBERTa Transformer deep learning.\n\n"
            "Supports: TRSTR-LLM, Baseline, and TRTR models.\n"
            "Location: tools/pipeline_gui.py\n"
        )
        messagebox.showinfo("About", info)


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = PipelineGUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
