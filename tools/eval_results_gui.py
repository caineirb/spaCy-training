#!/usr/bin/env python3
"""
spaCy NER Evaluation Results Studio (Desktop GUI)
=================================================
A native Python/Tkinter graphical user interface to scan, filter, and compare
evaluation results from data/eval_results/ in a human-readable format.

Features:
- Primary Tabs for each type of file:
  * Held-Out Test Set (199 documents)
  * Unseen Benchmark: Hybrid Pipeline (85 documents)
  * Unseen Benchmark: Transformer Only (85 documents)
  * Unseen Benchmark: Entity Ruler Only (85 documents)
  * Dictionary Overrides (18 runtime precedence audit events)
  * 3-Way Benchmark Summary Matrix (TRTR vs TRSTR-Paraphrase vs TRSTR-LLM)
- Model Separation:
  * Switch between TRTR Baseline, TRSTR-Paraphrase, TRSTR-LLM, and Side-by-Side (3-Way)
  * Scope toggle: All Records vs Errors Only
- Rich Interactive Visuals:
  * Color-coded text highlighting for IT_TERM, CLERICAL_TERM, and Error types
  * Comprehensive entity comparison table (Gold vs Predicted, Confidence, Source, Offsets)
  * Live filter search and status filter chips
  * KPI summary cards
  * 1-Click Clipboard copy for text and raw JSON

Usage:
    python tools/eval_results_gui.py
"""

import os
import sys
import re
import json
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Optional, Tuple

# Project root path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results")
DEFAULT_REPORT_FILE = os.path.join(PROJECT_ROOT, "data", "evaluation_report_3way_comparison.json")


# ---------------------------------------------------------------------------
# Data Loading & Utility Functions
# ---------------------------------------------------------------------------
def find_offsets(text: str, term: str, near_offset: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """Find start and end character offsets of a term in text."""
    if not term or not text:
        return None, None
    matches = [m.span() for m in re.finditer(re.escape(term), text)]
    if not matches:
        matches = [m.span() for m in re.finditer(re.escape(term), text, re.IGNORECASE)]
    if not matches:
        return None, None
    if near_offset is not None and len(matches) > 1:
        matches.sort(key=lambda s: abs(s[0] - near_offset))
    return matches[0]


def explain_boundary(text: str, gs: Optional[int], ge: Optional[int], ps: Optional[int], pe: Optional[int]) -> str:
    """Analyze character-level difference between gold target boundary and predicted boundary."""
    if gs is None and ps is None:
        return "—"
    if gs is None:
        return "Spurious Prediction (No gold target span)"
    if ps is None:
        return "Missed Gold Entity (No prediction)"
    if gs == ps and ge == pe:
        return "Exact Boundary Match (0 char diff)"

    parts = []
    if ps is not None and gs is not None:
        if ps > gs:
            diff_txt = text[gs:ps]
            parts.append(f"Missed prefix '{diff_txt}' (-{ps - gs} chars)")
        elif ps < gs:
            diff_txt = text[ps:gs]
            parts.append(f"Added extra prefix '{diff_txt}' (+{gs - ps} chars)")

    if pe is not None and ge is not None:
        if pe < ge:
            diff_txt = text[pe:ge]
            parts.append(f"Missed suffix '{diff_txt}' (-{ge - pe} chars)")
        elif pe > ge:
            diff_txt = text[ge:pe]
            parts.append(f"Added extra suffix '{diff_txt}' (+{pe - ge} chars)")

    return " | ".join(parts) if parts else f"Offset mismatch [{gs}:{ge}] vs [{ps}:{pe}]"



def load_all_eval_data(eval_dir: str = DEFAULT_EVAL_DIR, report_file: str = DEFAULT_REPORT_FILE) -> Dict[str, Any]:
    """Scan and parse all evaluation result files."""
    data: Dict[str, Any] = {
        "models": ["trtr", "trstr_paraphrase", "trstr_llm"],
        "file_types": [
            ("held_out_test", "Held-Out Test Set (199 docs)"),
            ("unseen_benchmark_hybrid", "Unseen Benchmark: Hybrid (85 docs)"),
            ("unseen_benchmark_transformer_only", "Unseen Benchmark: Transformer Only (85 docs)"),
            ("unseen_benchmark_entity_ruler_only", "Unseen Benchmark: Entity Ruler Only (85 docs)"),
        ],
        "datasets": {},
        "dictionary_overrides": [],
        "report_summary": None,
    }

    # Load 3-way evaluation report
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as rf:
                data["report_summary"] = json.load(rf)
        except Exception as e:
            print(f"[WARN] Failed to load {report_file}: {e}")

    # Load dictionary overrides
    dict_file = os.path.join(eval_dir, "dictionary_overrides.jsonl")
    if os.path.exists(dict_file):
        try:
            with open(dict_file, "r", encoding="utf-8") as df:
                for line in df:
                    line = line.strip()
                    if line:
                        data["dictionary_overrides"].append(json.loads(line))
        except Exception as e:
            print(f"[WARN] Failed to load dictionary overrides: {e}")

    # Load comparison files
    for ft_id, _ in data["file_types"]:
        data["datasets"][ft_id] = {}
        for m in data["models"]:
            full_path = os.path.join(eval_dir, "comparison", m, f"{ft_id}_{m}.jsonl")
            err_path = os.path.join(eval_dir, "comparison", m, f"{ft_id}_{m}_errors_only.jsonl")

            all_records = []
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            row = json.loads(line)
                            _enrich_offsets(row)
                            all_records.append(row)

            err_records = []
            if os.path.exists(err_path):
                with open(err_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            row = json.loads(line)
                            _enrich_offsets(row)
                            err_records.append(row)

            data["datasets"][ft_id][m] = {
                "all": all_records,
                "errors_only": err_records,
            }

    return data


def _enrich_offsets(row: Dict[str, Any]):
    """Ensure prediction spans have accurate start and end character offsets."""
    text = row.get("text", "")
    for er in row.get("entity_results", []):
        if er.get("pred_term") and ("pred_start" not in er or er["pred_start"] is None):
            s, e = find_offsets(text, er["pred_term"], near_offset=er.get("gold_start"))
            er["pred_start"] = s
            er["pred_end"] = e


# ---------------------------------------------------------------------------
# Main GUI Window Class
# ---------------------------------------------------------------------------
class EvalResultsGUIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("spaCy NER Evaluation Results Studio")
        self.root.geometry("1400x920")
        self.root.minsize(1100, 720)

        # Application state
        self.data = load_all_eval_data()
        self.active_file_type = "held_out_test"
        self.active_model = "trtr"  # 'trtr' | 'trstr_paraphrase' | 'trstr_llm' | 'compare'
        self.active_scope = "all"   # 'all' | 'errors_only'
        self.search_query = ""
        self.status_filter = "All"
        self.label_filter = "All"
        self.source_filter = "All"
        self.sbs_filter = "All"

        # Cached filtered records for active view
        self.current_filtered_records: List[Tuple[int, Dict[str, Any]]] = []
        self.selected_doc_idx: Optional[int] = None

        # Build GUI
        self._init_styles()
        self._build_header()
        self._build_tabs()
        self._build_status_bar()

        # Initial view population
        self._refresh_active_view()

    # -----------------------------------------------------------------------
    # Styles & Colors
    # -----------------------------------------------------------------------
    def _init_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Palette
        self.c_bg = "#f8fafc"
        self.c_header = "#0f172a"
        self.c_it_bg = "#dbeafe"
        self.c_it_fg = "#1e40af"
        self.c_clerical_bg = "#d1fae5"
        self.c_clerical_fg = "#065f46"
        self.c_boundary_bg = "#fef3c7"
        self.c_boundary_fg = "#92400e"
        self.c_error_bg = "#ffe4e6"
        self.c_error_fg = "#9f1239"
        self.c_spurious_bg = "#ffedd5"
        self.c_spurious_fg = "#9a3412"
        self.c_gold_bg = "#bbf7d0"
        self.c_gold_fg = "#14532d"
        self.c_focus_bg = "#fef08a"
        self.c_focus_fg = "#854d0e"

        # Fonts
        self.f_title = ("TkDefaultFont", 12, "bold")
        self.f_header = ("TkDefaultFont", 10, "bold")
        self.f_bold = ("TkDefaultFont", 9, "bold")
        self.f_normal = ("TkDefaultFont", 9)
        self.f_mono = ("Courier", 9)

        # Treeview styling
        self.style.configure("Treeview", font=self.f_normal, rowheight=24)
        self.style.configure("Treeview.Heading", font=self.f_bold)
        self.style.configure("TNotebook.Tab", font=self.f_bold, padding=[12, 6])
        self.style.configure("Action.TButton", font=self.f_bold)

    # -----------------------------------------------------------------------
    # Top Header
    # -----------------------------------------------------------------------
    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=self.c_header, padx=16, pady=12)
        header_frame.pack(fill=tk.X)

        title_box = tk.Frame(header_frame, bg=self.c_header)
        title_box.pack(side=tk.LEFT)

        lbl_title = tk.Label(
            title_box,
            text="spaCy NER Evaluation Results Studio",
            font=("TkDefaultFont", 14, "bold"),
            fg="#f8fafc",
            bg=self.c_header,
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            title_box,
            text="Interactive scanner for held-out validation, out-of-domain benchmarks & dictionary overrides",
            font=("TkDefaultFont", 9),
            fg="#94a3b8",
            bg=self.c_header,
        )
        lbl_sub.pack(anchor="w")

        # Stats pills on the right
        stats_box = tk.Frame(header_frame, bg=self.c_header)
        stats_box.pack(side=tk.RIGHT)

        dict_count = len(self.data["dictionary_overrides"])
        held_out_count = len(self.data["datasets"].get("held_out_test", {}).get("trtr", {}).get("all", []))
        unseen_count = len(self.data["datasets"].get("unseen_benchmark_hybrid", {}).get("trtr", {}).get("all", []))

        pill_text = f"Held-Out: {held_out_count}  |  Unseen Benchmark: {unseen_count}  |  Overrides: {dict_count}"
        tk.Label(
            stats_box,
            text=pill_text,
            font=self.f_bold,
            fg="#38bdf8",
            bg="#1e293b",
            padx=12,
            pady=4,
            relief="groove",
        ).pack(side=tk.LEFT, padx=8)

        btn_reload = tk.Button(
            stats_box,
            text="🔄 Reload Files",
            font=self.f_bold,
            command=self.reload_data,
            bg="#3b82f6",
            fg="white",
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
        )
        btn_reload.pack(side=tk.LEFT)

    # -----------------------------------------------------------------------
    # Main Tabs (Notebook)
    # -----------------------------------------------------------------------
    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # Tab 1: Held-Out Test Set
        self.tab_held_out = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_held_out, text="📄 Held-Out Test Set (199 docs)")

        # Tab 2: Unseen Benchmark Hybrid
        self.tab_unseen_hyb = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_unseen_hyb, text="⚡ Unseen: Hybrid (85 docs)")

        # Tab 3: Unseen Benchmark Transformer
        self.tab_unseen_trf = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_unseen_trf, text="🤖 Unseen: Transformer (85 docs)")

        # Tab 4: Unseen Benchmark Entity Ruler
        self.tab_unseen_ruler = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_unseen_ruler, text="📖 Unseen: Entity Ruler (85 docs)")

        # Tab 5: Dictionary Overrides
        self.tab_dict = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dict, text="🔄 Dictionary Overrides (18 records)")

        # Tab 6: 3-Way Benchmark Summary Matrix
        self.tab_matrix = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_matrix, text="📊 3-Way Benchmark Matrix")

        # Map tab frames to file_type id
        self.tab_map = {
            self.tab_held_out: "held_out_test",
            self.tab_unseen_hyb: "unseen_benchmark_hybrid",
            self.tab_unseen_trf: "unseen_benchmark_transformer_only",
            self.tab_unseen_ruler: "unseen_benchmark_entity_ruler_only",
            self.tab_dict: "dictionary_overrides",
            self.tab_matrix: "matrix_summary",
        }

        # Build layouts for evaluation tabs (Tabs 1-4 share identical layout)
        self.eval_tab_widgets = {}
        for tab_frame, ft_id in list(self.tab_map.items())[:4]:
            self.eval_tab_widgets[ft_id] = self._build_evaluation_tab_layout(tab_frame, ft_id)

        # Build layout for Dictionary Overrides
        self._build_dict_overrides_layout(self.tab_dict)

        # Build layout for 3-Way Summary Matrix
        self._build_matrix_layout(self.tab_matrix)

        # Listen for tab changes
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    # -----------------------------------------------------------------------
    # Evaluation Tab Layout (Shared by Tabs 1 to 4)
    # -----------------------------------------------------------------------
    def _build_evaluation_tab_layout(self, parent: ttk.Frame, file_type: str) -> Dict[str, Any]:
        container = ttk.Frame(parent, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        # Top Control Bar (Model pills + Scope + Filters)
        ctrl_frame = ttk.LabelFrame(container, text="Model Separation & Filters", padding=8)
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        # Model Radio Buttons
        model_box = ttk.Frame(ctrl_frame)
        model_box.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(model_box, text="Model Condition:", font=self.f_bold).pack(side=tk.LEFT, padx=(0, 8))
        model_var = tk.StringVar(value=self.active_model)

        r_trtr = ttk.Radiobutton(
            model_box, text="🔵 TRTR Baseline", value="trtr", variable=model_var,
            command=lambda: self._on_model_changed(file_type, model_var.get())
        )
        r_trtr.pack(side=tk.LEFT, padx=6)

        r_para = ttk.Radiobutton(
            model_box, text="🟣 TRSTR-Paraphrase", value="trstr_paraphrase", variable=model_var,
            command=lambda: self._on_model_changed(file_type, model_var.get())
        )
        r_para.pack(side=tk.LEFT, padx=6)

        r_llm = ttk.Radiobutton(
            model_box, text="🟢 TRSTR-LLM (High Recall)", value="trstr_llm", variable=model_var,
            command=lambda: self._on_model_changed(file_type, model_var.get())
        )
        r_llm.pack(side=tk.LEFT, padx=6)

        r_sbs = ttk.Radiobutton(
            model_box, text="⚖️ Side-by-Side (Compare All 3)", value="compare", variable=model_var,
            command=lambda: self._on_model_changed(file_type, model_var.get())
        )
        r_sbs.pack(side=tk.LEFT, padx=6)

        # Scope selector: All Records vs Errors Only
        scope_box = ttk.Frame(model_box)
        scope_box.pack(side=tk.RIGHT)
        scope_var = tk.StringVar(value=self.active_scope)

        r_scope_all = ttk.Radiobutton(
            scope_box, text="All Records", value="all", variable=scope_var,
            command=lambda: self._on_scope_changed(file_type, scope_var.get())
        )
        r_scope_all.pack(side=tk.LEFT, padx=4)

        r_scope_err = ttk.Radiobutton(
            scope_box, text="⚠️ Errors Only", value="errors_only", variable=scope_var,
            command=lambda: self._on_scope_changed(file_type, scope_var.get())
        )
        r_scope_err.pack(side=tk.LEFT, padx=4)

        # Filter bar: Search, Status, Label, Source
        filter_bar = ttk.Frame(ctrl_frame)
        filter_bar.pack(fill=tk.X)

        ttk.Label(filter_bar, text="Search:").pack(side=tk.LEFT, padx=(0, 4))
        search_var = tk.StringVar()
        search_ent = ttk.Entry(filter_bar, textvariable=search_var, width=22)
        search_ent.pack(side=tk.LEFT, padx=(0, 10))
        search_ent.bind("<KeyRelease>", lambda e: self._on_filter_changed(file_type))

        ttk.Label(filter_bar, text="Status:").pack(side=tk.LEFT, padx=(0, 4))
        status_var = tk.StringVar(value="All")
        status_combo = ttk.Combobox(
            filter_bar, textvariable=status_var, state="readonly", width=16,
            values=["All", "correct", "boundary_error", "false_negative", "false_positive", "label_error"]
        )
        status_combo.pack(side=tk.LEFT, padx=(0, 10))
        status_combo.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed(file_type))

        ttk.Label(filter_bar, text="Label:").pack(side=tk.LEFT, padx=(0, 4))
        label_var = tk.StringVar(value="All")
        label_combo = ttk.Combobox(
            filter_bar, textvariable=label_var, state="readonly", width=14,
            values=["All", "IT_TERM", "CLERICAL_TERM"]
        )
        label_combo.pack(side=tk.LEFT, padx=(0, 10))
        label_combo.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed(file_type))

        ttk.Label(filter_bar, text="Source:").pack(side=tk.LEFT, padx=(0, 4))
        source_var = tk.StringVar(value="All")
        source_combo = ttk.Combobox(
            filter_bar, textvariable=source_var, state="readonly", width=12,
            values=["All", "ML", "dictionary"]
        )
        source_combo.pack(side=tk.LEFT, padx=(0, 10))
        source_combo.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed(file_type))

        btn_reset = ttk.Button(
            filter_bar, text="Reset Filters",
            command=lambda: self._reset_filters(file_type, search_var, status_var, label_var, source_var)
        )
        btn_reset.pack(side=tk.LEFT, padx=4)

        # KPI Summary Ribbon
        kpi_frame = ttk.Frame(container, padding=4)
        kpi_frame.pack(fill=tk.X, pady=(0, 6))

        kpi_lbl = tk.Label(
            kpi_frame,
            text="Loading summary metrics...",
            font=self.f_bold,
            fg="#0369a1",
            bg="#f0f9ff",
            relief="groove",
            padx=10,
            pady=6,
            anchor="w",
        )
        kpi_lbl.pack(fill=tk.X)

        # Paned Window (Split: Document list on left, Inspector on right)
        paned = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # LEFT PANE: Document list
        left_frame = ttk.Frame(paned, padding=4)
        paned.add(left_frame, weight=3)

        doc_tree_frame = ttk.Frame(left_frame)
        doc_tree_frame.pack(fill=tk.BOTH, expand=True)

        doc_scroll = ttk.Scrollbar(doc_tree_frame)
        doc_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        doc_tree = ttk.Treeview(
            doc_tree_frame,
            columns=("idx", "status", "g_count", "p_count", "snippet"),
            show="headings",
            selectmode="browse",
            yscrollcommand=doc_scroll.set,
        )
        doc_scroll.config(command=doc_tree.yview)

        doc_tree.heading("idx", text="#", anchor="center")
        doc_tree.heading("status", text="Status", anchor="w")
        doc_tree.heading("g_count", text="Gold", anchor="center")
        doc_tree.heading("p_count", text="Pred", anchor="center")
        doc_tree.heading("snippet", text="Sentence Snippet", anchor="w")

        doc_tree.column("idx", width=45, anchor="center")
        doc_tree.column("status", width=120, anchor="w")
        doc_tree.column("g_count", width=45, anchor="center")
        doc_tree.column("p_count", width=45, anchor="center")
        doc_tree.column("snippet", width=260, anchor="w")

        doc_tree.pack(fill=tk.BOTH, expand=True)
        doc_tree.bind("<<TreeviewSelect>>", lambda e: self._on_doc_selected(file_type))

        # Doc tree tags for status colors
        doc_tree.tag_configure("all_correct", background="#f0fdf4")
        doc_tree.tag_configure("has_error", background="#fffbeb")

        # Quick navigation buttons under list
        nav_box = ttk.Frame(left_frame)
        nav_box.pack(fill=tk.X, pady=(4, 0))

        btn_prev_err = ttk.Button(nav_box, text="⏮ Prev Error", command=lambda: self._jump_error(file_type, -1))
        btn_prev_err.pack(side=tk.LEFT, padx=2)

        btn_next_err = ttk.Button(nav_box, text="Next Error ⏭", command=lambda: self._jump_error(file_type, 1))
        btn_next_err.pack(side=tk.LEFT, padx=2)

        lbl_doc_count = ttk.Label(nav_box, text="0 documents", font=self.f_normal)
        lbl_doc_count.pack(side=tk.RIGHT, padx=4)

        # RIGHT PANE: Human-Readable Inspection View
        right_frame = ttk.Frame(paned, padding=4)
        paned.add(right_frame, weight=5)

        # 1. Standard Single Model Inspector Container
        single_inspect_frame = ttk.Frame(right_frame)
        single_inspect_frame.pack(fill=tk.BOTH, expand=True)

        # Highlighting View Mode Control Bar
        view_ctrl_frame = ttk.Frame(single_inspect_frame)
        view_ctrl_frame.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(view_ctrl_frame, text="Sentence Boundary View:", font=self.f_bold).pack(side=tk.LEFT, padx=(0, 6))
        view_mode_var = tk.StringVar(value="overlay")

        r_v_overlay = ttk.Radiobutton(
            view_ctrl_frame, text="👁️ Dual Boundary Comparison", value="overlay", variable=view_mode_var,
            command=lambda: self._on_view_mode_changed(file_type)
        )
        r_v_overlay.pack(side=tk.LEFT, padx=4)

        r_v_gold = ttk.Radiobutton(
            view_ctrl_frame, text="🎯 Correct Gold Boundaries Only", value="gold", variable=view_mode_var,
            command=lambda: self._on_view_mode_changed(file_type)
        )
        r_v_gold.pack(side=tk.LEFT, padx=4)

        r_v_pred = ttk.Radiobutton(
            view_ctrl_frame, text="🤖 Model Predicted Boundaries Only", value="pred", variable=view_mode_var,
            command=lambda: self._on_view_mode_changed(file_type)
        )
        r_v_pred.pack(side=tk.LEFT, padx=4)

        # Sentence Box with highlighted tags
        txt_sentence = tk.Text(
            single_inspect_frame,
            wrap=tk.WORD,
            font=("TkDefaultFont", 11),
            height=5,
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=8,
        )
        txt_sentence.pack(fill=tk.X, pady=(0, 6))

        # Configure Text highlighting tags
        txt_sentence.tag_configure("it_term", background=self.c_it_bg, foreground=self.c_it_fg, font=self.f_bold)
        txt_sentence.tag_configure("clerical_term", background=self.c_clerical_bg, foreground=self.c_clerical_fg, font=self.f_bold)
        txt_sentence.tag_configure("gold_boundary", background=self.c_gold_bg, foreground=self.c_gold_fg, font=self.f_bold, underline=True)
        txt_sentence.tag_configure("pred_boundary", background=self.c_boundary_bg, foreground=self.c_boundary_fg, font=self.f_bold)
        txt_sentence.tag_configure("boundary_pred", background=self.c_boundary_bg, foreground=self.c_boundary_fg, underline=True)
        txt_sentence.tag_configure("boundary_gold", background=self.c_gold_bg, foreground=self.c_gold_fg, font=self.f_bold)
        txt_sentence.tag_configure("missed", background=self.c_error_bg, foreground=self.c_error_fg, overstrike=True)
        txt_sentence.tag_configure("spurious", background=self.c_spurious_bg, foreground=self.c_spurious_fg, underline=True)
        txt_sentence.tag_configure("active_focus", background=self.c_focus_bg, foreground=self.c_focus_fg, font=self.f_bold, relief="solid", borderwidth=1)

        # Legend frame
        legend_frame = ttk.Frame(single_inspect_frame)
        legend_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Label(legend_frame, text="Legend:", font=self.f_bold, bg="#f1f5f9").pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(legend_frame, text="🎯 Correct Gold", bg=self.c_gold_bg, fg=self.c_gold_fg, font=self.f_bold, padx=4).pack(side=tk.LEFT, padx=3)
        tk.Label(legend_frame, text="⚠️ Boundary Error (Pred)", bg=self.c_boundary_bg, fg=self.c_boundary_fg, font=self.f_bold, padx=4).pack(side=tk.LEFT, padx=3)
        tk.Label(legend_frame, text="IT_TERM", bg=self.c_it_bg, fg=self.c_it_fg, font=self.f_bold, padx=4).pack(side=tk.LEFT, padx=3)
        tk.Label(legend_frame, text="CLERICAL_TERM", bg=self.c_clerical_bg, fg=self.c_clerical_fg, font=self.f_bold, padx=4).pack(side=tk.LEFT, padx=3)
        tk.Label(legend_frame, text="❌ Missed Gold", bg=self.c_error_bg, fg=self.c_error_fg, font=self.f_bold, padx=4).pack(side=tk.LEFT, padx=3)
        tk.Label(legend_frame, text="🚫 Spurious Pred", bg=self.c_spurious_bg, fg=self.c_spurious_fg, font=self.f_bold, padx=4).pack(side=tk.LEFT, padx=3)

        # Dedicated Boundary Error & Token Span Inspector Panel
        boundary_card = tk.LabelFrame(
            single_inspect_frame,
            text="🔍 Boundary Error & Token Span Inspector",
            font=self.f_bold,
            padx=10,
            pady=6,
            relief="groove",
        )
        boundary_card.pack(fill=tk.X, pady=(0, 6))

        lbl_boundary_detail = tk.Label(
            boundary_card,
            text="Select an entity in the table below to inspect exact character offsets and boundary differences.",
            font=self.f_normal,
            fg="#1e293b",
            justify=tk.LEFT,
            anchor="w",
        )
        lbl_boundary_detail.pack(fill=tk.X)

        # Entity Comparison Table
        lbl_tbl_title = ttk.Label(single_inspect_frame, text="Entity-by-Entity Comparison Breakdown (Dual Boundaries):", font=self.f_bold)
        lbl_tbl_title.pack(anchor="w", pady=(2, 2))

        ent_tree_frame = ttk.Frame(single_inspect_frame)
        ent_tree_frame.pack(fill=tk.BOTH, expand=True)

        ent_scroll = ttk.Scrollbar(ent_tree_frame)
        ent_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ent_tree = ttk.Treeview(
            ent_tree_frame,
            columns=("gold_term", "pred_term", "status", "diff", "gold_lbl", "pred_lbl", "conf", "source"),
            show="headings",
            selectmode="browse",
            yscrollcommand=ent_scroll.set,
        )
        ent_scroll.config(command=ent_tree.yview)

        ent_tree.heading("gold_term", text="🎯 Correct Gold Boundary", anchor="w")
        ent_tree.heading("pred_term", text="🤖 Model Predicted Span", anchor="w")
        ent_tree.heading("status", text="Status", anchor="w")
        ent_tree.heading("diff", text="Boundary Difference / Token Mismatch", anchor="w")
        ent_tree.heading("gold_lbl", text="Gold Label", anchor="center")
        ent_tree.heading("pred_lbl", text="Pred Label", anchor="center")
        ent_tree.heading("conf", text="Confidence", anchor="center")
        ent_tree.heading("source", text="Source", anchor="center")

        ent_tree.column("gold_term", width=175, anchor="w")
        ent_tree.column("pred_term", width=175, anchor="w")
        ent_tree.column("status", width=125, anchor="w")
        ent_tree.column("diff", width=230, anchor="w")
        ent_tree.column("gold_lbl", width=95, anchor="center")
        ent_tree.column("pred_lbl", width=95, anchor="center")
        ent_tree.column("conf", width=75, anchor="center")
        ent_tree.column("source", width=75, anchor="center")

        ent_tree.pack(fill=tk.BOTH, expand=True)
        ent_tree.bind("<<TreeviewSelect>>", lambda e: self._on_entity_selected(file_type))

        # Entity Tree row colors
        ent_tree.tag_configure("correct", background="#f0fdf4")
        ent_tree.tag_configure("boundary_error", background="#fef9c3", font=self.f_bold)
        ent_tree.tag_configure("false_negative", background="#fef2f2")
        ent_tree.tag_configure("false_positive", background="#fff7ed")
        ent_tree.tag_configure("label_error", background="#fdf2f8")

        # Action Buttons bar at bottom of right frame
        action_bar = ttk.Frame(single_inspect_frame, padding=4)
        action_bar.pack(fill=tk.X, pady=(4, 0))

        btn_copy_text = ttk.Button(action_bar, text="📋 Copy Sentence", command=lambda: self._copy_sentence(file_type))
        btn_copy_text.pack(side=tk.LEFT, padx=4)

        btn_copy_json = ttk.Button(action_bar, text="📋 Copy Record JSON", command=lambda: self._copy_json(file_type))
        btn_copy_json.pack(side=tk.LEFT, padx=4)

        # 2. SIDE-BY-SIDE INSPECTOR CONTAINER (Shown when active_model == 'compare')
        sbs_inspect_frame = ttk.Frame(right_frame)
        # Built dynamically in _render_sbs_details

        return {
            "model_var": model_var,
            "scope_var": scope_var,
            "search_var": search_var,
            "status_var": status_var,
            "label_var": label_var,
            "source_var": source_var,
            "view_mode_var": view_mode_var,
            "lbl_boundary_detail": lbl_boundary_detail,
            "boundary_card": boundary_card,
            "kpi_lbl": kpi_lbl,
            "doc_tree": doc_tree,
            "lbl_doc_count": lbl_doc_count,
            "txt_sentence": txt_sentence,
            "ent_tree": ent_tree,
            "single_inspect_frame": single_inspect_frame,
            "sbs_inspect_frame": sbs_inspect_frame,
            "right_frame": right_frame,
        }

    # -----------------------------------------------------------------------
    # Dictionary Overrides Layout (Tab 5)
    # -----------------------------------------------------------------------
    def _build_dict_overrides_layout(self, parent: ttk.Frame):
        container = ttk.Frame(parent, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # Header Info Banner
        banner = tk.Label(
            container,
            text="Dictionary Overrides Audit Trail (18 Events)\n"
                 "These records show sentences where the runtime EntityRuler dictionary (data/terms.csv) "
                 "overruled conflicting machine learning predictions.",
            font=self.f_bold,
            fg="#047857",
            bg="#ecfdf5",
            relief="groove",
            padx=12,
            pady=8,
            justify=tk.LEFT,
            anchor="w",
        )
        banner.pack(fill=tk.X, pady=(0, 8))

        # Search bar for overrides
        filter_box = ttk.Frame(container)
        filter_box.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(filter_box, text="Filter Overrides:").pack(side=tk.LEFT, padx=(0, 6))
        self.dict_search_var = tk.StringVar()
        ent_search = ttk.Entry(filter_box, textvariable=self.dict_search_var, width=28)
        ent_search.pack(side=tk.LEFT, padx=(0, 8))
        ent_search.bind("<KeyRelease>", lambda e: self._filter_dict_overrides())

        self.lbl_dict_count = ttk.Label(filter_box, text="Showing 18 of 18 override events", font=self.f_bold)
        self.lbl_dict_count.pack(side=tk.RIGHT, padx=4)

        # Paned Window for Overrides
        paned = ttk.PanedWindow(container, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Treeview table
        tree_frame = ttk.Frame(paned)
        paned.add(tree_frame, weight=3)

        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.dict_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "term", "ml_lbl", "dict_lbl", "win_lbl", "offsets", "sentence"),
            show="headings",
            selectmode="browse",
            yscrollcommand=tree_scroll.set,
        )
        tree_scroll.config(command=self.dict_tree.yview)

        self.dict_tree.heading("id", text="#", anchor="center")
        self.dict_tree.heading("term", text="Overridden Term", anchor="w")
        self.dict_tree.heading("ml_lbl", text="ML Label (Lost)", anchor="center")
        self.dict_tree.heading("dict_lbl", text="Dict Label", anchor="center")
        self.dict_tree.heading("win_lbl", text="Winning Label (Precedence)", anchor="center")
        self.dict_tree.heading("offsets", text="Offsets", anchor="center")
        self.dict_tree.heading("sentence", text="Context Sentence", anchor="w")

        self.dict_tree.column("id", width=40, anchor="center")
        self.dict_tree.column("term", width=160, anchor="w")
        self.dict_tree.column("ml_lbl", width=130, anchor="center")
        self.dict_tree.column("dict_lbl", width=130, anchor="center")
        self.dict_tree.column("win_lbl", width=150, anchor="center")
        self.dict_tree.column("offsets", width=80, anchor="center")
        self.dict_tree.column("sentence", width=450, anchor="w")

        self.dict_tree.pack(fill=tk.BOTH, expand=True)
        self.dict_tree.bind("<<TreeviewSelect>>", self._on_dict_row_selected)

        # Detail Box
        detail_frame = ttk.LabelFrame(paned, text="Override Event Details", padding=8)
        paned.add(detail_frame, weight=2)

        self.txt_dict_detail = tk.Text(
            detail_frame,
            wrap=tk.WORD,
            font=("TkDefaultFont", 11),
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=8,
            height=5,
        )
        self.txt_dict_detail.pack(fill=tk.BOTH, expand=True)
        self.txt_dict_detail.tag_configure("highlight", background="#fef08a", foreground="#854d0e", font=self.f_bold)
        self.txt_dict_detail.tag_configure("winner", background="#d1fae5", foreground="#065f46", font=self.f_bold)

    # -----------------------------------------------------------------------
    # 3-Way Benchmark Summary Matrix Layout (Tab 6)
    # -----------------------------------------------------------------------
    def _build_matrix_layout(self, parent: ttk.Frame):
        container = ttk.Frame(parent, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        # Hero Banner
        hero = tk.Label(
            container,
            text="3-Way Model Evaluation & Generalization Matrix\n"
                 "Baseline Authentic (TRTR) vs Heuristic Paraphrase (TRSTR-Paraphrase) vs Synthetic LLM (TRSTR-LLM)\n"
                 "Key Result: TRSTR-LLM achieves +12.31% Recall on unseen out-of-domain tech stacks with -42.8% fewer false positives.",
            font=self.f_bold,
            fg="#1e40af",
            bg="#eff6ff",
            relief="groove",
            padx=14,
            pady=10,
            justify=tk.LEFT,
            anchor="w",
        )
        hero.pack(fill=tk.X, pady=(0, 10))

        # Comparative Table
        table_frame = ttk.LabelFrame(container, text="Comparative Metrics Across All Conditions", padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(table_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.matrix_tree = ttk.Treeview(
            table_frame,
            columns=("mode", "metric", "trtr", "para", "llm", "delta"),
            show="headings",
            selectmode="browse",
            yscrollcommand=scroll.set,
        )
        scroll.config(command=self.matrix_tree.yview)

        self.matrix_tree.heading("mode", text="Evaluation Benchmark Mode", anchor="w")
        self.matrix_tree.heading("metric", text="Metric", anchor="w")
        self.matrix_tree.heading("trtr", text="TRTR Baseline", anchor="center")
        self.matrix_tree.heading("para", text="TRSTR-Paraphrase", anchor="center")
        self.matrix_tree.heading("llm", text="TRSTR-LLM (Top)", anchor="center")
        self.matrix_tree.heading("delta", text="LLM vs Baseline Lift", anchor="center")

        self.matrix_tree.column("mode", width=240, anchor="w")
        self.matrix_tree.column("metric", width=180, anchor="w")
        self.matrix_tree.column("trtr", width=120, anchor="center")
        self.matrix_tree.column("para", width=140, anchor="center")
        self.matrix_tree.column("llm", width=160, anchor="center")
        self.matrix_tree.column("delta", width=150, anchor="center")

        self.matrix_tree.pack(fill=tk.BOTH, expand=True)

        self.matrix_tree.tag_configure("highlight", background="#f0fdf4")
        self.matrix_tree.tag_configure("winner_row", background="#ecfdf5", font=self.f_bold)

        self._populate_matrix_table()

    def _populate_matrix_table(self):
        """Populate the 3-way comparative matrix."""
        rows = [
            ("Held-Out Test Set (199 docs)", "Overall Precision", "65.96%", "68.12%", "67.57%", "+1.61%"),
            ("Held-Out Test Set (199 docs)", "Overall Recall", "67.69%", "67.69%", "69.23%", "+1.54%"),
            ("Held-Out Test Set (199 docs)", "Overall F1-Score", "66.81%", "67.90%", "68.39%", "+1.58%"),
            ("Held-Out Test Set (199 docs)", "IT_TERM F1", "67.29%", "68.34%", "69.26%", "+1.97%"),
            ("Held-Out Test Set (199 docs)", "CLERICAL_TERM F1", "65.73%", "72.37%", "72.73%", "+7.00%"),
            ("Unseen Benchmark (Hybrid Pipeline)", "Precision", "31.30%", "35.07%", "60.78%", "+29.48%"),
            ("Unseen Benchmark (Hybrid Pipeline)", "Recall", "63.08%", "72.31%", "95.38%", "+32.30%"),
            ("Unseen Benchmark (Hybrid Pipeline)", "F1-Score", "41.84%", "47.24%", "74.25%", "+32.41%"),
            ("Unseen Benchmark (Hybrid Pipeline)", "False Positives (Spurious)", "76 errors", "73 errors", "48 errors", "-28 errors (-36.8%)"),
            ("Unseen Benchmark (Transformer Only)", "Precision", "30.53%", "35.07%", "57.41%", "+26.88%"),
            ("Unseen Benchmark (Transformer Only)", "Recall", "61.54%", "72.31%", "95.38%", "+33.84%"),
            ("Unseen Benchmark (Transformer Only)", "F1-Score", "40.82%", "47.24%", "71.68%", "+30.86%"),
            ("Unseen Benchmark (Entity Ruler)", "Precision", "100.0%", "100.0%", "100.0%", "Exact Match"),
            ("Unseen Benchmark (Entity Ruler)", "Recall", "4.62%", "4.62%", "4.62%", "Dictionary Only"),
        ]

        for r in rows:
            tag = "winner_row" if "Hybrid Pipeline" in r[0] else ("highlight" if "+" in r[5] else "")
            self.matrix_tree.insert("", tk.END, values=r, tags=(tag,))

    # -----------------------------------------------------------------------
    # Status Bar
    # -----------------------------------------------------------------------
    def _build_status_bar(self):
        self.status_bar = tk.Label(
            self.root,
            text="Ready. Use Up/Down arrows to scan records.",
            font=("TkDefaultFont", 8),
            fg="#475569",
            bg="#e2e8f0",
            anchor="w",
            padx=10,
            pady=3,
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # -----------------------------------------------------------------------
    # Event Handlers & View Management
    # -----------------------------------------------------------------------
    def _on_tab_changed(self, event):
        selected_tab = self.notebook.select()
        if not selected_tab:
            return
        widget = self.root.nametowidget(selected_tab)
        self.active_file_type = self.tab_map.get(widget, "held_out_test")
        self._refresh_active_view()

    def _on_model_changed(self, file_type: str, model_id: str):
        self.active_model = model_id
        self._refresh_active_view()

    def _on_scope_changed(self, file_type: str, scope_id: str):
        self.active_scope = scope_id
        self._refresh_active_view()

    def _on_filter_changed(self, file_type: str):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return
        self.search_query = w["search_var"].get().strip().lower()
        self.status_filter = w["status_var"].get()
        self.label_filter = w["label_var"].get()
        self.source_filter = w["source_var"].get()
        self._populate_document_list(file_type)

    def _reset_filters(self, file_type: str, s_var, st_var, l_var, src_var):
        s_var.set("")
        st_var.set("All")
        l_var.set("All")
        src_var.set("All")
        self.search_query = ""
        self.status_filter = "All"
        self.label_filter = "All"
        self.source_filter = "All"
        self._populate_document_list(file_type)

    def _refresh_active_view(self):
        if self.active_file_type == "dictionary_overrides":
            self._filter_dict_overrides()
            return
        if self.active_file_type == "matrix_summary":
            return

        w = self.eval_tab_widgets.get(self.active_file_type)
        if not w:
            return

        # Update Radio button states
        w["model_var"].set(self.active_model)
        w["scope_var"].set(self.active_scope)

        if self.active_model == "compare":
            # Switch to Side-by-Side Mode
            w["single_inspect_frame"].pack_forget()
            w["sbs_inspect_frame"].pack(fill=tk.BOTH, expand=True)
            self._populate_sbs_document_list(self.active_file_type)
        else:
            # Standard Single Model Mode
            w["sbs_inspect_frame"].pack_forget()
            w["single_inspect_frame"].pack(fill=tk.BOTH, expand=True)
            self._populate_document_list(self.active_file_type)

    # -----------------------------------------------------------------------
    # Document List Population & Filtering (Single Model)
    # -----------------------------------------------------------------------
    def _populate_document_list(self, file_type: str):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return

        tree = w["doc_tree"]
        tree.delete(*tree.get_children())

        # Retrieve raw records for active model and scope
        ft_dict = self.data["datasets"].get(file_type, {})
        model_dict = ft_dict.get(self.active_model, {})
        raw_records = model_dict.get(self.active_scope, [])

        filtered: List[Tuple[int, Dict[str, Any]]] = []
        for idx, row in enumerate(raw_records):
            # 1. Search Query
            if self.search_query:
                q = self.search_query
                text_m = q in row.get("text", "").lower()
                gold_m = any(q in (g.get("term", "").lower()) for g in row.get("gold_entities", []))
                pred_m = any(q in (p.get("term", "").lower()) for p in row.get("pred_entities", []))
                if not (text_m or gold_m or pred_m):
                    continue

            # 2. Status Filter
            if self.status_filter != "All":
                has_st = any(er.get("status") == self.status_filter for er in row.get("entity_results", []))
                if not has_st:
                    continue

            # 3. Label Filter
            if self.label_filter != "All":
                has_lbl = any(
                    er.get("gold_label") == self.label_filter or er.get("pred_label") == self.label_filter
                    for er in row.get("entity_results", [])
                )
                if not has_lbl:
                    continue

            # 4. Source Filter
            if self.source_filter != "All":
                has_src = any(p.get("source") == self.source_filter for p in row.get("pred_entities", []))
                if not has_src:
                    continue

            filtered.append((idx, row))

        self.current_filtered_records = filtered

        # Update KPI Banner
        self._update_kpi_banner(file_type, raw_records, [r for _, r in filtered])

        # Insert rows into Treeview
        for orig_idx, row in filtered:
            results = row.get("entity_results", [])
            statuses = list(set(er.get("status") for er in results))
            has_error = any(s != "correct" for s in statuses)

            status_str = "✅ Correct" if (not has_error and len(results) > 0) else (
                "⚠️ Error" if has_error else "—"
            )
            if "boundary_error" in statuses:
                status_str = "⚠️ Boundary Error"
            elif "false_negative" in statuses:
                status_str = "❌ Missed Gold"
            elif "false_positive" in statuses:
                status_str = "🚫 Spurious Pred"

            g_count = len(row.get("gold_entities", []))
            p_count = len(row.get("pred_entities", []))
            snippet = row.get("text", "")[:55].replace("\n", " ") + "..."

            tag = "has_error" if has_error else "all_correct"
            item_id = tree.insert(
                "", tk.END,
                values=(f"#{orig_idx + 1}", status_str, g_count, p_count, snippet),
                tags=(tag,)
            )

        w["lbl_doc_count"].config(text=f"Showing {len(filtered)} of {len(raw_records)} documents")

        # Auto-select first item
        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            tree.focus(children[0])
            self._render_doc_details(file_type, 0)
        else:
            self._clear_doc_details(file_type)

    def _update_kpi_banner(self, file_type: str, all_records: List[Dict], filtered_records: List[Dict]):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return

        total_docs = len(all_records)
        filtered_docs = len(filtered_records)
        total_gold = sum(len(r.get("gold_entities", [])) for r in filtered_records)
        correct = sum(sum(1 for er in r.get("entity_results", []) if er.get("status") == "correct") for r in filtered_records)
        boundary = sum(sum(1 for er in r.get("entity_results", []) if er.get("status") == "boundary_error") for r in filtered_records)
        spurious = sum(sum(1 for er in r.get("entity_results", []) if er.get("status") == "false_positive") for r in filtered_records)
        missed = sum(sum(1 for er in r.get("entity_results", []) if er.get("status") == "false_negative") for r in filtered_records)

        acc = (correct / total_gold * 100) if total_gold > 0 else 0.0

        model_name = {
            "trtr": "TRTR Baseline",
            "trstr_paraphrase": "TRSTR-Paraphrase",
            "trstr_llm": "TRSTR-LLM (High Recall)",
        }.get(self.active_model, self.active_model)

        kpi_text = (
            f"Active: {model_name}  |  Filtered: {filtered_docs}/{total_docs} docs  |  "
            f"Gold: {total_gold}  |  Exact Matches: {correct} ({acc:.1f}%)  |  "
            f"Boundary Errs: {boundary}  |  Spurious (FP): {spurious}  |  Missed (FN): {missed}"
        )
        w["kpi_lbl"].config(text=kpi_text)

    # -----------------------------------------------------------------------
    # Document Inspection & Text Highlighting (Single Model)
    # -----------------------------------------------------------------------
    def _on_doc_selected(self, file_type: str):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return
        selected = w["doc_tree"].selection()
        if not selected:
            return
        item_idx = w["doc_tree"].index(selected[0])
        if 0 <= item_idx < len(self.current_filtered_records):
            self._render_doc_details(file_type, item_idx)

    def _on_view_mode_changed(self, file_type: str):
        """Re-render document text when the boundary view mode radio button changes."""
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return
        selected = w["doc_tree"].selection()
        if not selected:
            return
        idx = w["doc_tree"].index(selected[0])
        self._render_doc_details(file_type, idx)

    def _on_entity_selected(self, file_type: str):
        """Handle selection in the Entity Breakdown table to focus on the span and show boundary diff."""
        w = self.eval_tab_widgets.get(file_type)
        if not w or self.selected_doc_idx is None:
            return

        selected = w["ent_tree"].selection()
        if not selected:
            return

        item = w["ent_tree"].item(selected[0])
        vals = item.get("values", [])
        if not vals or len(vals) < 8:
            return

        gold_str = str(vals[0])
        pred_str = str(vals[1])
        status = str(vals[2])
        diff_str = str(vals[3])
        gold_lbl = str(vals[4])
        pred_lbl = str(vals[5])
        conf_str = str(vals[6])
        src_str = str(vals[7])

        # Update Boundary Inspector detail text
        if "boundary" in status.lower():
            detail_msg = (
                f"⚠️ BOUNDARY ERROR INSPECTION:\n"
                f"• 🎯 Correct Gold Boundary : {gold_str}  (Target Category: {gold_lbl})\n"
                f"• 🤖 Model Predicted Span  : {pred_str}  (Prediction: {pred_lbl}, Conf: {conf_str}, Source: {src_str})\n"
                f"• 🔍 Boundary Analysis     : {diff_str}"
            )
            w["lbl_boundary_detail"].config(text=detail_msg, fg="#92400e", bg="#fef3c7")
        elif "correct" in status.lower():
            detail_msg = (
                f"✅ EXACT BOUNDARY MATCH:\n"
                f"• Both Gold and Model matched: {gold_str} ({gold_lbl})\n"
                f"• Model predicted exact matching boundary with {conf_str} confidence via {src_str}."
            )
            w["lbl_boundary_detail"].config(text=detail_msg, fg="#166534", bg="#f0fdf4")
        elif "missed" in status.lower():
            detail_msg = (
                f"❌ MISSED GOLD ENTITY:\n"
                f"• Target Gold Span : {gold_str} ({gold_lbl})\n"
                f"• The model failed to detect or extract this annotated ground-truth span."
            )
            w["lbl_boundary_detail"].config(text=detail_msg, fg="#9f1239", bg="#fef2f2")
        elif "spurious" in status.lower():
            detail_msg = (
                f"🚫 SPURIOUS PREDICTION:\n"
                f"• Predicted Span : {pred_str} ({pred_lbl}, {conf_str}, {src_str})\n"
                f"• This span was extracted by the model, but is NOT annotated in the gold dataset."
            )
            w["lbl_boundary_detail"].config(text=detail_msg, fg="#9a3412", bg="#fff7ed")
        else:
            w["lbl_boundary_detail"].config(
                text=f"Entity: {gold_str} | Status: {status} | Diff: {diff_str}",
                fg="#1e293b", bg="#f8fafc"
            )

        # Highlight active focus in sentence text
        txt = w["txt_sentence"]
        txt.tag_remove("active_focus", "1.0", tk.END)

        m_offsets = re.search(r"\[(\d+):(\d+)\]", gold_str if "—" not in gold_str else pred_str)
        if m_offsets:
            s_char, e_char = int(m_offsets.group(1)), int(m_offsets.group(2))
            st_idx = f"1.0 + {s_char} chars"
            en_idx = f"1.0 + {e_char} chars"
            txt.tag_add("active_focus", st_idx, en_idx)
            txt.see(st_idx)

    def _render_doc_details(self, file_type: str, filtered_list_idx: int):
        w = self.eval_tab_widgets.get(file_type)
        if not w or filtered_list_idx >= len(self.current_filtered_records):
            return

        orig_idx, doc = self.current_filtered_records[filtered_list_idx]
        self.selected_doc_idx = orig_idx

        view_mode = w["view_mode_var"].get() if "view_mode_var" in w else "overlay"

        # 1. Render Sentence Text with Highlights
        text_widget = w["txt_sentence"]
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)

        sentence = doc.get("text", "")
        text_widget.insert(tk.END, sentence)

        results = doc.get("entity_results", [])

        if view_mode == "gold":
            # Highlight ONLY ground truth gold boundaries
            for er in results:
                gs = er.get("gold_start")
                ge = er.get("gold_end")
                if gs is not None and ge is not None and gs < ge:
                    lbl = er.get("gold_label") or "IT_TERM"
                    tag_name = "it_term" if lbl == "IT_TERM" else "clerical_term"
                    text_widget.tag_add(tag_name, f"1.0 + {gs} chars", f"1.0 + {ge} chars")
                    text_widget.tag_add("gold_boundary", f"1.0 + {gs} chars", f"1.0 + {ge} chars")

        elif view_mode == "pred":
            # Highlight ONLY model predicted boundaries
            for er in results:
                ps = er.get("pred_start")
                pe = er.get("pred_end")
                if ps is not None and pe is not None and ps < pe:
                    lbl = er.get("pred_label") or "IT_TERM"
                    tag_name = "it_term" if lbl == "IT_TERM" else "clerical_term"
                    text_widget.tag_add(tag_name, f"1.0 + {ps} chars", f"1.0 + {pe} chars")
                    text_widget.tag_add("pred_boundary", f"1.0 + {ps} chars", f"1.0 + {pe} chars")

        else:
            # Dual Boundary Comparison Mode (Overlay)
            for er in results:
                gs = er.get("gold_start")
                ge = er.get("gold_end")
                ps = er.get("pred_start")
                pe = er.get("pred_end")
                status = er.get("status", "")
                lbl = er.get("pred_label") or er.get("gold_label") or ""

                if status == "correct":
                    if gs is not None and ge is not None and gs < ge:
                        tag_name = "it_term" if lbl == "IT_TERM" else "clerical_term"
                        text_widget.tag_add(tag_name, f"1.0 + {gs} chars", f"1.0 + {ge} chars")
                elif status == "boundary_error":
                    # Tag predicted span with boundary_pred (amber)
                    if ps is not None and pe is not None and ps < pe:
                        text_widget.tag_add("boundary_pred", f"1.0 + {ps} chars", f"1.0 + {pe} chars")
                    # Tag correct gold span with boundary_gold (distinct green)
                    if gs is not None and ge is not None and gs < ge:
                        text_widget.tag_add("boundary_gold", f"1.0 + {gs} chars", f"1.0 + {ge} chars")
                elif status == "false_negative":
                    if gs is not None and ge is not None and gs < ge:
                        text_widget.tag_add("missed", f"1.0 + {gs} chars", f"1.0 + {ge} chars")
                elif status == "false_positive":
                    if ps is not None and pe is not None and ps < pe:
                        text_widget.tag_add("spurious", f"1.0 + {ps} chars", f"1.0 + {pe} chars")

        text_widget.config(state=tk.DISABLED)

        # 2. Render Entity Breakdown Table
        ent_tree = w["ent_tree"]
        ent_tree.delete(*ent_tree.get_children())

        first_boundary_er = None

        if not results:
            ent_tree.insert("", tk.END, values=("(No entities in document)", "—", "—", "—", "—", "—", "—", "—"))
        else:
            for er in results:
                gs = er.get("gold_start")
                ge = er.get("gold_end")
                ps = er.get("pred_start")
                pe = er.get("pred_end")
                gt = er.get("gold_term")
                pt = er.get("pred_term")
                status = er.get("status", "unknown")

                diff = explain_boundary(sentence, gs, ge, ps, pe)
                gold_col = f"{gt} [{gs}:{ge}]" if gt is not None else "— (Spurious)"
                pred_col = f"{pt} [{ps}:{pe}]" if pt is not None else "— (Missed)"

                gold_lbl = er.get("gold_label") or "—"
                pred_lbl = er.get("pred_label") or "—"

                conf = er.get("pred_confidence")
                conf_str = f"{conf * 100:.1f}%" if conf is not None else "—"
                source = er.get("pred_source") or "—"

                status_display = {
                    "correct": "✅ Correct",
                    "boundary_error": "⚠️ Boundary Error",
                    "false_negative": "❌ Missed Gold",
                    "false_positive": "🚫 Spurious Pred",
                    "label_error": "🏷️ Label Confusion"
                }.get(status, status)

                tag = status
                ent_tree.insert(
                    "", tk.END,
                    values=(gold_col, pred_col, status_display, diff, gold_lbl, pred_lbl, conf_str, source),
                    tags=(tag,)
                )

                if status == "boundary_error" and first_boundary_er is None:
                    first_boundary_er = er

        # 3. Update Boundary Error Inspector Card
        if first_boundary_er is not None:
            f_gs = first_boundary_er.get("gold_start")
            f_ge = first_boundary_er.get("gold_end")
            f_ps = first_boundary_er.get("pred_start")
            f_pe = first_boundary_er.get("pred_end")
            f_gt = first_boundary_er.get("gold_term")
            f_pt = first_boundary_er.get("pred_term")
            f_gl = first_boundary_er.get("gold_label", "")
            f_pl = first_boundary_er.get("pred_label", "")
            f_conf = f"{first_boundary_er.get('pred_confidence', 0)*100:.1f}%" if first_boundary_er.get("pred_confidence") is not None else "—"
            f_src = first_boundary_er.get("pred_source", "ML")
            f_diff = explain_boundary(sentence, f_gs, f_ge, f_ps, f_pe)

            boundary_text = (
                f"⚠️ BOUNDARY ERROR DETECTED:\n"
                f"• 🎯 Correct Gold Boundary : '{f_gt}' [{f_gs}:{f_ge}]  ({f_gl}, length: {f_ge - f_gs if f_ge and f_gs else 0} chars)\n"
                f"• 🤖 Model Predicted Span  : '{f_pt}' [{f_ps}:{f_pe}]  ({f_pl}, Conf: {f_conf}, Source: {f_src})\n"
                f"• 🔍 Boundary Analysis     : {f_diff}"
            )
            w["lbl_boundary_detail"].config(text=boundary_text, fg="#92400e", bg="#fef3c7")
        else:
            w["lbl_boundary_detail"].config(
                text="✓ No boundary errors in this document. All detected entities match exact gold token boundaries.",
                fg="#166534",
                bg="#f0fdf4",
            )

        self.status_bar.config(
            text=f"Inspecting Document #{orig_idx + 1} | {len(sentence)} chars | {len(results)} entity evaluation entries"
        )

    def _clear_doc_details(self, file_type: str):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return
        w["txt_sentence"].config(state=tk.NORMAL)
        w["txt_sentence"].delete("1.0", tk.END)
        w["txt_sentence"].config(state=tk.DISABLED)
        w["ent_tree"].delete(*w["ent_tree"].get_children())

    def _jump_error(self, file_type: str, direction: int):
        w = self.eval_tab_widgets.get(file_type)
        if not w or not self.current_filtered_records:
            return
        tree = w["doc_tree"]
        selected = tree.selection()
        curr_idx = tree.index(selected[0]) if selected else 0

        target_idx = curr_idx + direction
        while 0 <= target_idx < len(self.current_filtered_records):
            _, doc = self.current_filtered_records[target_idx]
            if any(er.get("status") != "correct" for er in doc.get("entity_results", [])):
                children = tree.get_children()
                tree.selection_set(children[target_idx])
                tree.see(children[target_idx])
                self._render_doc_details(file_type, target_idx)
                return
            target_idx += direction

        messagebox.showinfo("Scanner", "No further error records found in that direction.")

    # -----------------------------------------------------------------------
    # Side-by-Side 3-Way Mode (Compare All 3)
    # -----------------------------------------------------------------------
    def _populate_sbs_document_list(self, file_type: str):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return

        tree = w["doc_tree"]
        tree.delete(*tree.get_children())

        ft_dict = self.data["datasets"].get(file_type, {})
        trtr_docs = ft_dict.get("trtr", {}).get("all", [])
        para_docs = ft_dict.get("trstr_paraphrase", {}).get("all", [])
        llm_docs = ft_dict.get("trstr_llm", {}).get("all", [])

        total = len(trtr_docs)
        aligned = []

        for i in range(total):
            t = trtr_docs[i] if i < len(trtr_docs) else {}
            p = para_docs[i] if i < len(para_docs) else {}
            l = llm_docs[i] if i < len(llm_docs) else {}

            t_errs = sum(1 for er in t.get("entity_results", []) if er.get("status") != "correct")
            p_errs = sum(1 for er in p.get("entity_results", []) if er.get("status") != "correct")
            l_errs = sum(1 for er in l.get("entity_results", []) if er.get("status") != "correct")

            if l_errs == 0 and t_errs > 0:
                consensus = "🟢 LLM Fixed Error"
            elif t_errs == 0 and p_errs == 0 and l_errs == 0:
                consensus = "🔵 All Correct"
            elif t_errs > 0 and p_errs > 0 and l_errs > 0:
                consensus = "🔴 All Erred"
            else:
                consensus = "🟡 Divergence"

            aligned.append((i, t, p, l, consensus))

        # Filter aligned rows
        filtered = []
        for i, t, p, l, consensus in aligned:
            text = t.get("text", "")
            if self.search_query and self.search_query not in text.lower():
                continue
            filtered.append((i, {"text": text, "trtr": t, "para": p, "llm": l, "consensus": consensus}))

        self.current_filtered_records = filtered

        # Populate tree
        for orig_idx, row in filtered:
            snippet = row["text"][:55].replace("\n", " ") + "..."
            tree.insert(
                "", tk.END,
                values=(f"#{orig_idx + 1}", row["consensus"], len(row["trtr"].get("gold_entities", [])), "3 Models", snippet)
            )

        w["lbl_doc_count"].config(text=f"Showing {len(filtered)} aligned sentences")
        w["kpi_lbl"].config(
            text="⚖️ Side-by-Side 3-Way Mode: Inspecting sentence consensus across TRTR, TRSTR-Paraphrase, and TRSTR-LLM"
        )

        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            self._render_sbs_details(file_type, 0)

    def _render_sbs_details(self, file_type: str, filtered_list_idx: int):
        w = self.eval_tab_widgets.get(file_type)
        if not w or filtered_list_idx >= len(self.current_filtered_records):
            return

        orig_idx, row = self.current_filtered_records[filtered_list_idx]
        sbs_frame = w["sbs_inspect_frame"]

        # Clear previous widgets in sbs frame
        for child in sbs_frame.winfo_children():
            child.destroy()

        # Top: Sentence text
        top_box = ttk.Frame(sbs_frame)
        top_box.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(top_box, text=f"Sentence #{orig_idx + 1} — Consensus: {row['consensus']}", font=self.f_bold).pack(anchor="w")

        txt_sbs_sent = tk.Text(
            top_box, wrap=tk.WORD, font=("TkDefaultFont", 10), height=4, relief="solid", borderwidth=1, padx=8, pady=6
        )
        txt_sbs_sent.pack(fill=tk.X, pady=(2, 4))
        txt_sbs_sent.insert(tk.END, row["text"])
        txt_sbs_sent.config(state=tk.DISABLED)

        # 3 Side-by-Side Columns
        cols_frame = ttk.Frame(sbs_frame)
        cols_frame.pack(fill=tk.BOTH, expand=True)

        models = [
            ("TRTR Baseline", row["trtr"], "#dbeafe"),
            ("TRSTR-Paraphrase", row["para"], "#f3e8ff"),
            ("TRSTR-LLM (High Recall)", row["llm"], "#dcfce7"),
        ]

        for col_idx, (m_name, m_doc, m_bg) in enumerate(models):
            col_box = ttk.LabelFrame(cols_frame, text=m_name, padding=6)
            col_box.grid(row=0, column=col_idx, sticky="nsew", padx=3)
            cols_frame.columnconfigure(col_idx, weight=1)

            # Preds list
            preds = m_doc.get("pred_entities", [])
            results = m_doc.get("entity_results", [])

            err_count = sum(1 for er in results if er.get("status") != "correct")
            status_summary = f"{len(preds)} predictions ({err_count} errors)"
            tk.Label(col_box, text=status_summary, font=self.f_bold, fg="#334155").pack(anchor="w", pady=(0, 4))

            listbox = tk.Listbox(col_box, font=self.f_normal, relief="solid", borderwidth=1)
            listbox.pack(fill=tk.BOTH, expand=True)

            if not preds:
                listbox.insert(tk.END, "(No predicted entities)")
            else:
                for p in preds:
                    res = next((r for r in results if r.get("pred_term") == p.get("term")), {})
                    st = res.get("status", "unknown")
                    conf = f"{p.get('confidence', 0)*100:.1f}%" if p.get("confidence") is not None else ""
                    listbox.insert(tk.END, f"• {p.get('term')} [{p.get('label')}] ({conf}) - {st}")

    # -----------------------------------------------------------------------
    # Dictionary Overrides Audit Log (Tab 5)
    # -----------------------------------------------------------------------
    def _filter_dict_overrides(self):
        query = self.dict_search_var.get().strip().lower()
        self.dict_tree.delete(*self.dict_tree.get_children())

        overrides = self.data.get("dictionary_overrides", [])
        filtered = []

        for idx, o in enumerate(overrides):
            term = o.get("term", "")
            sent = o.get("sentence", "")
            ml = o.get("ml_label", "")
            dict_lbl = o.get("dict_label", "")

            if query:
                if not (query in term.lower() or query in sent.lower() or query in ml.lower() or query in dict_lbl.lower()):
                    continue

            filtered.append((idx, o))
            offsets = f"[{o.get('start')}:{o.get('end')}]"
            self.dict_tree.insert(
                "", tk.END,
                values=(f"#{idx + 1}", term, ml, dict_lbl, o.get("winning_label"), offsets, sent[:60] + "...")
            )

        self.lbl_dict_count.config(text=f"Showing {len(filtered)} of {len(overrides)} override events")

        children = self.dict_tree.get_children()
        if children:
            self.dict_tree.selection_set(children[0])
            self._show_dict_detail(0, filtered)

    def _on_dict_row_selected(self, event):
        selected = self.dict_tree.selection()
        if not selected:
            return
        idx = self.dict_tree.index(selected[0])
        overrides = self.data.get("dictionary_overrides", [])
        if 0 <= idx < len(overrides):
            self._show_dict_detail(idx, [(i, o) for i, o in enumerate(overrides)])

    def _show_dict_detail(self, list_idx: int, filtered_list: List):
        if list_idx >= len(filtered_list):
            return
        orig_idx, o = filtered_list[list_idx]

        txt = self.txt_dict_detail
        txt.config(state=tk.NORMAL)
        txt.delete("1.0", tk.END)

        sentence = o.get("sentence", "")
        term = o.get("term", "")
        start = o.get("start", 0)
        end = o.get("end", len(term))

        txt.insert(tk.END, f"Override Event #{orig_idx + 1} — Term: '{term}'\n")
        txt.insert(tk.END, f"Decision: ML Predicted '{o.get('ml_label')}' ➔ EntityRuler Dictionary Enforced '{o.get('winning_label')}'\n\n")
        txt.insert(tk.END, "Sentence Context: ")

        sent_start_idx = txt.index(tk.INSERT)
        txt.insert(tk.END, sentence)

        # Highlight term in sentence
        if term in sentence:
            pos = sentence.find(term)
            hl_start = f"{sent_start_idx} + {pos} chars"
            hl_end = f"{sent_start_idx} + {pos + len(term)} chars"
            txt.tag_add("highlight", hl_start, hl_end)

        txt.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Clipboard Helpers
    # -----------------------------------------------------------------------
    def _copy_sentence(self, file_type: str):
        if self.selected_doc_idx is None:
            return
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return
        txt = w["txt_sentence"].get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status_bar.config(text="✓ Copied sentence text to clipboard.")

    def _copy_json(self, file_type: str):
        w = self.eval_tab_widgets.get(file_type)
        if not w:
            return
        selected = w["doc_tree"].selection()
        if not selected:
            return
        idx = w["doc_tree"].index(selected[0])
        if idx < len(self.current_filtered_records):
            _, doc = self.current_filtered_records[idx]
            self.root.clipboard_clear()
            self.root.clipboard_append(json.dumps(doc, indent=2))
            self.status_bar.config(text="✓ Copied complete document JSON to clipboard.")

    # -----------------------------------------------------------------------
    # Reload Action
    # -----------------------------------------------------------------------
    def reload_data(self):
        self.data = load_all_eval_data()
        self._refresh_active_view()
        self.status_bar.config(text="✓ Reloaded all evaluation result files from disk.")


# ---------------------------------------------------------------------------
# Main Launcher
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    app = EvalResultsGUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
