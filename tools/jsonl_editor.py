import re
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import gc


LABELS = ["IT_TERM", "CLERICAL_TERM"]

# Entity visual tag colors
COLOR_IT_BG = "#dbeafe"          # Light soft blue
COLOR_IT_FG = "#1e40af"          # Deep blue
COLOR_CLERICAL_BG = "#d1fae5"    # Light emerald green
COLOR_CLERICAL_FG = "#065f46"    # Deep forest green
COLOR_SELECT_BG = "#fde047"      # Bright highlighter yellow
COLOR_SELECT_FG = "#854d0e"      # Deep golden brown


def make_entity_pattern(entity_text: str) -> re.Pattern:
    """
    Builds a case-insensitive regex pattern for matching entity_text as an exact term.
    Uses negative lookbehinds/lookaheads to prevent matching subwords (e.g. 'coding' in 'encoding'
    or 'IT' in 'with'), while safely supporting punctuation-containing terms like 'C++', '.NET', 'Node.js'.
    """
    text = entity_text.strip()
    escaped = re.escape(text)
    prefix = r"(?<!\w)" if text and (text[0].isalnum() or text[0] == "_") else ""
    suffix = r"(?!\w)" if text and (text[-1].isalnum() or text[-1] == "_") else ""
    return re.compile(prefix + escaped + suffix, re.IGNORECASE)


class JSONLEntityAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("JSONL Entity Annotator")
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)

        self.data = []
        self.current_index = None
        self.file_path = None
        self.sync_consistency = tk.BooleanVar(value=True)

        self.create_menu()
        self.create_ui()

    # ============================================================
    # UI
    # ============================================================

    def create_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label="Open JSONL",
            command=self.open_file,
            accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Save JSONL",
            command=self.save_file,
            accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Save JSONL As...",
            command=self.save_file_as
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit",
            command=self.root.quit
        )

        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(
            label="Remove Duplicates",
            command=self.remove_duplicates,
            accelerator="Ctrl+D"
        )
        edit_menu.add_separator()
        edit_menu.add_checkbutton(
            label="Auto-sync File Consistency",
            variable=self.sync_consistency,
            command=self._update_sync_badge
        )
        edit_menu.add_command(
            label="Audit Dataset Consistency...",
            command=self.audit_dataset_consistency
        )
        menubar.add_cascade(label="Edit", menu=edit_menu)

        self.root.config(menu=menubar)

        self.root.bind("<Control-o>", lambda event: self.open_file())
        self.root.bind("<Control-s>", lambda event: self.save_file())
        self.root.bind("<Control-d>", lambda event: self.remove_duplicates())

    def create_ui(self):
        # Status bar - packed first with side=tk.BOTTOM so it remains at the window's bottom
        self.status_var = tk.StringVar(
            value="Open a JSONL file to begin."
        )

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        ).pack(
            side=tk.BOTTOM,
            fill=tk.X
        )

        # ========================================================
        # TOP SECTION - TEXT RECORDS
        # ========================================================

        top_frame = ttk.LabelFrame(
            self.root,
            text="JSONL Records",
            padding=8
        )
        top_frame.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(10, 5)
        )

        # Top toolbar
        top_toolbar = ttk.Frame(top_frame)
        top_toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(
            top_toolbar,
            text="Open JSONL",
            command=self.open_file
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            top_toolbar,
            text="Save",
            command=self.save_file
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            top_toolbar,
            text="Add Record",
            command=self.add_record
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            top_toolbar,
            text="Remove Duplicates",
            command=self.remove_duplicates
        ).pack(side=tk.LEFT, padx=5)

        self.record_count_label = ttk.Label(
            top_toolbar,
            text="No file loaded"
        )
        self.record_count_label.pack(side=tk.RIGHT)

        # Treeview for records
        top_tree_frame = ttk.Frame(top_frame)
        top_tree_frame.pack(fill=tk.BOTH, expand=True)

        self.text_tree = ttk.Treeview(
            top_tree_frame,
            columns=("index", "text"),
            show="headings",
            selectmode="browse"
        )

        self.text_tree.heading(
            "index",
            text="#"
        )
        self.text_tree.heading(
            "text",
            text="Text"
        )

        self.text_tree.column(
            "index",
            width=60,
            anchor=tk.CENTER,
            stretch=False
        )

        self.text_tree.column(
            "text",
            width=1000,
            anchor=tk.W
        )

        text_scroll_y = ttk.Scrollbar(
            top_tree_frame,
            orient=tk.VERTICAL,
            command=self.text_tree.yview
        )

        text_scroll_x = ttk.Scrollbar(
            top_tree_frame,
            orient=tk.HORIZONTAL,
            command=self.text_tree.xview
        )

        self.text_tree.configure(
            yscrollcommand=text_scroll_y.set,
            xscrollcommand=text_scroll_x.set
        )

        self.text_tree.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        text_scroll_y.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        text_scroll_x.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        top_tree_frame.rowconfigure(0, weight=1)
        top_tree_frame.columnconfigure(0, weight=1)

        self.text_tree.bind(
            "<<TreeviewSelect>>",
            self.on_record_selected
        )

        # ========================================================
        # BOTTOM SECTION - ENTITIES
        # ========================================================

        bottom_frame = ttk.LabelFrame(
            self.root,
            text="Entities",
            padding=8
        )
        bottom_frame.pack(
            side=tk.BOTTOM,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(5, 10)
        )

        # Current sentence display
        sentence_frame = ttk.Frame(bottom_frame)
        sentence_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            pady=(0, 8)
        )

        sent_header = ttk.Frame(sentence_frame)
        sent_header.pack(fill=tk.X, pady=(0, 3))

        ttk.Label(
            sent_header,
            text="Selected sentence:",
            font=("TkDefaultFont", 9, "bold")
        ).pack(
            side=tk.LEFT
        )

        tk.Label(
            sent_header,
            text="IT_TERM",
            background=COLOR_IT_BG,
            foreground=COLOR_IT_FG,
            font=("TkDefaultFont", 8, "bold"),
            padx=5,
            pady=1,
            relief=tk.SOLID,
            borderwidth=1
        ).pack(side=tk.LEFT, padx=(12, 4))

        tk.Label(
            sent_header,
            text="CLERICAL_TERM",
            background=COLOR_CLERICAL_BG,
            foreground=COLOR_CLERICAL_FG,
            font=("TkDefaultFont", 8, "bold"),
            padx=5,
            pady=1,
            relief=tk.SOLID,
            borderwidth=1
        ).pack(side=tk.LEFT, padx=4)

        ttk.Label(
            sent_header,
            text="💡 Tip: Highlight text then click 'Add Entity' to auto-populate.",
            font=("TkDefaultFont", 8, "italic"),
            foreground="#64748b"
        ).pack(side=tk.RIGHT)

        self.selected_text_box = tk.Text(
            sentence_frame,
            height=3,
            wrap=tk.WORD,
            font=("DejaVu Sans", 10),
            padx=6,
            pady=4
        )

        self.selected_text_box.pack(
            fill=tk.X,
            pady=(2, 0)
        )

        self.selected_text_box.tag_configure(
            "IT_TERM",
            background=COLOR_IT_BG,
            foreground=COLOR_IT_FG,
            font=("DejaVu Sans", 10, "bold")
        )
        self.selected_text_box.tag_configure(
            "CLERICAL_TERM",
            background=COLOR_CLERICAL_BG,
            foreground=COLOR_CLERICAL_FG,
            font=("DejaVu Sans", 10, "bold")
        )
        self.selected_text_box.tag_configure(
            "SELECTED_SPAN",
            background=COLOR_SELECT_BG,
            foreground=COLOR_SELECT_FG,
            font=("DejaVu Sans", 10, "bold"),
            underline=True
        )
        self.selected_text_box.bind("<Button-1>", self.on_text_box_click)

        self.selected_text_box.config(
            state=tk.DISABLED
        )

        # ========================================================
        # ENTITY BUTTONS (Anchored to bottom so they always remain visible)
        # ========================================================

        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(
            side=tk.BOTTOM,
            fill=tk.X,
            pady=(8, 0)
        )

        ttk.Button(
            button_frame,
            text="Add Entity",
            command=self.add_entity
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            button_frame,
            text="Edit Entity",
            command=self.edit_entity
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Delete Entity",
            command=self.delete_entity
        ).pack(side=tk.LEFT, padx=5)

        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=12, fill=tk.Y)

        self.sync_check = ttk.Checkbutton(
            button_frame,
            text="Auto-sync file consistency (all records)",
            variable=self.sync_consistency,
            command=self._update_sync_badge
        )
        self.sync_check.pack(side=tk.LEFT, padx=4)

        self.sync_badge_label = ttk.Label(
            button_frame,
            text="⚡ Global Sync Active",
            foreground="#15803d",
            font=("TkDefaultFont", 8, "bold")
        )
        self.sync_badge_label.pack(side=tk.LEFT, padx=6)

        # ========================================================
        # ENTITY TABLE (Expands into remaining space)
        # ========================================================

        entity_tree_frame = ttk.Frame(bottom_frame)
        entity_tree_frame.pack(
            side=tk.TOP,
            fill=tk.BOTH,
            expand=True
        )

        self.entity_tree = ttk.Treeview(
            entity_tree_frame,
            columns=("entity", "start", "end", "label"),
            show="headings",
            selectmode="browse"
        )

        self.entity_tree.heading(
            "entity",
            text="Entity Text"
        )

        self.entity_tree.heading(
            "start",
            text="Start"
        )

        self.entity_tree.heading(
            "end",
            text="End"
        )

        self.entity_tree.heading(
            "label",
            text="Label"
        )

        self.entity_tree.column(
            "entity",
            width=600,
            anchor=tk.W
        )

        self.entity_tree.column(
            "start",
            width=100,
            anchor=tk.CENTER
        )

        self.entity_tree.column(
            "end",
            width=100,
            anchor=tk.CENTER
        )

        self.entity_tree.column(
            "label",
            width=180,
            anchor=tk.CENTER
        )

        entity_scroll = ttk.Scrollbar(
            entity_tree_frame,
            orient=tk.VERTICAL,
            command=self.entity_tree.yview
        )

        self.entity_tree.configure(
            yscrollcommand=entity_scroll.set
        )

        self.entity_tree.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        entity_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        entity_tree_frame.rowconfigure(0, weight=1)
        entity_tree_frame.columnconfigure(0, weight=1)

        self.entity_tree.bind(
            "<<TreeviewSelect>>",
            self.on_entity_selected
        )
        self.entity_tree.bind(
            "<Delete>",
            lambda e: self.delete_entity()
        )



    # ============================================================
    # FILE OPERATIONS
    # ============================================================

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open JSONL File",
            filetypes=[
                ("JSONL files", "*.jsonl"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        try:
            records = []

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                for line_number, line in enumerate(f, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        messagebox.showerror(
                            "Invalid JSONL",
                            f"Error on line {line_number}:\n\n{e}"
                        )
                        return

                    if "text" not in record:
                        messagebox.showerror(
                            "Invalid Record",
                            f"Line {line_number} does not contain "
                            f"a 'text' field."
                        )
                        return

                    if "entities" not in record:
                        record["entities"] = []

                    if not isinstance(
                        record["entities"],
                        list
                    ):
                        messagebox.showerror(
                            "Invalid Entities",
                            f"Line {line_number} has an invalid "
                            f"'entities' field."
                        )
                        return

                    records.append(record)

            self.data = records
            self.file_path = path
            self.current_index = None
            gc.collect()

            self.populate_text_table()

            self.status_var.set(
                f"Loaded {len(self.data)} records from "
                f"{Path(path).name}"
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not open file:\n\n{e}"
            )

    def save_file(self):
        if not self.data:
            messagebox.showwarning(
                "Nothing to Save",
                "No JSONL file is currently loaded."
            )
            return

        if not self.file_path:
            self.save_file_as()
            return

        self.write_jsonl(self.file_path)

    def save_file_as(self):
        if not self.data:
            messagebox.showwarning(
                "Nothing to Save",
                "No JSONL data is currently loaded."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save JSONL File",
            defaultextension=".jsonl",
            filetypes=[
                ("JSONL files", "*.jsonl"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.file_path = path
        self.write_jsonl(path)

    def write_jsonl(self, path):
        try:
            with open(
                path,
                "w",
                encoding="utf-8"
            ) as f:

                for record in self.data:
                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

            self.status_var.set(
                f"Saved: {Path(path).name}"
            )

            messagebox.showinfo(
                "Saved",
                "The JSONL file was saved successfully."
            )

        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Could not save the file:\n\n{e}"
            )

    # ============================================================
    # TOP TABLE & DEDUPLICATION
    # ============================================================

    def _normalize_text_for_hash(self, text: str) -> str:
        """
        Trims leading, trailing, and middle whitespace for hash set comparison,
        without modifying the original text or invalidating entity offsets.
        """
        if not text:
            return ""
        # split() trims leading/trailing spaces and collapses consecutive middle whitespace
        return " ".join(text.split())

    def remove_duplicates(self):
        """
        Removes duplicate records based on normalized text comparison using a hash set.
        Original texts and entity offsets in retained records are preserved without edits.
        """
        if not self.data:
            messagebox.showinfo(
                "Remove Duplicates",
                "No records are currently loaded."
            )
            return

        seen_hashes = set()
        unique_data = []
        duplicate_count = 0

        for record in self.data:
            raw_text = record.get("text", "")
            # Trim start, end, and middle spaces for hash set lookup only:
            norm_key = self._normalize_text_for_hash(raw_text)

            if norm_key in seen_hashes:
                duplicate_count += 1
            else:
                seen_hashes.add(norm_key)
                unique_data.append(record)

        del seen_hashes

        if duplicate_count == 0:
            messagebox.showinfo(
                "Remove Duplicates",
                "No duplicate records found."
            )
            return

        confirm = messagebox.askyesno(
            "Remove Duplicates",
            f"Found {duplicate_count} duplicate record(s).\n\n"
            f"Keep {len(unique_data)} unique records and remove duplicates?"
        )
        if not confirm:
            del unique_data
            return

        # Release previous row and data references
        self.current_index = None
        self.clear_entity_table()
        self.clear_selected_text()

        self.data = unique_data
        del unique_data
        gc.collect()

        self.populate_text_table()

        self.status_var.set(
            f"Removed {duplicate_count} duplicate record(s). "
            f"{len(self.data)} records remaining."
        )

    def populate_text_table(self):
        children = self.text_tree.get_children()
        if children:
            self.text_tree.delete(*children)

        for index, record in enumerate(self.data):
            self.text_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    index + 1,
                    record["text"]
                )
            )

        self.record_count_label.config(
            text=f"{len(self.data)} records"
        )

        # Clear entity table and selected text
        self.clear_entity_table()
        self.clear_selected_text()

    def on_record_selected(self, event=None):
        selection = self.text_tree.selection()

        if not selection:
            self.current_index = None
            self.clear_entity_table()
            self.clear_selected_text()
            return

        index = int(selection[0])

        # If switching away from previous row, remove previous row data from memory/UI
        if self.current_index is not None and self.current_index != index:
            self.clear_entity_table()
            self.clear_selected_text()

        self.current_index = index
        record = self.data[index]

        # Display selected text
        self.selected_text_box.config(state=tk.NORMAL)
        self.selected_text_box.delete("1.0", tk.END)
        self.selected_text_box.edit_reset()
        self.selected_text_box.insert(
            "1.0",
            record["text"]
        )
        self.selected_text_box.config(state=tk.DISABLED)

        self.populate_entity_table()

        self.status_var.set(
            f"Record {index + 1} selected — "
            f"{len(record.get('entities', []))} entities"
        )

    # ============================================================
    # ENTITY TABLE
    # ============================================================

    def _update_sync_badge(self):
        if self.sync_consistency.get():
            self.sync_badge_label.config(text="⚡ Global Sync Active", foreground="#15803d")
        else:
            self.sync_badge_label.config(text="○ Sync Disabled", foreground="#94a3b8")

    def clear_selected_text(self):
        self.selected_text_box.config(state=tk.NORMAL)
        for tag in ["IT_TERM", "CLERICAL_TERM", "SELECTED_SPAN"]:
            self.selected_text_box.tag_remove(tag, "1.0", tk.END)
        self.selected_text_box.delete(
            "1.0",
            tk.END
        )
        self.selected_text_box.edit_reset()
        self.selected_text_box.config(state=tk.DISABLED)

    def clear_entity_table(self):
        children = self.entity_tree.get_children()
        if children:
            self.entity_tree.delete(*children)

    def highlight_entities_in_text(self):
        """Highlights entities in the selected text box using colored tags."""
        if self.current_index is None or not (0 <= self.current_index < len(self.data)):
            return

        record = self.data[self.current_index]
        text = record.get("text", "")

        # Clear existing highlight tags
        for tag in ["IT_TERM", "CLERICAL_TERM", "SELECTED_SPAN"]:
            self.selected_text_box.tag_remove(tag, "1.0", tk.END)

        for entity in record.get("entities", []):
            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label", "")

            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(text)
            ):
                idx_start = f"1.0+{start}c"
                idx_end = f"1.0+{end}c"
                if label in LABELS:
                    self.selected_text_box.tag_add(label, idx_start, idx_end)

    def on_entity_selected(self, event=None):
        """Highlights the selected entity span in the sentence text box."""
        if self.current_index is None or not (0 <= self.current_index < len(self.data)):
            return

        selection = self.entity_tree.selection()
        if not selection:
            self.selected_text_box.tag_remove("SELECTED_SPAN", "1.0", tk.END)
            return

        item_vals = self.entity_tree.item(selection[0], "values")
        if len(item_vals) >= 3:
            try:
                start = int(item_vals[1])
                end = int(item_vals[2])
                self.selected_text_box.tag_remove("SELECTED_SPAN", "1.0", tk.END)
                idx_start = f"1.0+{start}c"
                idx_end = f"1.0+{end}c"
                self.selected_text_box.tag_add("SELECTED_SPAN", idx_start, idx_end)
                self.selected_text_box.see(idx_start)
            except (ValueError, TypeError):
                pass

    def on_text_box_click(self, event=None):
        """When clicking an entity in the text box, select that entity in the entity table."""
        if self.current_index is None or not (0 <= self.current_index < len(self.data)):
            return

        idx = self.selected_text_box.index(f"@{event.x},{event.y}")
        text_before = self.selected_text_box.get("1.0", idx)
        offset = len(text_before)

        record = self.data[self.current_index]
        for ent_idx, ent in enumerate(record.get("entities", [])):
            if ent.get("start", -1) <= offset <= ent.get("end", -1):
                iid = str(ent_idx)
                if self.entity_tree.exists(iid):
                    self.entity_tree.selection_set(iid)
                    self.entity_tree.see(iid)
                    self.on_entity_selected()
                break

    def populate_entity_table(self):
        self.clear_entity_table()

        if self.current_index is None or not (0 <= self.current_index < len(self.data)):
            return

        record = self.data[self.current_index]

        # Sort entities by start position
        record["entities"].sort(
            key=lambda x: x.get("start", 0)
        )

        text = record["text"]

        for index, entity in enumerate(
            record["entities"]
        ):
            start = entity.get("start")
            end = entity.get("end")
            label = entity.get("label", "")

            entity_text = ""

            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start <= end <= len(text)
            ):
                entity_text = text[start:end]

            self.entity_tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    entity_text,
                    start,
                    end,
                    label
                )
            )

        # Highlight tags in sentence text box
        self.highlight_entities_in_text()

    # ============================================================
    # GLOBAL CONSISTENCY SYNCHRONIZATION ENGINE
    # ============================================================

    def sync_add_entity(
        self,
        entity_text: str,
        label: str
    ) -> Tuple[int, int, int]:
        """
        Scans all records in self.data for occurrences of entity_text (case-insensitive substring)
        and adds or updates them with the specified label for dataset consistency.

        Returns: (added_count, relabeled_count, affected_records_count)
        """
        if not self.data or not entity_text.strip():
            return (0, 0, 0)

        pat = make_entity_pattern(entity_text)
        added_count = 0
        relabeled_count = 0
        affected_records = set()

        for rec_idx, record in enumerate(self.data):
            rec_text = record.get("text", "")
            if not rec_text:
                continue

            record_modified = False
            existing_entities = record.setdefault("entities", [])

            for m in pat.finditer(rec_text):
                m_start, m_end = m.start(), m.end()

                overlap = False
                exact_match = None

                for e in existing_entities:
                    e_start = e.get("start")
                    e_end = e.get("end")
                    if e_start == m_start and e_end == m_end:
                        exact_match = e
                        break
                    elif m_start < e_end and m_end > e_start:
                        # Partial overlap with another entity span
                        overlap = True
                        break

                if exact_match is not None:
                    if exact_match.get("label") != label:
                        exact_match["label"] = label
                        relabeled_count += 1
                        record_modified = True
                elif not overlap:
                    existing_entities.append({
                        "start": m_start,
                        "end": m_end,
                        "label": label
                    })
                    added_count += 1
                    record_modified = True

            if record_modified:
                existing_entities.sort(key=lambda x: x.get("start", 0))
                affected_records.add(rec_idx)

        return (added_count, relabeled_count, len(affected_records))

    def sync_edit_entity(
        self,
        old_text: str,
        old_label: str,
        new_text: str,
        new_label: str
    ) -> Tuple[int, int, int]:
        """
        Propagates entity edits across all records in self.data for dataset consistency.
        If term text is identical (label change or casing only), updates all matching entities.
        If term text changed, removes old entities and tags new term occurrences.

        Returns: (relabeled_or_removed_count, added_count, affected_records_count)
        """
        if not self.data:
            return (0, 0, 0)

        old_norm = old_text.strip().lower()
        new_norm = new_text.strip().lower()
        relabeled_count = 0
        added_count = 0
        affected_records = set()

        if old_norm == new_norm:
            # Case 1: Label change only (or case change)
            for rec_idx, record in enumerate(self.data):
                rec_text = record.get("text", "")
                record_modified = False
                for e in record.get("entities", []):
                    s, end = e.get("start", 0), e.get("end", 0)
                    if 0 <= s <= end <= len(rec_text):
                        if rec_text[s:end].lower() == old_norm:
                            if e.get("label") != new_label:
                                e["label"] = new_label
                                relabeled_count += 1
                                record_modified = True
                if record_modified:
                    affected_records.add(rec_idx)
        else:
            # Case 2: Entity term changed
            new_pat = make_entity_pattern(new_text)
            for rec_idx, record in enumerate(self.data):
                rec_text = record.get("text", "")
                record_modified = False
                kept_entities = []

                # Remove old matching entities
                for e in record.get("entities", []):
                    s, end = e.get("start", 0), e.get("end", 0)
                    if 0 <= s <= end <= len(rec_text) and rec_text[s:end].lower() == old_norm:
                        relabeled_count += 1
                        record_modified = True
                    else:
                        kept_entities.append(e)

                record["entities"] = kept_entities

                # Add new term occurrences where valid
                for m in new_pat.finditer(rec_text):
                    m_start, m_end = m.start(), m.end()
                    overlap = False
                    exact_match = None

                    for e in kept_entities:
                        e_start = e.get("start")
                        e_end = e.get("end")
                        if e_start == m_start and e_end == m_end:
                            exact_match = e
                            break
                        elif m_start < e_end and m_end > e_start:
                            overlap = True
                            break

                    if exact_match is not None:
                        if exact_match.get("label") != new_label:
                            exact_match["label"] = new_label
                            record_modified = True
                    elif not overlap:
                        kept_entities.append({
                            "start": m_start,
                            "end": m_end,
                            "label": new_label
                        })
                        added_count += 1
                        record_modified = True

                if record_modified:
                    record["entities"].sort(key=lambda x: x.get("start", 0))
                    affected_records.add(rec_idx)

        return (relabeled_count, added_count, len(affected_records))

    def sync_delete_entity(
        self,
        entity_text: str
    ) -> Tuple[int, int]:
        """
        Removes all occurrences of entity_text (case-insensitive) across all records in self.data.

        Returns: (deleted_count, affected_records_count)
        """
        if not self.data or not entity_text.strip():
            return (0, 0)

        norm = entity_text.strip().lower()
        deleted_count = 0
        affected_records = set()

        for rec_idx, record in enumerate(self.data):
            rec_text = record.get("text", "")
            kept_entities = []
            record_modified = False

            for e in record.get("entities", []):
                s, end = e.get("start", 0), e.get("end", 0)
                if 0 <= s <= end <= len(rec_text) and rec_text[s:end].lower() == norm:
                    deleted_count += 1
                    record_modified = True
                else:
                    kept_entities.append(e)

            if record_modified:
                record["entities"] = kept_entities
                affected_records.add(rec_idx)

        return (deleted_count, len(affected_records))

    def audit_dataset_consistency(self):
        """Scans the entire dataset for inconsistent entity annotations and displays a report."""
        if not self.data:
            messagebox.showinfo("Dataset Audit", "No JSONL records loaded.")
            return

        term_labels: Dict[str, Dict[str, int]] = {}
        for record in self.data:
            rec_text = record.get("text", "")
            for e in record.get("entities", []):
                s, end = e.get("start", 0), e.get("end", 0)
                label = e.get("label", "")
                if 0 <= s <= end <= len(rec_text) and label:
                    t = rec_text[s:end].lower()
                    if t not in term_labels:
                        term_labels[t] = {}
                    term_labels[t][label] = term_labels[t].get(label, 0) + 1

        conflicts = {t: counts for t, counts in term_labels.items() if len(counts) > 1}

        audit_win = tk.Toplevel(self.root)
        audit_win.title("Dataset Consistency Audit Report")
        audit_win.geometry("750x450")
        audit_win.transient(self.root)

        frame = ttk.Frame(audit_win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        if not conflicts:
            ttk.Label(
                frame,
                text="🎉 No label conflicts detected across all records!",
                font=("TkDefaultFont", 11, "bold"),
                foreground="#15803d"
            ).pack(pady=20)
            ttk.Label(
                frame,
                text=f"Analyzed {len(self.data):,} records with {len(term_labels):,} unique entity terms.",
                foreground="#64748b"
            ).pack()
            ttk.Button(frame, text="Close", command=audit_win.destroy).pack(pady=15)
            return

        ttk.Label(
            frame,
            text=f"⚠️ Found {len(conflicts)} term(s) with conflicting labels:",
            font=("TkDefaultFont", 10, "bold"),
            foreground="#b91c1c"
        ).pack(anchor="w", pady=(0, 8))

        cols = ("term", "breakdown", "suggested")
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        tree.heading("term", text="Entity Term")
        tree.heading("breakdown", text="Label Counts")
        tree.heading("suggested", text="Majority Label")
        tree.column("term", width=220)
        tree.column("breakdown", width=300)
        tree.column("suggested", width=160, anchor="center")

        for term, counts in sorted(conflicts.items(), key=lambda x: -sum(x[1].values())):
            breakdown_str = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in counts.items())
            maj_lbl = max(counts.items(), key=lambda x: x[1])[0]
            tree.insert("", "end", values=(term, breakdown_str, f"{maj_lbl} (auto)"))

        tree.pack(fill=tk.BOTH, expand=True)

        def _resolve_all_majority():
            resolved = 0
            for term, counts in conflicts.items():
                maj_lbl = max(counts.items(), key=lambda x: x[1])[0]
                self.sync_add_entity(term, maj_lbl)
                resolved += 1
            self.populate_entity_table()
            messagebox.showinfo(
                "Resolved",
                f"Synchronized {resolved} conflicting terms to their majority labels across the dataset.",
                parent=audit_win
            )
            audit_win.destroy()

        btn_bar = ttk.Frame(frame)
        btn_bar.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_bar, text="Resolve All to Majority Label", command=_resolve_all_majority).pack(side=tk.LEFT)
        ttk.Button(btn_bar, text="Close", command=audit_win.destroy).pack(side=tk.RIGHT)

    # ============================================================
    # ADD ENTITY
    # ============================================================

    def add_entity(self):
        if self.current_index is None:
            messagebox.showwarning(
                "No Record Selected",
                "Please select a sentence first."
            )
            return

        text = self.data[self.current_index]["text"]

        # Pre-fill with highlighted selection if present
        selected_text = ""
        try:
            if self.selected_text_box.tag_ranges(tk.SEL):
                selected_text = self.selected_text_box.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
        except Exception:
            pass

        dialog = EntityDialog(
            self.root,
            title="Add Entity",
            sentence=text,
            entity_text=selected_text,
            sync_default=self.sync_consistency.get()
        )

        self.root.wait_window(dialog)

        if not dialog.result:
            return

        entity_text = dialog.entity_text.strip()
        label = dialog.label
        sync_all = dialog.sync_all

        positions = self.find_all_occurrences(text, entity_text)
        if not positions:
            messagebox.showerror(
                "Text Not Found",
                f"The entity text '{entity_text}' was not found in the selected sentence.\n\n"
                "Please paste the exact text as it appears in this sentence."
            )
            return

        if sync_all:
            # Sync across entire dataset (including current record)
            added_count, relabeled_count, affected_records = self.sync_add_entity(
                entity_text, label
            )
            self.populate_entity_table()
            self.status_var.set(
                f"Added '{entity_text}' [{label}] — Synced consistency: {added_count} added, "
                f"{relabeled_count} relabeled across {affected_records} records."
            )
        else:
            # Single-record addition
            if len(positions) > 1:
                occurrence = OccurrenceDialog(
                    self.root,
                    sentence=text,
                    entity_text=entity_text,
                    positions=positions
                )
                self.root.wait_window(occurrence)

                if occurrence.selected_position is None:
                    return

                if occurrence.selected_position == "ALL":
                    target_positions = positions
                else:
                    target_positions = [occurrence.selected_position]
            else:
                target_positions = [positions[0]]

            added_here = 0
            for start in target_positions:
                end = start + len(entity_text)
                if not self.has_overlap(start, end):
                    self.data[self.current_index]["entities"].append({
                        "start": start,
                        "end": end,
                        "label": label
                    })
                    added_here += 1

            self.data[self.current_index]["entities"].sort(key=lambda x: x.get("start", 0))
            self.populate_entity_table()
            self.status_var.set(f"Added entity: '{entity_text}' ({added_here} instance(s) in current record)")

    # ============================================================
    # EDIT ENTITY
    # ============================================================

    def edit_entity(self):
        if self.current_index is None:
            messagebox.showwarning(
                "No Record Selected",
                "Please select a sentence first."
            )
            return

        selection = self.entity_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Entity Selected",
                "Please select an entity to edit."
            )
            return

        entity_index = int(selection[0])
        record = self.data[self.current_index]
        entity = record["entities"][entity_index]
        text = record["text"]

        old_start = entity["start"]
        old_end = entity["end"]
        old_text = text[old_start:old_end]
        old_label = entity.get("label", LABELS[0])

        dialog = EntityDialog(
            self.root,
            title="Edit Entity",
            sentence=text,
            entity_text=old_text,
            label=old_label,
            sync_default=self.sync_consistency.get()
        )

        self.root.wait_window(dialog)

        if not dialog.result:
            return

        new_text = dialog.entity_text.strip()
        new_label = dialog.label
        sync_all = dialog.sync_all

        if new_text == old_text and new_label == old_label:
            return

        if sync_all:
            relabeled_or_removed, added_count, affected_records = self.sync_edit_entity(
                old_text, old_label, new_text, new_label
            )
            self.populate_entity_table()
            if old_text.lower() == new_text.lower():
                self.status_var.set(
                    f"Relabeled '{old_text}' -> [{new_label}] — Synced across {affected_records} records "
                    f"({relabeled_or_removed} entities updated)."
                )
            else:
                self.status_var.set(
                    f"Edited '{old_text}' -> '{new_text}' [{new_label}] — Synced across {affected_records} records "
                    f"({relabeled_or_removed} old removed, {added_count} new added)."
                )
        else:
            positions = self.find_all_occurrences(text, new_text)
            if not positions:
                messagebox.showerror(
                    "Text Not Found",
                    f"The new entity text '{new_text}' was not found in the selected sentence.\n\n"
                    "Please paste the exact text as it appears in this sentence."
                )
                return

            if new_text == old_text:
                start = old_start
            elif len(positions) == 1:
                start = positions[0]
            else:
                occurrence = OccurrenceDialog(
                    self.root,
                    sentence=text,
                    entity_text=new_text,
                    positions=positions,
                    exclude_range=(old_start, old_end)
                )
                self.root.wait_window(occurrence)

                if occurrence.selected_position is None:
                    return

                if occurrence.selected_position == "ALL":
                    start = positions[0]
                else:
                    start = occurrence.selected_position

            end = start + len(new_text)

            if self.has_overlap(start, end, exclude_index=entity_index):
                messagebox.showerror(
                    "Overlapping Entity",
                    "The edited entity overlaps with another entity."
                )
                return

            entity["start"] = start
            entity["end"] = end
            entity["label"] = new_label

            self.data[self.current_index]["entities"].sort(key=lambda x: x.get("start", 0))
            self.populate_entity_table()
            self.status_var.set(f"Edited entity: '{new_text}' [{new_label}] (current record only)")

    # ============================================================
    # DELETE ENTITY
    # ============================================================

    def delete_entity(self):
        if self.current_index is None:
            messagebox.showwarning(
                "No Record Selected",
                "Please select a sentence first."
            )
            return

        selection = self.entity_tree.selection()
        if not selection:
            messagebox.showwarning(
                "No Entity Selected",
                "Please select an entity to delete."
            )
            return

        entity_index = int(selection[0])
        record = self.data[self.current_index]
        entity = record["entities"][entity_index]

        start = entity["start"]
        end = entity["end"]
        entity_text = record["text"][start:end]
        label = entity.get("label", "")

        norm_text = entity_text.strip().lower()
        other_matches = sum(
            1
            for r in self.data
            for e in r.get("entities", [])
            if r["text"][e["start"]:e["end"]].lower() == norm_text
        )
        records_with_match = sum(
            1
            for r in self.data
            if any(r["text"][e["start"]:e["end"]].lower() == norm_text for e in r.get("entities", []))
        )

        if self.sync_consistency.get() and other_matches > 1:
            res = messagebox.askyesnocancel(
                "Confirm Delete & Consistency Sync",
                f"Delete entity: '{entity_text}' [{label}]?\n\n"
                f"Found {other_matches} total occurrence(s) across {records_with_match} record(s).\n\n"
                f"• Click [Yes] to delete '{entity_text}' from ALL records in the file (Global Consistency Sync).\n"
                f"• Click [No] to delete from this current record ONLY.\n"
                f"• Click [Cancel] to abort."
            )
            if res is None:
                return
            elif res is True:
                deleted_count, affected_records = self.sync_delete_entity(entity_text)
                self.populate_entity_table()
                self.status_var.set(
                    f"Deleted entity '{entity_text}' — Synced: removed {deleted_count} occurrences across {affected_records} records."
                )
            else:
                del record["entities"][entity_index]
                self.populate_entity_table()
                self.status_var.set(f"Deleted entity '{entity_text}' from current record only.")
        else:
            confirm = messagebox.askyesno(
                "Confirm Delete",
                f"Are you sure you want to delete this entity?\n\n"
                f"Text: {entity_text}\n"
                f"Start: {start}\n"
                f"End: {end}\n"
                f"Label: {label}"
            )
            if not confirm:
                return

            if self.sync_consistency.get():
                deleted_count, affected_records = self.sync_delete_entity(entity_text)
                self.populate_entity_table()
                self.status_var.set(f"Deleted entity '{entity_text}' ({deleted_count} removed).")
            else:
                del record["entities"][entity_index]
                self.populate_entity_table()
                self.status_var.set(f"Deleted entity '{entity_text}' from current record.")

    # ============================================================
    # ENTITY VALIDATION
    # ============================================================

    def find_all_occurrences(
        self,
        text,
        substring
    ):
        """
        Return every start position where substring occurs (case-insensitive exact entity match).
        """
        if not text or not substring:
            return []

        pat = make_entity_pattern(substring)
        return [m.start() for m in pat.finditer(text)]

    def has_overlap(
        self,
        start,
        end,
        exclude_index=None
    ):
        """
        Checks whether [start, end) overlaps an existing entity in the current record.
        """
        if self.current_index is None:
            return False

        entities = self.data[
            self.current_index
        ].get("entities", [])

        for index, entity in enumerate(entities):
            if exclude_index is not None and index == exclude_index:
                continue

            existing_start = entity.get("start")
            existing_end = entity.get("end")

            if existing_start is not None and existing_end is not None:
                if start < existing_end and end > existing_start:
                    return True

        return False

    def add_record(self):
        dialog = NewRecordDialog(self.root)

        self.root.wait_window(dialog)

        if not dialog.result:
            return

        new_text = dialog.text_value.strip()

        if not new_text:
            messagebox.showwarning(
                "Invalid Text",
                "The text cannot be empty."
            )
            return

        # Add new record with no entities initially
        self.data.append({
            "text": new_text,
            "entities": []
        })

        # Refresh the top table
        self.populate_text_table()

        # Select the newly added record
        new_index = len(self.data) - 1
        self.current_index = new_index

        self.text_tree.selection_set(
            str(new_index)
        )

        self.text_tree.focus(
            str(new_index)
        )

        self.text_tree.see(
            str(new_index)
        )

        self.on_record_selected()

        self.status_var.set(
            f"Added new record #{new_index + 1}"
        )

class NewRecordDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Add New Record")
        self.geometry("700x300")
        self.resizable(True, True)

        self.result = False
        self.text_value = ""

        self.transient(parent)
        self.grab_set()

        # --------------------------------------------------------
        # Instructions
        # --------------------------------------------------------

        ttk.Label(
            self,
            text="Enter the text for the new JSONL record:"
        ).pack(
            anchor=tk.W,
            padx=15,
            pady=(15, 5)
        )

        # --------------------------------------------------------
        # Text input
        # --------------------------------------------------------

        text_frame = ttk.Frame(self)
        text_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=15
        )

        self.text_box = tk.Text(
            text_frame,
            wrap=tk.WORD,
            height=8
        )

        scrollbar = ttk.Scrollbar(
            text_frame,
            orient=tk.VERTICAL,
            command=self.text_box.yview
        )

        self.text_box.configure(
            yscrollcommand=scrollbar.set
        )

        self.text_box.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        text_frame.rowconfigure(
            0,
            weight=1
        )

        text_frame.columnconfigure(
            0,
            weight=1
        )

        # --------------------------------------------------------
        # Buttons
        # --------------------------------------------------------

        button_frame = ttk.Frame(self)
        button_frame.pack(
            fill=tk.X,
            padx=15,
            pady=15
        )

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel
        ).pack(
            side=tk.RIGHT,
            padx=(5, 0)
        )

        ttk.Button(
            button_frame,
            text="Add Record",
            command=self.accept
        ).pack(
            side=tk.RIGHT
        )

        self.text_box.focus_set()

        self.bind(
            "<Control-Return>",
            lambda event: self.accept()
        )

        self.bind(
            "<Escape>",
            lambda event: self.cancel()
        )

    def accept(self):
        text = self.text_box.get(
            "1.0",
            tk.END
        ).strip()

        if not text:
            messagebox.showwarning(
                "Invalid Text",
                "The text cannot be empty.",
                parent=self
            )
            return

        self.text_value = text
        self.result = True

        self.destroy()

    def cancel(self):
        self.result = False
        self.destroy()


# ================================================================
# ENTITY DIALOG
# ================================================================

class EntityDialog(tk.Toplevel):

    def __init__(
        self,
        parent,
        title,
        sentence,
        entity_text="",
        label=LABELS[0],
        sync_default=True
    ):
        super().__init__(parent)

        self.title(title)
        self.geometry("640x310")
        self.resizable(False, False)

        self.result = False
        self.entity_text = ""
        self.label = LABELS[0]
        self.sync_all = sync_default

        self.transient(parent)
        self.grab_set()

        # Sentence
        ttk.Label(
            self,
            text="Selected sentence:"
        ).pack(
            anchor=tk.W,
            padx=15,
            pady=(12, 4)
        )

        sentence_frame = ttk.Frame(self)
        sentence_frame.pack(
            fill=tk.X,
            padx=15
        )

        sentence_box = tk.Text(
            sentence_frame,
            height=3,
            wrap=tk.WORD,
            font=("DejaVu Sans", 9)
        )

        sentence_box.pack(
            fill=tk.X
        )

        sentence_box.insert(
            "1.0",
            sentence
        )

        sentence_box.config(
            state=tk.DISABLED
        )

        # Entity text
        ttk.Label(
            self,
            text="Entity text:"
        ).pack(
            anchor=tk.W,
            padx=15,
            pady=(10, 4)
        )

        self.entity_var = tk.StringVar(
            value=entity_text
        )

        entity_entry = ttk.Entry(
            self,
            textvariable=self.entity_var
        )

        entity_entry.pack(
            fill=tk.X,
            padx=15
        )

        # Label & Sync frame
        opts_frame = ttk.Frame(self)
        opts_frame.pack(
            fill=tk.X,
            padx=15,
            pady=10
        )

        ttk.Label(
            opts_frame,
            text="Label:"
        ).pack(
            side=tk.LEFT
        )

        self.label_var = tk.StringVar(
            value=label
        )

        label_dropdown = ttk.Combobox(
            opts_frame,
            textvariable=self.label_var,
            values=LABELS,
            state="readonly",
            width=18
        )

        label_dropdown.pack(
            side=tk.LEFT,
            padx=(8, 20)
        )

        self.sync_var = tk.BooleanVar(value=sync_default)
        sync_check = ttk.Checkbutton(
            opts_frame,
            text="Sync across entire file (case-insensitive consistency)",
            variable=self.sync_var
        )
        sync_check.pack(
            side=tk.LEFT
        )

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(
            side=tk.BOTTOM,
            fill=tk.X,
            padx=15,
            pady=12
        )

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel
        ).pack(
            side=tk.RIGHT,
            padx=(5, 0)
        )

        ttk.Button(
            button_frame,
            text="OK",
            command=self.accept
        ).pack(
            side=tk.RIGHT
        )

        entity_entry.focus_set()

        self.bind(
            "<Return>",
            lambda event: self.accept()
        )

        self.bind(
            "<Escape>",
            lambda event: self.cancel()
        )

    def accept(self):
        entity_text = self.entity_var.get().strip()
        label = self.label_var.get()

        if not entity_text:
            messagebox.showwarning(
                "Invalid Entity",
                "Entity text cannot be empty.",
                parent=self
            )
            return

        if label not in LABELS:
            messagebox.showwarning(
                "Invalid Label",
                "Please select a valid entity label.",
                parent=self
            )
            return

        self.entity_text = entity_text
        self.label = label
        self.sync_all = self.sync_var.get()
        self.result = True

        self.destroy()

    def cancel(self):
        self.result = False
        self.destroy()


# ================================================================
# OCCURRENCE SELECTION DIALOG
# ================================================================

class OccurrenceDialog(tk.Toplevel):

    def __init__(
        self,
        parent,
        sentence,
        entity_text,
        positions,
        exclude_range=None
    ):
        super().__init__(parent)

        self.title("Select Entity Occurrence")
        self.geometry("700x350")
        self.resizable(True, True)

        self.selected_position = None

        self.transient(parent)
        self.grab_set()

        ttk.Label(
            self,
            text=(
                f'The text "{entity_text}" occurs '
                f"{len(positions)} times.\n"
                "Select which occurrence should be annotated, or annotate all occurrences:"
            ),
            wraplength=650
        ).pack(
            anchor=tk.W,
            padx=15,
            pady=15
        )

        tree_frame = ttk.Frame(self)
        tree_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=15
        )

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("position", "context"),
            show="headings",
            selectmode="browse"
        )

        self.tree.heading(
            "position",
            text="Start"
        )

        self.tree.heading(
            "context",
            text="Context"
        )

        self.tree.column(
            "position",
            width=80,
            anchor=tk.CENTER
        )

        self.tree.column(
            "context",
            width=550,
            anchor=tk.W
        )

        scrollbar = ttk.Scrollbar(
            tree_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )

        self.tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.tree.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        for position in positions:

            # During editing, don't offer the old position
            # if another occurrence exists.
            if (
                exclude_range is not None
                and exclude_range[0] == position
            ):
                pass

            context_start = max(
                0,
                position - 40
            )

            context_end = min(
                len(sentence),
                position + len(entity_text) + 40
            )

            context = sentence[
                context_start:context_end
            ]

            context = context.replace(
                "\n",
                " "
            )

            self.tree.insert(
                "",
                tk.END,
                iid=str(position),
                values=(
                    position,
                    context
                )
            )

        button_frame = ttk.Frame(self)
        button_frame.pack(
            fill=tk.X,
            padx=15,
            pady=15
        )

        ttk.Button(
            button_frame,
            text="Annotate All Occurrences",
            command=self.accept_all
        ).pack(
            side=tk.LEFT,
            padx=(0, 5)
        )

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel
        ).pack(
            side=tk.RIGHT,
            padx=(5, 0)
        )

        ttk.Button(
            button_frame,
            text="Use Selected",
            command=self.accept
        ).pack(
            side=tk.RIGHT
        )

        self.tree.bind(
            "<Double-1>",
            lambda event: self.accept()
        )

    def accept_all(self):
        self.selected_position = "ALL"
        self.destroy()

    def accept(self):
        selection = self.tree.selection()

        if not selection:
            messagebox.showwarning(
                "No Occurrence Selected",
                "Please select an occurrence.",
                parent=self
            )
            return

        self.selected_position = int(
            selection[0]
        )

        self.destroy()

    def cancel(self):
        self.selected_position = None
        self.destroy()


# ================================================================
# MAIN
# ================================================================

def main():
    root = tk.Tk()

    # Use ttk's native theme
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = JSONLEntityAnnotator(root)

    root.mainloop()


if __name__ == "__main__":
    main()
