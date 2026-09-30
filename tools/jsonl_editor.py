import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import gc


LABELS = ["IT_TERM", "CLERICAL_TERM"]


class JSONLEntityAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("JSONL Entity Annotator")
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)

        self.data = []
        self.current_index = None
        self.file_path = None

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

        ttk.Label(
            sentence_frame,
            text="Selected text:"
        ).pack(
            anchor=tk.W
        )

        self.selected_text_box = tk.Text(
            sentence_frame,
            height=3,
            wrap=tk.WORD
        )

        self.selected_text_box.pack(
            fill=tk.X,
            pady=(5, 0)
        )

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

    def clear_selected_text(self):
        self.selected_text_box.config(state=tk.NORMAL)
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

    def populate_entity_table(self):
        self.clear_entity_table()

        if self.current_index is None:
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

        text = self.data[
            self.current_index
        ]["text"]

        dialog = EntityDialog(
            self.root,
            title="Add Entity",
            sentence=text
        )

        self.root.wait_window(dialog)

        if not dialog.result:
            return

        entity_text = dialog.entity_text
        label = dialog.label

        positions = self.find_all_occurrences(
            text,
            entity_text
        )

        if not positions:
            messagebox.showerror(
                "Text Not Found",
                "The entity text was not found in the selected "
                "sentence.\n\n"
                "Please paste the exact text as it appears."
            )
            return

        # If there are multiple matches, let the user choose.
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

            start = occurrence.selected_position

        else:
            start = positions[0]

        end = start + len(entity_text)

        # Check overlap
        if self.has_overlap(
            start,
            end
        ):
            messagebox.showerror(
                "Overlapping Entity",
                "This entity overlaps with an existing entity."
            )
            return

        self.data[
            self.current_index
        ]["entities"].append(
            {
                "start": start,
                "end": end,
                "label": label
            }
        )

        self.populate_entity_table()

        self.status_var.set(
            f"Added entity: '{entity_text}'"
        )

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

        record = self.data[
            self.current_index
        ]

        entity = record["entities"][entity_index]

        text = record["text"]

        old_start = entity["start"]
        old_end = entity["end"]

        old_text = text[
            old_start:old_end
        ]

        dialog = EntityDialog(
            self.root,
            title="Edit Entity",
            sentence=text,
            entity_text=old_text,
            label=entity.get("label", LABELS[0])
        )

        self.root.wait_window(dialog)

        if not dialog.result:
            return

        new_entity_text = dialog.entity_text
        new_label = dialog.label

        positions = self.find_all_occurrences(
            text,
            new_entity_text
        )

        if not positions:
            messagebox.showerror(
                "Text Not Found",
                "The new entity text was not found in the "
                "selected sentence.\n\n"
                "Please paste the exact text as it appears."
            )
            return

        # If the new text is the same as the old text,
        # prefer keeping the existing position.
        if new_entity_text == old_text:
            start = old_start

        elif len(positions) == 1:
            start = positions[0]

        else:
            occurrence = OccurrenceDialog(
                self.root,
                sentence=text,
                entity_text=new_entity_text,
                positions=positions,
                exclude_range=(old_start, old_end)
            )

            self.root.wait_window(occurrence)

            if occurrence.selected_position is None:
                return

            start = occurrence.selected_position

        end = start + len(new_entity_text)

        # Temporarily exclude the entity being edited
        if self.has_overlap(
            start,
            end,
            exclude_index=entity_index
        ):
            messagebox.showerror(
                "Overlapping Entity",
                "The edited entity overlaps with another entity."
            )
            return

        entity["start"] = start
        entity["end"] = end
        entity["label"] = new_label

        self.populate_entity_table()

        self.status_var.set(
            f"Edited entity: '{new_entity_text}'"
        )

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

        record = self.data[
            self.current_index
        ]

        entity = record["entities"][entity_index]

        start = entity["start"]
        end = entity["end"]

        entity_text = record["text"][
            start:end
        ]

        label = entity.get(
            "label",
            ""
        )

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

        del record["entities"][entity_index]

        self.populate_entity_table()

        self.status_var.set(
            f"Deleted entity: '{entity_text}'"
        )

    # ============================================================
    # ENTITY VALIDATION
    # ============================================================

    def find_all_occurrences(
        self,
        text,
        substring
    ):
        """
        Return every start position where substring occurs.
        """

        positions = []

        if not substring:
            return positions

        start = 0

        while True:
            position = text.find(
                substring,
                start
            )

            if position == -1:
                break

            positions.append(position)

            # +1 allows overlapping textual occurrences
            start = position + 1

        return positions

    def has_overlap(
        self,
        start,
        end,
        exclude_index=None
    ):
        """
        Checks whether [start, end) overlaps an existing entity.
        """

        if self.current_index is None:
            return False

        entities = self.data[
            self.current_index
        ]["entities"]

        for index, entity in enumerate(entities):

            if exclude_index is not None:
                if index == exclude_index:
                    continue

            existing_start = entity["start"]
            existing_end = entity["end"]

            # Standard interval-overlap test
            if (
                start < existing_end
                and end > existing_start
            ):
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
        label=LABELS[0]
    ):
        super().__init__(parent)

        self.title(title)
        self.geometry("600x260")
        self.resizable(False, False)

        self.result = False
        self.entity_text = ""
        self.label = LABELS[0]

        self.transient(parent)
        self.grab_set()

        # Sentence
        ttk.Label(
            self,
            text="Selected sentence:"
        ).pack(
            anchor=tk.W,
            padx=15,
            pady=(15, 5)
        )

        sentence_frame = ttk.Frame(self)
        sentence_frame.pack(
            fill=tk.X,
            padx=15
        )

        sentence_box = tk.Text(
            sentence_frame,
            height=4,
            wrap=tk.WORD
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
            pady=(12, 5)
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

        # Label
        label_frame = ttk.Frame(self)
        label_frame.pack(
            fill=tk.X,
            padx=15,
            pady=12
        )

        ttk.Label(
            label_frame,
            text="Label:"
        ).pack(
            side=tk.LEFT
        )

        self.label_var = tk.StringVar(
            value=label
        )

        label_dropdown = ttk.Combobox(
            label_frame,
            textvariable=self.label_var,
            values=LABELS,
            state="readonly",
            width=20
        )

        label_dropdown.pack(
            side=tk.LEFT,
            padx=(10, 0)
        )

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(
            side=tk.BOTTOM,
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
        entity_text = self.entity_var.get()
        label = self.label_var.get()

        if not entity_text.strip():
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
                "Select which occurrence should be annotated:"
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
