import tkinter as tk
from tkinter import ttk, messagebox
import json


class SpaCyAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("spaCy Training Data Annotator")
        self.root.geometry("850x650")
        self.root.minsize(700, 550)

        self.entities = []

        self.build_ui()

    def build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        # Text
        ttk.Label(main, text="Text").pack(anchor="w")

        self.text_box = tk.Text(main, height=7, wrap="word")
        self.text_box.pack(fill="x", pady=(4, 12))
        self.text_box.bind("<KeyRelease>", lambda e: self.update_output())

        # Annotation inputs
        annotation_frame = ttk.LabelFrame(main, text="Add Entity", padding=10)
        annotation_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(annotation_frame, text="Term:").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )

        self.term_var = tk.StringVar()
        self.term_entry = ttk.Entry(annotation_frame, textvariable=self.term_var)
        self.term_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(annotation_frame, text="Label:").grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )

        self.label_var = tk.StringVar(value="IT_TERM")
        self.label_combo = ttk.Combobox(
            annotation_frame,
            textvariable=self.label_var,
            values=["IT_TERM", "CLERICAL_TERM"],
            width=18,
        )
        self.label_combo.grid(row=0, column=3, padx=(0, 10))

        ttk.Button(
            annotation_frame,
            text="Add Entity",
            command=self.add_entity
        ).grid(row=0, column=4)

        annotation_frame.columnconfigure(1, weight=1)

        # Entity list
        ttk.Label(main, text="Entities").pack(anchor="w")

        list_frame = ttk.Frame(main)
        list_frame.pack(fill="both", expand=True, pady=(4, 12))

        columns = ("term", "label", "start", "end")

        self.entity_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            height=8,
        )

        self.entity_tree.heading("term", text="Term")
        self.entity_tree.heading("label", text="Label")
        self.entity_tree.heading("start", text="Start")
        self.entity_tree.heading("end", text="End")

        self.entity_tree.column("term", width=300)
        self.entity_tree.column("label", width=180)
        self.entity_tree.column("start", width=80, anchor="center")
        self.entity_tree.column("end", width=80, anchor="center")

        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.entity_tree.yview,
        )
        self.entity_tree.configure(yscrollcommand=scrollbar.set)

        self.entity_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Buttons
        button_frame = ttk.Frame(main)
        button_frame.pack(fill="x", pady=(0, 12))

        ttk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_selected
        ).pack(side="left")

        ttk.Button(
            button_frame,
            text="Clear Entities",
            command=self.clear_entities
        ).pack(side="left", padx=6)

        # Output
        ttk.Label(main, text="Output (JSON)").pack(anchor="w")

        self.output_box = tk.Text(
            main,
            height=7,
            wrap="word",
        )
        self.output_box.pack(fill="x", pady=(4, 8))

        output_buttons = ttk.Frame(main)
        output_buttons.pack(fill="x")

        ttk.Button(
            output_buttons,
            text="Copy Output",
            command=self.copy_output
        ).pack(side="left")

        ttk.Button(
            output_buttons,
            text="Clear Everything",
            command=self.clear_all
        ).pack(side="left", padx=6)

        self.update_output()

    def get_text(self):
        return self.text_box.get("1.0", "end-1c")

    def add_entity(self):
        text = self.get_text()
        term = self.term_var.get()
        label = self.label_var.get().strip()

        if not text:
            messagebox.showwarning("Missing Text", "Enter some text first.")
            return

        if not term:
            messagebox.showwarning("Missing Term", "Enter a term.")
            return

        if not label:
            messagebox.showwarning("Missing Label", "Enter a label.")
            return

        # Find the first occurrence that isn't already represented
        start = text.find(term)

        if start == -1:
            messagebox.showwarning(
                "Term Not Found",
                f'Could not find "{term}" in the text.'
            )
            return

        end = start + len(term)

        # Check for an exact duplicate
        for entity in self.entities:
            if (
                entity["start"] == start
                and entity["end"] == end
                and entity["label"] == label
            ):
                messagebox.showwarning(
                    "Duplicate Entity",
                    "This entity has already been added."
                )
                return

        # Check overlap
        for entity in self.entities:
            if start < entity["end"] and end > entity["start"]:
                messagebox.showwarning(
                    "Overlapping Entity",
                    f'The term overlaps with "{entity["term"]}".'
                )
                return

        self.entities.append({
            "term": term,
            "start": start,
            "end": end,
            "label": label,
        })

        self.entities.sort(key=lambda x: x["start"])

        self.term_var.set("")
        self.refresh_entity_list()
        self.update_output()

    def remove_selected(self):
        selected = self.entity_tree.selection()

        if not selected:
            return

        indexes = sorted(
            [self.entity_tree.index(item) for item in selected],
            reverse=True
        )

        for index in indexes:
            del self.entities[index]

        self.refresh_entity_list()
        self.update_output()

    def clear_entities(self):
        self.entities.clear()
        self.refresh_entity_list()
        self.update_output()

    def clear_all(self):
        self.text_box.delete("1.0", "end")
        self.term_var.set("")
        self.entities.clear()
        self.refresh_entity_list()
        self.update_output()

    def refresh_entity_list(self):
        for item in self.entity_tree.get_children():
            self.entity_tree.delete(item)

        for entity in self.entities:
            self.entity_tree.insert(
                "",
                "end",
                values=(
                    entity["term"],
                    entity["label"],
                    entity["start"],
                    entity["end"],
                ),
            )

    def generate_data(self):
        text = self.get_text()

        return {
            "text": text,
            "entities": [
                {
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": entity["label"],
                }
                for entity in self.entities
            ],
        }

    def update_output(self):
        data = self.generate_data()

        output = json.dumps(
            data,
            ensure_ascii=False,
            separators=(", ", ": "),
        )

        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", output)

    def copy_output(self):
        output = self.output_box.get("1.0", "end-1c")

        self.root.clipboard_clear()
        self.root.clipboard_append(output)
        self.root.update()

        messagebox.showinfo("Copied", "Training data copied to clipboard.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpaCyAnnotationApp(root)
    root.mainloop()
