"""PaperCore GUI - Tkinter interface for academic PDF conversion."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from papercore import PaperConverter, SectionClassifier


class TextHandler(logging.Handler):
    """Routes logging output to a tkinter Text widget (thread-safe)."""

    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text = text_widget

    def emit(self, record: logging.LogRecord):
        msg = self.format(record) + "\n"
        self.text.after(0, self._append, msg)

    def _append(self, msg: str):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, msg)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)


class PaperCoreGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PaperCore")
        self.root.geometry("700x520")
        self.root.minsize(560, 420)
        self._build_ui()
        self._setup_logging()
        self._converting = False

    def _build_ui(self):
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        pad = {"padx": 10, "pady": 4}

        # -- Title --
        title = tk.Label(
            root,
            text="PaperCore - Academic PDF Converter",
            font=("Segoe UI", 13, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 2))

        subtitle = tk.Label(
            root,
            text="Three-Zone Strategy: Metadata / Full Retention / Smart Compression",
            font=("Segoe UI", 9),
            fg="#666",
        )
        subtitle.grid(row=0, column=0, sticky="e", padx=10, pady=(10, 2))

        # -- Input / Output frame --
        io_frame = ttk.LabelFrame(root, text="Paths", padding=8)
        io_frame.grid(row=1, column=0, sticky="ew", **pad)
        io_frame.columnconfigure(1, weight=1)

        ttk.Label(io_frame, text="Input:").grid(row=0, column=0, sticky="w")
        self.input_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.input_var).grid(
            row=0, column=1, sticky="ew", padx=(6, 4)
        )

        btn_frame_in = ttk.Frame(io_frame)
        btn_frame_in.grid(row=0, column=2)
        ttk.Button(
            btn_frame_in, text="File", width=5, command=self._browse_file
        ).pack(side=tk.LEFT, padx=1)
        ttk.Button(
            btn_frame_in, text="Folder", width=6, command=self._browse_input_dir
        ).pack(side=tk.LEFT, padx=1)

        ttk.Label(io_frame, text="Output:").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self.output_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=(6, 4), pady=(4, 0)
        )
        ttk.Button(
            io_frame, text="Browse", width=12, command=self._browse_output_dir
        ).grid(row=1, column=2, pady=(4, 0))

        # -- Options frame --
        opt_frame = ttk.Frame(root)
        opt_frame.grid(row=2, column=0, sticky="ew", **pad)

        self.no_compress_var = tk.BooleanVar()
        ttk.Checkbutton(
            opt_frame,
            text="Disable Zone C compression (keep all text)",
            variable=self.no_compress_var,
        ).pack(side=tk.LEFT)

        self.convert_btn = ttk.Button(
            opt_frame, text="Convert", command=self._on_convert
        )
        self.convert_btn.pack(side=tk.RIGHT, padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            opt_frame, textvariable=self.status_var, foreground="#888"
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # -- Log area --
        log_frame = ttk.LabelFrame(root, text="Log", padding=4)
        log_frame.grid(row=3, column=0, sticky="nsew", **pad)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _setup_logging(self):
        handler = TextHandler(self.log_text)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        # Attach to papercore logger
        pc_logger = logging.getLogger("papercore")
        pc_logger.setLevel(logging.INFO)
        pc_logger.addHandler(handler)
        # Also capture root for docling messages
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        root_logger.addHandler(handler)

    # -- Browse callbacks ---------------------------------------------------

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _browse_input_dir(self):
        path = filedialog.askdirectory(title="Select folder containing PDFs")
        if path:
            self.input_var.set(path)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    # -- Conversion ---------------------------------------------------------

    def _on_convert(self):
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showwarning("PaperCore", "Please select an input file or folder.")
            return

        input_path = Path(input_path)
        if not input_path.exists():
            messagebox.showerror("PaperCore", f"Path does not exist:\n{input_path}")
            return

        output_str = self.output_var.get().strip()
        output_path = Path(output_str) if output_str else None

        no_compress = self.no_compress_var.get()

        # Clear log
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

        self._set_converting(True)

        thread = threading.Thread(
            target=self._run_conversion,
            args=(input_path, output_path, no_compress),
            daemon=True,
        )
        thread.start()

    def _run_conversion(
        self, input_path: Path, output_path: Path | None, no_compress: bool
    ):
        pc_logger = logging.getLogger("papercore")
        try:
            converter = PaperConverter()
            if no_compress:
                converter._classifier = SectionClassifier(zone_c_patterns={})

            if input_path.is_file() and input_path.suffix.lower() == ".pdf":
                out_dir = output_path or input_path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                md_path = out_dir / f"{input_path.stem}.md"
                markdown = converter.convert_single(input_path)
                md_path.write_text(markdown, encoding="utf-8")
                pc_logger.info(f"Saved: {md_path}")
            elif input_path.is_dir():
                converter.convert_folder(input_path, output_path)
            else:
                pc_logger.error(f"Input must be a PDF file or folder: {input_path}")

            self.root.after(0, self._on_done, None)
        except Exception as e:
            pc_logger.error(f"Conversion failed: {e}")
            self.root.after(0, self._on_done, str(e))

    def _on_done(self, error: str | None):
        self._set_converting(False)
        if error:
            self.status_var.set("Failed")
            messagebox.showerror("PaperCore", f"Conversion failed:\n{error}")
        else:
            self.status_var.set("Done")

    def _set_converting(self, active: bool):
        self._converting = active
        if active:
            self.convert_btn.configure(state=tk.DISABLED)
            self.status_var.set("Converting...")
        else:
            self.convert_btn.configure(state=tk.NORMAL)

    def run(self):
        self.root.mainloop()


def main():
    app = PaperCoreGUI()
    app.run()


if __name__ == "__main__":
    main()
