"""
ODERPQC – GUI for Linear Code + ARQ Simulator based on Core_new.py

Place this file in the same folder as Core_new.py and run:
    python GUI_new.py
"""

import threading
import traceback
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

# Optional: matplotlib for plots in the "Grid Sweep" tab
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import Core_channel as core  # Requires galois, sklearn, etc.


class LinearCodeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.configure(bg="#ffffff")
        self.title(" 🔒 PQCoder")
        self.geometry("1150x750")
        self.minsize(950, 650)

        # State: current code / decoder / k_tag
        self.code = None
        self.decoder = None
        self.k_tag = None

        # State for latest results (for Clear / CSV export / re-plot)
        self.grid_last_results: list[dict] = []
        self.batch_last_per_msg: list[dict] = []

        # Grid-sweep background execution state
        self.grid_worker_thread: threading.Thread | None = None
        self.grid_stop_requested: bool = False

        # Grid-sweep result window (graph-only popup)
        self.grid_results_window: tk.Toplevel | None = None
        self.grid_results_fig = None
        self.grid_results_ax = None
        self.grid_results_canvas = None
        self.grid_point_info_var: tk.StringVar | None = None

        # Manual plaintext storage (single + batch)
        self.single_manual_plaintext_bits: str | None = None
        self.batch_manual_plaintexts_bits: list[str] = []



        self._create_style()
        self._build_main_layout()

        # Initialize with default code
        self._on_code_changed()

    # ---------------------------------------------------------
    # Style
    # ---------------------------------------------------------
    def _create_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_bg = "#ffffff"
        header_bg = "#e3e7f3"

        # --- frames ---
        style.configure("TFrame", background=base_bg)
        style.configure("Main.TFrame", background=base_bg)
        style.configure("Header.TFrame", background=header_bg)
        style.configure("Card.TFrame", background=base_bg, relief="groove", borderwidth=1)
        # *** ADD THIS (you use White.TFrame but never defined it) ***
        style.configure("White.TFrame", background=base_bg)

        # --- labels ---
        style.configure("TLabel", background=base_bg, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=header_bg, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=header_bg, font=("Segoe UI", 15, "bold"))
        style.configure("Section.TLabel", background=base_bg, font=("Segoe UI", 11, "bold"))
        style.configure(
            "SectionEmoji.TLabel",
            background=base_bg,
            font=("Segoe UI Emoji", 11, "bold"),
        )

        # --- buttons ---
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map(
            "Accent.TButton",
            foreground=[("active", "#ffffff")],
            background=[("!disabled", "#4a6cf7"), ("pressed", "#3a57c4")],
        )

        # --- radiobuttons / checkbuttons: force white everywhere ---
        style.configure("TRadiobutton", background=base_bg)
        style.map(
            "TRadiobutton",
            background=[("active", base_bg), ("selected", base_bg)]
        )

        style.configure("TCheckbutton", background=base_bg)
        style.map(
            "TCheckbutton",
            background=[("active", base_bg), ("selected", base_bg)]
        )
        style = ttk.Style()

        # --- Treeview base ---
        style.configure(
            "Custom.Treeview",
            background="white",
            foreground="black",
            rowheight=26,
            fieldbackground="white",
            bordercolor="#d0d0d0",
            borderwidth=1,
            relief="flat",
        )

        # --- Header style ---
        style.configure(
            "Custom.Treeview.Heading",
            background="#f2f2f2",
            foreground="black",
            font=("Segoe UI", 10, "bold"),
            bordercolor="#d0d0d0",
            borderwidth=1,
        )

        style.map(
            "Custom.Treeview.Heading",
            background=[("active", "#e6e6e6")]
        )

        # --- Row hover effect ---
        style.map(
            "Custom.Treeview",
            background=[("selected", "#cce0ff")]
        )
        style.configure(
            "Bit0.TLabel",
            background="#f2f4f7",  # אפור בהיר מאוד
            foreground="#000000",
            font=("Consolas", 10),
            padding=1,
        )
        # --- G' Matrix style ---
        style.configure(
            "Bit1.TLabel",
            background="#d9dde3",  # מעט כהה יותר
            foreground="#000000",
            font=("Consolas", 10),
            padding=1,
        )

        style.map("Bit0.TLabel", background=[("active", "#f2f4f7")])
        style.map("Bit1.TLabel", background=[("active", "#d9dde3")])

        # Notebook, Treeview etc...

    # ---------------------------------------------------------
    # Main layout
    # ---------------------------------------------------------
    def _build_main_layout(self):
        # Top frame: title + code selection + global actions
        top = ttk.Frame(self, style="Header.TFrame", padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        title = ttk.Label(top, text="🔐  PQCoder – Linear Code Simulator", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")

        # Global "Clear All" button
        clear_all_btn = ttk.Button(
            top,
            text="Clear All",
            style="Accent.TButton",
            command=self._clear_all_results,
        )
        clear_all_btn.grid(row=0, column=1, sticky="e", padx=(20, 0))

        # Code selection
        code_frame = ttk.Frame(top, style="Header.TFrame")
        code_frame.grid(row=1, column=0, columnspan=2, pady=(8, 0), sticky="we")
        code_frame.columnconfigure(4, weight=1)

        ttk.Label(code_frame, text="Code:", style="Header.TLabel").grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )

        self.code_name_var = tk.StringVar()
        code_names = list(core.LinearCode_dict.keys())
        if not code_names:
            code_names = ["(no codes in LinearCode_dict)"]
        self.code_name_var.set(code_names[0])

        self.code_combo = ttk.Combobox(
            code_frame,
            textvariable=self.code_name_var,
            values=code_names,
            state="readonly",
            width=28,
        )
        self.code_combo.grid(row=0, column=1, padx=(0, 15), sticky="w")
        self.code_combo.bind("<<ComboboxSelected>>", lambda e: self._on_code_changed())

        self.code_info_label = ttk.Label(code_frame, text="", style="Header.TLabel")
        self.code_info_label.grid(row=0, column=2, padx=(0, 10), sticky="w")

        self.k_tag_label = ttk.Label(code_frame, text="", style="Header.TLabel")
        self.k_tag_label.grid(row=0, column=3, padx=(0, 10), sticky="w")

        # Button to display public key matrix G_tag
        display_matrix_btn = ttk.Button(
            code_frame,
            text="Display Public Key Matrix",
            command=self._show_public_key_matrix,
        )
        display_matrix_btn.grid(row=0, column=4, padx=(10, 0), sticky="e")

        # Note about BCH
        self.note_label = ttk.Label(
            top,
            style="Header.TLabel",
        )
        self.note_label.grid(row=2, column=0, columnspan=2, pady=(4, 0), sticky="w")

        # Main notebook
        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.single_tab = ttk.Frame(notebook, style="TFrame", padding=10)
        self.batch_tab = ttk.Frame(notebook, style="TFrame", padding=10)
        self.grid_tab = ttk.Frame(notebook, style="TFrame", padding=10)

        notebook.add(self.single_tab, text="Single Run")
        notebook.add(self.batch_tab, text="Batch Simulation")
        notebook.add(self.grid_tab, text="Grid Sweep")

        self._build_single_tab()
        self._build_batch_tab()
        self._build_grid_tab()

        # Mapping for grid X/Y selection – set after tabs are built
        self._init_grid_axis_maps()

    # ---------------------------------------------------------
    # Code selection logic
    # ---------------------------------------------------------
    def _on_code_changed(self):
        name = self.code_name_var.get()
        if name not in core.LinearCode_dict:
            messagebox.showerror("Error", f"Code '{name}' not found in LinearCode_dict.")
            return

        try:
            A, S, P = core.LinearCode_dict[name]
            self.code = core.LinearCode(A, S, P, name=name)
            self.decoder = core.Decoder(self.code)
            self.k_tag = self.code.k - core.CRC.generator_degree
            if self.k_tag <= 0:
                raise ValueError(
                    f"k_tag <= 0. Message is too short for CRC: k={self.code.k}, "
                    f"CRC length={core.CRC.generator_degree}"
                )

            self.code_info_label.config(
                text=f"n={self.code.n}, k={self.code.k}, d_min={self.code.d_min}, t={self.code.max_errors_num}"
            )
            self.k_tag_label.config(text=f"Message length = {self.k_tag} bits")

            # Update hints that depend on k_tag
            if hasattr(self, "single_plaintext_hint_label") and self.single_plaintext_hint_label is not None:
                self.single_plaintext_hint_label.config(
                    text=f"Manual plaintext bits (length = {self.k_tag}):"
                )
            if hasattr(self, "batch_plaintext_hint_label") and self.batch_plaintext_hint_label is not None:
                self.batch_plaintext_hint_label.config(
                    text=(
                            "random plaintexts will be generated"
                    )
                )

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error while building code", str(e))

    def _show_public_key_matrix(self):
        """
        Show the public generator matrix G_tag in a nice tiled layout:
        each bit is a small square cell, 0/1 distinguished by background color.
        """

        if self.code is None:
            messagebox.showerror("Error", "Code is not initialized.")
            return

        G_tag = self.code.G_tag.view(np.ndarray).astype(int)
        rows, cols = G_tag.shape

        win = tk.Toplevel(self)
        win.title("🔑 Public generator matrix")
        win.geometry("900x600")

        # ---- Top container ----
        top = ttk.Frame(win, style="Main.TFrame", padding=10)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(
            top,
            text=f"🔑Public generator matrix (shape = {rows} x {cols})",
            style="SectionEmoji.TLabel",  # or Section.TLabel if you prefer
        ).pack(anchor="w", pady=(0, 8))

        # ---- Scrollable matrix area (Canvas + inner frame) ----
        matrix_container = ttk.Frame(top, style="Card.TFrame")
        matrix_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            matrix_container,
            borderwidth=0,
            highlightthickness=0,
            bg="#ffffff",
        )
        vscroll = ttk.Scrollbar(
            matrix_container, orient="vertical", command=canvas.yview
        )
        hscroll = ttk.Scrollbar(
            matrix_container, orient="horizontal", command=canvas.xview
        )
        canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")

        matrix_container.rowconfigure(0, weight=1)
        matrix_container.columnconfigure(0, weight=1)

        inner = ttk.Frame(canvas, style="White.TFrame")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(event):
            # Update scroll region when content size changes
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(event):
            # Make inner frame follow the canvas width (nice resizing)
            canvas.itemconfigure(inner_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_config)

        # ---- Populate matrix cells ----
        for r in range(rows):
            for c in range(cols):
                bit = int(G_tag[r, c])
                style_name = "Bit1.TLabel" if bit == 1 else "Bit0.TLabel"

                ttk.Label(
                    inner,
                    text=str(bit),
                    style=style_name,
                    anchor="center",
                    width=2,
                ).grid(
                    row=r,
                    column=c,
                    padx=(0, 1),
                    pady=(0, 1),
                )

    # ---------------------------------------------------------
    # Single Run Tab
    # ---------------------------------------------------------
    def _build_single_tab(self):
        container = ttk.Frame(self.single_tab, style="Main.TFrame", padding=10)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ===== LEFT: parameters card =====
        left = ttk.Frame(container, style="Card.TFrame", padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left, text="Single Run Parameters", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Label(left, text="Channel error probability per bit (0–1):").grid(
            row=1, column=0, sticky="w"
        )
        self.single_p_var = tk.StringVar(value="0.05")
        ttk.Entry(left, textvariable=self.single_p_var, width=10).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(left, text="Number of deliberate errors:").grid(
            row=2, column=0, sticky="w", pady=(5, 0)
        )
        self.single_delib_var = tk.StringVar(value="1")
        ttk.Entry(left, textvariable=self.single_delib_var, width=10).grid(
            row=2, column=1, sticky="w", pady=(5, 0)
        )

        # Plaintext source
        ttk.Label(left, text="Plaintext source:").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        self.single_msg_mode = tk.StringVar(value="random")
        rb_random = ttk.Radiobutton(
            left,
            text="Random plaintext",
            value="random",
            variable=self.single_msg_mode,
            command=self._update_single_plaintext_state,
        )
        rb_manual = ttk.Radiobutton(
            left,
            text="Manual plaintext",
            value="manual",
            variable=self.single_msg_mode,
            command=self._update_single_plaintext_state,
        )
        rb_random.grid(row=4, column=0, columnspan=3, sticky="w")
        rb_manual.grid(row=5, column=0, columnspan=3, sticky="w")

        # Summary + Message Insert button (no direct entry box)
        self.single_plaintext_summary = ttk.Label(
            left,
            text="Random plaintext will be generated (length = k_tag).",
        )
        self.single_plaintext_summary.grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        self.single_insert_btn = ttk.Button(
            left,
            text=" Insert Message",
            command=self._open_single_plaintext_dialog,
        )
        self.single_insert_btn.grid(row=7, column=0, columnspan=3, sticky="w", pady=(3, 0))

        # Run + Clear buttons
        btn_frame = ttk.Frame(left, style="Main.TFrame")
        btn_frame.grid(row=8, column=0, columnspan=3, sticky="we", pady=(10, 0))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        run_btn = ttk.Button(
            btn_frame,
            text="Run Single Example",
            style="Accent.TButton",
            command=self._run_single_example_clicked,
        )
        run_btn.grid(row=0, column=0, sticky="we", padx=(0, 5))

        clear_btn = ttk.Button(
            btn_frame,
            text="Clear",
            command=self._clear_single_tab,
        )
        clear_btn.grid(row=0, column=1, sticky="we", padx=(5, 0))

        # ===== RIGHT: nice cards instead of ScrolledText =====
        right = ttk.Frame(container, style="White.TFrame", padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(
            right,
            text="Single Run – Detailed Report",
            style="Section.TLabel"
        ).pack(side=tk.TOP, anchor="w", pady=(0, 5))

        # Scrollable area (Canvas + inner frame) to host the cards
        self.single_output_container = ttk.Frame(right, style="White.TFrame")
        self.single_output_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.single_output_canvas = tk.Canvas(
            self.single_output_container,
            borderwidth=0,
            highlightthickness=0,
            bg="#ffffff",
        )
        vscroll = ttk.Scrollbar(
            self.single_output_container,
            orient="vertical",
            command=self.single_output_canvas.yview,
        )
        self.single_output_canvas.configure(yscrollcommand=vscroll.set)

        self.single_output_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Inner frame that will actually hold the cards
        self.single_output_inner = ttk.Frame(self.single_output_canvas, style="White.TFrame")

        self.single_output_inner_id = self.single_output_canvas.create_window(
            (0, 0),
            window=self.single_output_inner,
            anchor="nw"
            )

        # Update scroll region when content size changes
        def _on_frame_config(event):
            self.single_output_canvas.configure(
                scrollregion=self.single_output_canvas.bbox("all")
            )

        self.single_output_inner.bind("<Configure>", _on_frame_config)

        def _on_canvas_config(event):
            self.single_output_canvas.itemconfigure(
                self.single_output_inner_id,
                width=event.width
            )

        self.single_output_canvas.bind("<Configure>", _on_canvas_config)

        # Initial placeholder
        placeholder = ttk.Label(
            self.single_output_inner,
            text="Run a single example to see a detailed report here.",
            style="TLabel",
            justify="left",
        )
        placeholder.pack(anchor="w", pady=10)

        # Initial state for the Message Insert button / label
        self._update_single_plaintext_state()

    def _open_single_plaintext_dialog(self):
        """
        Dialog for entering a single plaintext bit string (manual mode).
        Now styled like the rest of the GUI and with an OK button that
        applies the message and closes the dialog.
        """
        if self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized (k_tag is unknown).")
            return

        win = tk.Toplevel(self)
        win.title("Manual plaintext for Single Run")
        win.transient(self)
        win.grab_set()

        # Card-style frame
        main = ttk.Frame(win, style="Card.TFrame", padding=10)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(
            main,
            text="Manual plaintext for Single Run",
            style="Section.TLabel",
        ).pack(side=tk.TOP, anchor="w", pady=(0, 6))

        ttk.Label(
            main,
            text=(
                f"Enter a binary plaintext of length {self.k_tag} bits.\n"
                "Only characters '0' and '1' are allowed."
            ),
            justify="left",
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        entry_var = tk.StringVar(value=self.single_manual_plaintext_bits or "")
        entry = ttk.Entry(main, textvariable=entry_var, width=max(32, self.k_tag))
        entry.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        entry.focus_set()

        # Bit-count label
        bit_count_var = tk.StringVar()
        bit_count_label = ttk.Label(main, textvariable=bit_count_var)
        bit_count_label.pack(side=tk.TOP, anchor="w", pady=(0, 8))

        # Bottom button row
        btn_frame = ttk.Frame(main, style="Main.TFrame")
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        # We create OK now so inner functions can enable/disable it
        ok_btn = ttk.Button(btn_frame, text="OK")
        ok_btn.pack(side=tk.RIGHT, padx=(5, 0))
        ok_btn.state(["disabled"])  # enabled only when input is valid

        def _validate_bits(bits_str: str) -> bool:
            """Return True if bits_str is exactly k_tag bits of 0/1."""
            if len(bits_str) != self.k_tag:
                return False
            if any(c not in "01" for c in bits_str):
                return False
            return True

        def update_bit_count(*_):
            bits_str = entry_var.get().replace(" ", "")
            bit_count_var.set(f"Bits entered: {len(bits_str)} / {self.k_tag}")
            # Enable OK only if the bits are valid
            if _validate_bits(bits_str):
                ok_btn.state(["!disabled"])
            else:
                ok_btn.state(["disabled"])

        entry_var.trace_add("write", update_bit_count)
        update_bit_count()

        def _store_plaintext(bits_str: str) -> bool:
            """Validate and store plaintext. Return True on success."""
            bits_str = bits_str.replace(" ", "")
            if not _validate_bits(bits_str):
                # Give a more detailed error message:
                if len(bits_str) != self.k_tag:
                    messagebox.showerror(
                        "Plaintext length error",
                        f"Plaintext must have exactly {self.k_tag} bits.",
                        parent=win,
                    )
                elif any(c not in "01" for c in bits_str):
                    messagebox.showerror(
                        "Plaintext error",
                        "Plaintext must contain only '0' and '1'.",
                        parent=win,
                    )
                return False

            self.single_manual_plaintext_bits = bits_str
            self._update_single_plaintext_state()
            return True

        def on_apply():
            bits_str = entry_var.get()
            if _store_plaintext(bits_str):
                # Keep dialog open – user can still modify and press OK later
                update_bit_count()

        def on_ok():
            bits_str = entry_var.get()
            if _store_plaintext(bits_str):
                win.destroy()

        def on_clear():
            entry_var.set("")
            update_bit_count()

        # Buttons: Clear | Apply | OK
        clear_btn = ttk.Button(btn_frame, text="Clear", command=on_clear)
        clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        apply_btn = ttk.Button(btn_frame, text="Apply", command=on_apply)
        apply_btn.pack(side=tk.LEFT)

        ok_btn.configure(command=on_ok)

        def on_close():
            # Just close; no warning needed – this is a single message
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def _update_single_plaintext_state(self):
        """
        Enable / disable the 'Message Insert' button and update the summary label
        according to the plaintext mode and whether a manual message is stored.
        """
        mode = self.single_msg_mode.get()
        if mode == "manual":
            # Enable insert button
            self.single_insert_btn.configure(state="normal")
            if self.single_manual_plaintext_bits:
                n_bits = len(self.single_manual_plaintext_bits)
                self.single_plaintext_summary.configure(
                    text=f"Manual plaintext stored ({n_bits} bits). "
                         f"Click 'Message Insert...' to replace."
                )
            else:
                self.single_plaintext_summary.configure(
                    text=" click 'Insert Message' to enter a message "
                )
        else:
            # Random mode: disable insert button and clear manual storage
            self.single_insert_btn.configure(state="disabled")
            self.single_plaintext_summary.configure(
                text="Random plaintext will be generated."
            )


    def _run_single_example_clicked(self):
        if self.code is None or self.decoder is None or self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized.")
            return

        # Parse inputs
        try:
            p = float(self.single_p_var.get())
            errors_num = int(self.single_delib_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values for p and deliberate errors.")
            return

        if not (0.0 <= p <= 1.0):
            messagebox.showerror(
                "Invalid probability",
                "Channel error probability per bit must be in the range [0, 1].",
            )
            return

        if errors_num < 0 or errors_num > int(self.code.max_errors_num):
            messagebox.showerror(
                "Invalid deliberate errors",
                f"Deliberate errors must be between 0 and t={self.code.max_errors_num}.",
            )
            return

        mode = self.single_msg_mode.get()
        is_random = (mode == "random")
        plaintext_vec = None

        if not is_random:
            bits_str = self.single_manual_plaintext_bits
            if not bits_str:
                messagebox.showerror(
                    "Plaintext missing",
                    "Manual plaintext mode is selected but no message was inserted.\n"
                    "Click 'Message Insert...' to provide a message.",
                )
                return
            if len(bits_str) != self.k_tag:
                messagebox.showerror(
                    "Plaintext length error",
                    f"Stored plaintext must have exactly {self.k_tag} bits.",
                )
                return
            if any(c not in "01" for c in bits_str):
                messagebox.showerror(
                    "Plaintext error",
                    "Stored plaintext must contain only '0' and '1'.",
                )
                return
            plaintext_vec = core.GF2([int(b) for b in bits_str])

        def worker():
            try:
                report_obj = core.run_single_example_report(
                    code=self.code,
                    error_prob_channel=p,
                    errors_num=errors_num,
                    is_random_message=is_random,
                    decoder=self.decoder,
                    k_tag=self.k_tag,
                    plaintext=plaintext_vec,
                )
                report = report_obj.to_dict(serializable=True)
            except Exception:
                report = {
                    "error": "Error while running single example",
                    "traceback": traceback.format_exc(),
                }


            self.after(0, lambda: self._update_single_output(report))

        threading.Thread(target=worker, daemon=True).start()

    def _draw_single_run_cards(self, report: dict):
        """
        Build the nice card-style view for a single run.
        """

        # Clear previous content
        for child in self.single_output_inner.winfo_children():
            child.destroy()

        # -------- Error card (if any) --------
        if "error" in report:
            err_card = ttk.Frame(self.single_output_inner, style="Card.TFrame", padding=15)
            err_card.pack(fill="x", pady=(5, 5))

            ttk.Label(
                err_card,
                text="⚠ Error while running single example",
                style="Section.TLabel",
            ).grid(row=0, column=0, sticky="w", pady=(0, 5))

            ttk.Label(
                err_card,
                text=report.get("traceback", ""),
                justify="left",
                font=("Consolas", 9),
            ).grid(row=1, column=0, sticky="w")
            return  # nothing more to draw

        # -------- Card 1: Code Parameters --------
        code_card = ttk.Frame(self.single_output_inner, style="Card.TFrame", padding=15)
        code_card.pack(fill="x", pady=(5, 10))

        ttk.Label(
            code_card,
            text="⚙️  Code Parameters",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        code_fields = [
            ("Code:", report.get("code_parameters")),
            ("Max correctable errors:", report.get("t")),
            ("Message length:", report.get("message length")),
            ("Channel error probability per bit:", report.get("error_prob_channel")),
            ("Deliberate errors (requested):", report.get("requested_deliberate_errors")),
        ]

        for i, (label, value) in enumerate(code_fields, start=1):
            ttk.Label(
                code_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                code_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

        # -------- Card 2: Error injection --------
        tx_card = ttk.Frame(self.single_output_inner, style="Card.TFrame", padding=15)
        tx_card.pack(fill="x", pady=(5, 10))

        ttk.Label(
            tx_card,
            text="\u200B📡  Error Injection",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        tx_fields = [
            ("Deliberate errors used:", report.get("Total number of deliberate errors ")),
            ("Channel flips:", report.get("Number of channel flips")),
            ("Total injected errors:", report.get("Total number of errors ")),
        ]

        for i, (label, value) in enumerate(tx_fields, start=1):
            ttk.Label(
                tx_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                tx_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

        # -------- Card 3: Transmission details (antenna) --------
        tx_card = ttk.Frame(self.single_output_inner, style="Card.TFrame", padding=15)
        tx_card.pack(fill="x", pady=(5, 10))

        ttk.Label(
            tx_card,
            text=" 📩  Transmission Details",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        tx_fields = [
            ("Plaintext:", report.get("Plaintext")),
            ("CRC bits:", report.get("CRC bits")),
            ("Received Codeword:", report.get("Received r = c ⊕ e_tot")),
            ("CRC remainder:", report.get("CRC remainder")),
            ("Recovered Plaintext:", report.get("Recovered data (strip CRC)")),
        ]

        for i, (label, value) in enumerate(tx_fields, start=1):
            ttk.Label(
                tx_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                tx_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

        # -------- Card 4: Results --------
        res_card = ttk.Frame(self.single_output_inner, style="Card.TFrame", padding=15)
        res_card.pack(fill="x", pady=(0, 10))

        ttk.Label(
            res_card,
            text="📊 Results",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        crc_ok = bool(report.get("CRC check", False))
        dec_ok = bool(report.get("decode success", False))

        def _yes_no_icon(flag: bool) -> str:
            return "✅ Yes" if flag else "❌ No"

        res_fields = [
            ("CRC check:", _yes_no_icon(crc_ok)),
            ("Decode success:", _yes_no_icon(dec_ok)),
        ]

        for i, (label, value) in enumerate(res_fields, start=1):
            ttk.Label(
                res_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                res_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

        # -------- Card 5: Timing --------
        time_card = ttk.Frame(self.single_output_inner, style="Card.TFrame", padding=15)
        time_card.pack(fill="x", pady=(0, 10))

        ttk.Label(
            time_card,
            text="🕒 Timing",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        time_fields = [
            ("Encoding time per plaintext bit (µs):", report.get("encoding_us_per_plain_bit")),
            ("Decoding time per code bit (µs):", report.get("decoding_us_per_code_bit")),
        ]

        for i, (label, value) in enumerate(time_fields, start=1):
            ttk.Label(
                time_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                time_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

    def _update_single_output(self, report: dict):
        self._draw_single_run_cards(report)

    def _clear_single_tab(self):
        self.single_p_var.set("0.05")
        self.single_delib_var.set("1")
        self.single_msg_mode.set("random")
        self.single_manual_plaintext_bits = None
        self._update_single_plaintext_state()

        # איפוס האזור הגרפי
        for child in self.single_output_inner.winfo_children():
            child.destroy()

        placeholder = ttk.Label(
            self.single_output_inner,
            text="Run a single example to see a detailed, card-style report here.",
            style="TLabel",
            justify="left",
        )
        placeholder.pack(anchor="w", pady=10)

        self.single_last_report = None

    # ---------------------------------------------------------
    # Batch Simulation Tab
    # ---------------------------------------------------------
    def _build_batch_tab(self):
        container = ttk.Frame(self.batch_tab, style="Main.TFrame", padding=10)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------------- Left side: parameters ----------------
        left = ttk.Frame(container, style="Card.TFrame", padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left, text="Batch Simulation Parameters", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Label(left, text="Number of messages:").grid(row=1, column=0, sticky="w")
        self.batch_num_messages_var = tk.StringVar(value="50")
        ttk.Entry(left, textvariable=self.batch_num_messages_var, width=10).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(left, text="Channel error probability per bit (0–1):").grid(
            row=2, column=0, sticky="w", pady=(5, 0)
        )
        self.batch_p_var = tk.StringVar(value="0.05")
        ttk.Entry(left, textvariable=self.batch_p_var, width=10).grid(
            row=2, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(left, text="Deliberate errors:").grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )
        self.batch_delib_var = tk.StringVar(value="1")
        ttk.Entry(left, textvariable=self.batch_delib_var, width=10).grid(
            row=3, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(left, text="Max retries per message (optional):").grid(
            row=4, column=0, sticky="w", pady=(5, 0)
        )
        self.batch_max_retries_var = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.batch_max_retries_var, width=10).grid(
            row=4, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(left, text="Max duration per message [s] (optional):").grid(
            row=5, column=0, sticky="w", pady=(5, 0)
        )
        self.batch_max_duration_var = tk.StringVar(value="")
        ttk.Entry(left, textvariable=self.batch_max_duration_var, width=10).grid(
            row=5, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(left, text="Plaintext source:").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(10, 0)
        )

        self.batch_plaintext_mode = tk.StringVar(value="random")
        rb_random = ttk.Radiobutton(
            left,
            text="Random plaintexts",
            value="random",
            variable=self.batch_plaintext_mode,
            command=self._update_batch_plaintext_state,
        )
        rb_manual = ttk.Radiobutton(
            left,
            text="Manual plaintexts",
            value="manual",
            variable=self.batch_plaintext_mode,
            command=self._update_batch_plaintext_state,
        )
        rb_random.grid(row=7, column=0, columnspan=3, sticky="w")
        rb_manual.grid(row=8, column=0, columnspan=3, sticky="w")

        # Summary + Message Insert button (no free text area here)
        self.batch_plaintext_hint_label = ttk.Label(
            left,
            text="Random plaintexts will be generated.",
        )
        self.batch_plaintext_hint_label.grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        self.batch_insert_btn = ttk.Button(
            left,
            text=" Insert Plaintexts ",
            command=self._open_batch_plaintext_dialog,
        )
        self.batch_insert_btn.grid(row=10, column=0, columnspan=3, sticky="w", pady=(3, 0))

        btn_frame = ttk.Frame(left, style="Main.TFrame")
        btn_frame.grid(row=11, column=0, columnspan=3, sticky="we", pady=(10, 0))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        run_btn = ttk.Button(
            btn_frame,
            text="Run Batch Simulation",
            style="Accent.TButton",
            command=self._run_batch_clicked,
        )
        run_btn.grid(row=0, column=0, sticky="we", padx=(0, 5))

        clear_btn = ttk.Button(
            btn_frame,
            text="Clear",
            command=self._clear_batch_tab,
        )
        clear_btn.grid(row=0, column=1, sticky="we", padx=(5, 0))

        # ---------------- Right side: notebook with two tabs ----------------
        right = ttk.Frame(container, style="Main.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.batch_notebook = ttk.Notebook(right)
        self.batch_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Tab 1: Batch Summary Metrics ---
        summary_tab = ttk.Frame(self.batch_notebook, style="Main.TFrame", padding=10)
        self.batch_notebook.add(summary_tab, text="Batch Summary Metrics")

        ttk.Label(summary_tab, text="Batch Summary Metrics", style="Section.TLabel").pack(
            side=tk.TOP, anchor="w", pady=(0, 5)
        )

        # Scrollable area with cards (like Single Run)
        self.batch_summary_container = ttk.Frame(summary_tab, style="White.TFrame")
        self.batch_summary_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.batch_summary_canvas = tk.Canvas(
            self.batch_summary_container,
            borderwidth=0,
            highlightthickness=0,
            bg="#ffffff",
        )
        vscroll = ttk.Scrollbar(
            self.batch_summary_container,
            orient="vertical",
            command=self.batch_summary_canvas.yview,
        )
        self.batch_summary_canvas.configure(yscrollcommand=vscroll.set)

        self.batch_summary_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Inner frame that will hold the cards
        self.batch_summary_inner = ttk.Frame(self.batch_summary_canvas, style="White.TFrame")
        self.batch_summary_inner_id = self.batch_summary_canvas.create_window(
            (0, 0),
            window=self.batch_summary_inner,
            anchor="nw",
        )

        def _on_batch_frame_config(event):
            self.batch_summary_canvas.configure(
                scrollregion=self.batch_summary_canvas.bbox("all")
            )

        self.batch_summary_inner.bind("<Configure>", _on_batch_frame_config)

        def _on_batch_canvas_config(event):
            self.batch_summary_canvas.itemconfigure(
                self.batch_summary_inner_id,
                width=event.width,
            )

        self.batch_summary_canvas.bind("<Configure>", _on_batch_canvas_config)

        # Initial placeholder
        placeholder = ttk.Label(
            self.batch_summary_inner,
            text="Run a batch simulation to see summary metrics here.",
            style="TLabel",
            justify="left",
        )
        placeholder.pack(anchor="w", pady=10)


        # --- Tab 2: Per-Message Rollup ---
        rollup_tab = ttk.Frame(self.batch_notebook, style="Main.TFrame", padding=10)
        self.batch_notebook.add(rollup_tab, text="Per-Message Rollup")

        header_frame = ttk.Frame(rollup_tab, style="Card.TFrame")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(
            header_frame,
            text="Per-Message Rollup",
            style="Section.TLabel",
        ).pack(side=tk.LEFT, anchor="w", pady=(0, 5))

        export_btn = ttk.Button(
            header_frame,
            text="Export table to CSV",
            command=self._export_batch_table_to_csv,
        )
        export_btn.pack(side=tk.RIGHT, padx=(0, 5), pady=(0, 5))

        tree_container = ttk.Frame(rollup_tab, style="Card.TFrame")
        tree_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = (
            "msg_index",
            "transmissions",
            "success",
            "reason",
            "avg_channel_flips",
            "avg_total_errors",
            #"avg_channel_ber",
        )
        self.batch_tree = ttk.Treeview(
            tree_container,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="Custom.Treeview"
        )

        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.batch_tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.batch_tree.xview)
        self.batch_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.batch_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Alternating row colors
        self.batch_tree.tag_configure("odd", background="#ffffff")
        self.batch_tree.tag_configure("even", background="#f7f7f7")

        tree_container.rowconfigure(0, weight=1)
        tree_container.columnconfigure(0, weight=1)

        self.batch_tree.heading("msg_index", text="#")
        self.batch_tree.heading("transmissions", text="Transmissions")
        self.batch_tree.heading("success", text="Success")
        self.batch_tree.heading("reason", text="Reason")
        self.batch_tree.heading("avg_channel_flips", text="Average channel errors per transmission")
        self.batch_tree.heading("avg_total_errors", text="Average errors per transmission")
        #self.batch_tree.heading("avg_channel_ber", text="Average BER channel")

        self.batch_tree.column("msg_index", width=40, anchor="center")
        self.batch_tree.column("transmissions", width=100, anchor="center")
        self.batch_tree.column("success", width=70, anchor="center")
        self.batch_tree.column("reason", width=230, anchor="center")
        self.batch_tree.column("avg_channel_flips", width=170, anchor="center")
        self.batch_tree.column("avg_total_errors", width=160, anchor="center")
        #self.batch_tree.column("avg_channel_ber", width=140, anchor="center")

        # Initial state for manual messages controls
        self._update_batch_plaintext_state()

    def _draw_batch_summary_cards(self, rep: dict):
        """
        Build the card-style summary view for the batch simulation.
        """

        # Clear previous content
        for child in self.batch_summary_inner.winfo_children():
            child.destroy()

        # -------- Error card--------
        if "error" in rep:
            err_card = ttk.Frame(self.batch_summary_inner, style="Card.TFrame", padding=15)
            err_card.pack(fill="x", pady=(5, 5))

            ttk.Label(
                err_card,
                text="⚠ Error while running batch simulation",
                style="Section.TLabel",
            ).grid(row=0, column=0, sticky="w", pady=(0, 5))

            ttk.Label(
                err_card,
                text=rep.get("traceback", ""),
                justify="left",
                font=("Consolas", 9),
            ).grid(row=1, column=0, sticky="w")
            return

        # -------- Card 1: Code & batch configuration --------
        cfg_card = ttk.Frame(self.batch_summary_inner, style="Card.TFrame", padding=15)
        cfg_card.pack(fill="x", pady=(5, 10))

        ttk.Label(
            cfg_card,
            text="⚙️  Code & Batch Details",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        cfg_fields = [
            ("Code:", rep.get("Code Parameters")),
            ("max correctable errors:", rep.get("t")),
            ("Message length:", rep.get("k_tag")),
            ("Channel error probability per bit:", rep.get("Channel Error Probability (p)")),
            ("Deliberate errors number:", rep.get("Deliberate Errors (Encoder Noise)")),
            ("Number of messages:", rep.get("Number of Messages")),
        ]

        for i, (label, value) in enumerate(cfg_fields, start=1):
            ttk.Label(
                cfg_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                cfg_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

        # -------- Card 2: Transmission Details --------
        perf_card = ttk.Frame(self.batch_summary_inner, style="Card.TFrame", padding=15)
        perf_card.pack(fill="x", pady=(0, 10))

        ttk.Label(
            perf_card,
            text="\u200B📡 Transmission Details",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        perf_fields = [
            ("Average channel flips per transmission:",
             rep.get("Average Channel Errors per Transmission")),
            ("Average total errors per transmission:",
             rep.get("Average Total Errors per Transmission")),
            #("Average BER (channel) per transmission:",
            # rep.get("Average BER channel Error per Transmission")),
            #("Average BER (overall) per transmission:",
            # rep.get("Average BER Error per Transmission")),
            ("Average transmissions per message:",
             rep.get("Average Attempts per Message")),
            ("Total transmissions:",
             rep.get("Total Transmissions")),
        ]

        for i, (label, value) in enumerate(perf_fields, start=1):
            ttk.Label(
                perf_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                perf_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)

        # -------- Card 3: Batch performance metrics --------
        perf_card = ttk.Frame(self.batch_summary_inner, style="Card.TFrame", padding=15)
        perf_card.pack(fill="x", pady=(0, 10))

        ttk.Label(
            perf_card,
            text="📊 Batch Results",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        perf_fields = [
            ("CRC correct detection rate (%):",
             rep.get("CRC Correct Decodability Detection (%)")),
            ("CRC Check- ✅ & Decoded message- ❌ (%):",
             rep.get("False Positives (%)")),
            ("CRC Check- \u200B❌  & Decoded message- \u200B✅  (%):",
             rep.get("False Negatives (%)")),
            ("Decoder success rate (%):",
             rep.get("Decoder Success Rate (%)")),
        ]

        for i, (label, value) in enumerate(perf_fields, start=1):
            ttk.Label(
                perf_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                perf_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)
        # -------- Card 3: Timing --------
        time_card = ttk.Frame(self.batch_summary_inner, style="Card.TFrame", padding=15)
        time_card.pack(fill="x", pady=(0, 10))

        ttk.Label(
            time_card,
            text="🕒 Timing",
            style="SectionEmoji.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        time_fields = [
            ("Average encoding time per plaintext bit (µs):",
             rep.get("Average Encoding Time per plaintext bit (us)")),
            ("Average decoding time per code bit (µs):",
             rep.get("Average Decoding Time per code bit (us)")),
        ]

        for i, (label, value) in enumerate(time_fields, start=1):
            ttk.Label(
                time_card,
                text=label,
                font=("Segoe UI", 10, "bold"),
            ).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 10))
            ttk.Label(
                time_card,
                text=str(value),
                font=("Segoe UI", 10),
            ).grid(row=i, column=1, sticky="w", pady=2)


    def _update_batch_plaintext_state(self):
        """
        Enable / disable the 'Message Insert' button and update the description
        according to the plaintext mode and how many messages are stored.
        """
        mode = self.batch_plaintext_mode.get()
        if mode == "manual":
            self.batch_insert_btn.configure(state="normal")
            # Try to read desired number of messages (for the hint only)
            try:
                num_messages = int(self.batch_num_messages_var.get())
            except ValueError:
                num_messages = None

            count = len(self.batch_manual_plaintexts_bits)

            if num_messages is not None and num_messages > 0:
                self.batch_plaintext_hint_label.configure(
                    text=f"Manual plaintexts: {count} / {num_messages} messages stored.\n"
                         f"Click 'Insert plaintexts' to add or edit messages."
                )
            else:
                self.batch_plaintext_hint_label.configure(
                    text=f"Manual plaintexts: {count} messages stored.\n"
                         f"Set Number of messages and click 'Insert plaintexts'."
                )
        else:
            self.batch_insert_btn.configure(state="disabled")
            self.batch_plaintext_hint_label.configure(
                text="Random plaintexts will be generated."
            )

    def _open_batch_plaintext_dialog(self):
        """
        Dialog window for entering manual plaintexts for batch simulation.

        Styled like the rest of the GUI and with an OK button that becomes
        enabled only after the required number of messages is entered.
        """
        if self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized (k_tag is unknown).")
            return

        # Determine how many messages must be entered
        try:
            num_messages = int(self.batch_num_messages_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid number of messages",
                "Please enter a valid integer in 'Number of messages' before inserting messages.",
            )
            return
        if num_messages <= 0:
            messagebox.showerror(
                "Invalid number of messages",
                "Number of messages must be > 0.",
            )
            return

        win = tk.Toplevel(self)
        win.title("Manual plaintexts for Batch Simulation")
        win.transient(self)
        win.grab_set()

        main = ttk.Frame(win, style="Card.TFrame", padding=10)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Header: how many messages are required and how many already entered
        count_var = tk.StringVar()
        ttk.Label(
            main,
            textvariable=count_var,
            style="Section.TLabel",
        ).pack(side=tk.TOP, anchor="w", pady=(0, 4))

        ttk.Label(
            main,
            text=(
                f"Enter a binary plaintext message of length {self.k_tag} bits.\n"
                "Only characters '0' and '1' are allowed.\n"
                "Press 'Apply' after each valid message."
            ),
            justify="left",
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        entry_var = tk.StringVar()
        entry = ttk.Entry(main, textvariable=entry_var, width=max(32, self.k_tag))
        entry.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        entry.focus_set()

        # Bit-count label
        bit_count_var = tk.StringVar()
        ttk.Label(main, textvariable=bit_count_var).pack(
            side=tk.TOP, anchor="w", pady=(0, 8)
        )

        # Bottom buttons row
        btn_frame = ttk.Frame(main, style="Main.TFrame")
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        # OK button (enabled only when all messages entered)
        ok_btn = ttk.Button(btn_frame, text="OK")
        ok_btn.pack(side=tk.RIGHT, padx=(5, 0))
        ok_btn.state(["disabled"])

        def update_bit_count(*_):
            bits_str = entry_var.get().replace(" ", "")
            bit_count_var.set(f"Bits entered: {len(bits_str)} / {self.k_tag}")

        def update_count_label():
            current = len(self.batch_manual_plaintexts_bits)
            count_var.set(f"Messages entered: {current} / {num_messages}")

            # Enable OK only once all messages are entered
            if current >= num_messages:
                ok_btn.state(["!disabled"])
            else:
                ok_btn.state(["disabled"])

        entry_var.trace_add("write", update_bit_count)
        update_bit_count()
        update_count_label()

        def _validate_and_store(bits_str: str) -> bool:
            bits_str = bits_str.replace(" ", "")
            if len(bits_str) != self.k_tag:
                messagebox.showerror(
                    "Plaintext length error",
                    f"Plaintext must have exactly {self.k_tag} bits.",
                    parent=win,
                )
                return False
            if any(c not in "01" for c in bits_str):
                messagebox.showerror(
                    "Plaintext error",
                    "Plaintext must contain only '0' and '1'.",
                    parent=win,
                )
                return False
            self.batch_manual_plaintexts_bits.append(bits_str)
            return True

        def on_apply():
            current = len(self.batch_manual_plaintexts_bits)
            if current >= num_messages:
                messagebox.showerror(
                    "Too many messages",
                    "You have already entered the required number of messages.\n"
                    "Increase 'Number of messages' if you want to send more.",
                    parent=win,
                )
                return

            bits_str = entry_var.get()
            if _validate_and_store(bits_str):
                entry_var.set("")
                update_bit_count()
                update_count_label()
                self._update_batch_plaintext_state()

        def on_clear():
            entry_var.set("")
            update_bit_count()

        def on_ok():
            # OK is enabled only when count == num_messages
            win.destroy()

        # Buttons: Clear | Apply | OK
        clear_btn = ttk.Button(btn_frame, text="Clear", command=on_clear)
        clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        apply_btn = ttk.Button(btn_frame, text="Apply", command=on_apply)
        apply_btn.pack(side=tk.LEFT)

        ok_btn.configure(command=on_ok)

        def on_close():
            current = len(self.batch_manual_plaintexts_bits)
            if current < num_messages:
                messagebox.showwarning(
                    "Not enough messages",
                    f"You entered {current} messages, but {num_messages} are required.\n"
                    "You can still run the simulation, but it will fail if not enough\n"
                    "manual messages are provided.",
                    parent=win,
                )
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def _run_batch_clicked(self):
        if self.code is None or self.decoder is None or self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized.")
            return

        try:
            num_messages = int(self.batch_num_messages_var.get())
            p = float(self.batch_p_var.get())
            errors_num = int(self.batch_delib_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please check numeric parameters.")
            return

        if num_messages <= 0:
            messagebox.showerror("Invalid number of messages", "Number of messages must be > 0.")
            return
        if not (0.0 <= p <= 1.0):
            messagebox.showerror(
                "Invalid probability",
                "Channel error probability per bit must be in the range [0, 1].",
            )
            return
        if errors_num < 0 or errors_num > int(self.code.max_errors_num):
            messagebox.showerror(
                "Invalid deliberate errors",
                f"Deliberate errors must be between 0 and t={self.code.max_errors_num}.",
            )
            return

        # Optional max_retries / max_duration
        max_retries = None
        txt = self.batch_max_retries_var.get().strip()
        if txt:
            try:
                max_retries = int(txt)
            except ValueError:
                messagebox.showerror("Invalid max retries", "Max retries must be an integer.")
                return

        max_duration = None
        txt = self.batch_max_duration_var.get().strip()
        if txt:
            try:
                max_duration = float(txt)
            except ValueError:
                messagebox.showerror("Invalid duration", "Max duration must be a float (seconds).")
                return

        # Plaintexts: Random or Manual
        mode = self.batch_plaintext_mode.get()
        is_random = (mode == "random")
        plaintexts = None

        if not is_random:
            # Use messages collected via the Message Insert dialog
            count_manual = len(self.batch_manual_plaintexts_bits)
            if count_manual != num_messages:
                messagebox.showerror(
                    "Not enough messages",
                    f"Manual mode is selected but {count_manual} messages were entered,\n"
                    f"while 'Number of messages' is {num_messages}.",
                )
                return

            # Convert each stored bit-string to GF2 vector
            plaintexts = [
                core.GF2([int(b) for b in bits_str])
                for bits_str in self.batch_manual_plaintexts_bits
            ]


        def worker():
            try:
                report = core.simulate_messages_batch(
                    code=self.code,
                    error_prob_channel=p,
                    errors_num=errors_num,
                    is_random_message=is_random,
                    decoder=self.decoder,
                    k_tag=self.k_tag,
                    num_messages=num_messages,
                    plaintexts=plaintexts,
                    max_retries_per_msg=max_retries,
                    max_duration_per_msg_s=max_duration,
                    keep_deliberate_errors=True,
                )
                rep_dict = report.to_dict()
                per_msg_list = rep_dict.get("per_message", [])
            except Exception:
                rep_dict = {
                    "error": "Error while running batch simulation",
                    "traceback": traceback.format_exc(),
                }
                per_msg_list = []

            self.after(0, lambda: self._update_batch_ui(rep_dict, per_msg_list))


        threading.Thread(target=worker, daemon=True).start()

    def _update_batch_ui(self, rep_dict: dict, per_msg_list):
        # Summary – draw cards instead of plain text
        self._draw_batch_summary_cards(rep_dict)

        # Save for CSV export
        self.batch_last_per_msg = list(per_msg_list)

        # Table
        self.batch_tree.delete(*self.batch_tree.get_children())

        for idx, m in enumerate(per_msg_list, start=1):
            tag = "odd" if idx % 2 == 1 else "even"

            self.batch_tree.insert(
                "",
                tk.END,
                values=(
                    idx,
                    m.get("attempts"),
                    m.get("success"),
                    m.get("reason"),
                    f"{m.get('avg_channel_flips_per_attempt', 0):.2f}",
                    f"{m.get('avg_total_errors_per_attempt', 0):.2f}",
                ),
                tags=(tag,)
            )


    def _clear_batch_tab(self):
        self.batch_num_messages_var.set("50")
        self.batch_p_var.set("0.05")
        self.batch_delib_var.set("1")
        self.batch_max_retries_var.set("")
        self.batch_max_duration_var.set("")
        self.batch_plaintext_mode.set("random")
        self.batch_manual_plaintexts_bits = []
        self._update_batch_plaintext_state()

        # Reset the card area
        for child in self.batch_summary_inner.winfo_children():
            child.destroy()

        placeholder = ttk.Label(
            self.batch_summary_inner,
            text="Run a batch simulation to see summary metrics here.",
            style="TLabel",
            justify="left",
        )
        placeholder.pack(anchor="w", pady=10)

        # Clear table
        self.batch_tree.delete(*self.batch_tree.get_children())
        self.batch_last_per_msg = []

    def _export_batch_table_to_csv(self):
        if not self.batch_last_per_msg:
            messagebox.showinfo("Export to CSV", "No batch results to export.")
            return

        filename = filedialog.asksaveasfilename(
            title="Export batch table to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        # Write from per_message list – includes more fields than table
        keys = sorted(self.batch_last_per_msg[0].keys())
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.batch_last_per_msg:
                    writer.writerow(row)
            messagebox.showinfo("Export to CSV", f"Batch table exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ---------------------------------------------------------
    # Grid Sweep Tab (graphs)
    # ---------------------------------------------------------
    def _build_grid_tab(self):
        # Top parameters + buttons – only input area in this tab
        params_frame = ttk.Frame(self.grid_tab, style="Card.TFrame", padding=10)
        params_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)

        ttk.Label(
            params_frame,
            text="Grid Sweep",
            style="Section.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # All input rows one under another – cleaner layout
        ttk.Label(
            params_frame,
            text="Channel error probabilities list  (comma-separated or [start:step:end], 0–1):",
        ).grid(row=1, column=0, sticky="w")
        self.grid_p_list_var = tk.StringVar(value="0.01, 0.05, 0.1")
        ttk.Entry(params_frame, textvariable=self.grid_p_list_var, width=40).grid(
            row=1, column=1, sticky="we", padx=(5, 0)
        )

        ttk.Label(
            params_frame,
            text="Deliberate errors list (comma-separated or [start:step:end]):",
        ).grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.grid_delib_list_var = tk.StringVar(value="0, 1, 2, 3")
        ttk.Entry(params_frame, textvariable=self.grid_delib_list_var, width=40).grid(
            row=2, column=1, sticky="we", padx=(5, 0), pady=(5, 0)
        )

        ttk.Label(params_frame, text="Messages per grid point:").grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )
        self.grid_num_messages_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.grid_num_messages_var, width=10).grid(
            row=3, column=1, sticky="w", pady=(5, 0)
        )

        # Note: plaintexts are always random for grid sweeps – no user option here.

        # Axis selection (still configured here, but controls for X/Y will also appear in result window)
        ttk.Label(params_frame, text="X-axis metric:").grid(
            row=5, column=0, sticky="w", pady=(10, 0)
        )
        self.grid_x_var = tk.StringVar()
        self.grid_x_combo = ttk.Combobox(
            params_frame,
            textvariable=self.grid_x_var,
            state="readonly",
            width=32,
        )
        self.grid_x_combo.grid(row=5, column=1, sticky="w", pady=(10, 0))

        ttk.Label(params_frame, text="Y-axis metric:").grid(
            row=6, column=0, sticky="w", pady=(5, 0)
        )
        self.grid_y_var = tk.StringVar()
        self.grid_y_combo = ttk.Combobox(
            params_frame,
            textvariable=self.grid_y_var,
            state="readonly",
            width=32,
        )
        self.grid_y_combo.grid(row=6, column=1, sticky="w", pady=(5, 0))

        # Buttons: Run + Stop (no Clear button on this tab)
        self.grid_run_btn = ttk.Button(
            params_frame,
            text="Run Grid Sweep",
            style="Accent.TButton",
            command=self._run_grid_clicked,
        )
        self.grid_run_btn.grid(row=7, column=0, columnspan=2, pady=(15, 0), sticky="we")

        self.grid_stop_btn = ttk.Button(
            params_frame,
            text="Stop",
            command=self._stop_grid_clicked,
            state="disabled",
        )
        self.grid_stop_btn.grid(row=8, column=0, columnspan=2, pady=(5, 0), sticky="we")

        # No grid_text / plot area directly in this tab –
        # results and graph will appear in a dedicated popup window.

        # Prepare placeholders for matplotlib axes/canvas (used in popup)
        self.grid_ax = None
        self.grid_canvas = None


    def _init_grid_axis_maps(self):
        # Human-readable → metrics keys returned by metrics_for_plots()
        self.GRID_X_OPTIONS = {
            "Channel error probability": "error_prob_channel",
            "Number of deliberate errors": "deliberate_errors_requested",
        }
        self.GRID_Y_OPTIONS = {
            "Decoder Success Rate (%)": "decoder_success_pct",
            "CRC Correct Detection (%)": "crc_detection_pct",
            "CRC OK & Wrong decode (%)": "false_positive_pct",
            "CRC fail & Right decode (%)": "false_negative_pct",
            #"BER Channel Error": "Average BER channel Error per Transmission",
            #"BER (overall)": "Average BER Error per Transmission",
            "Average transmissions per message": "Average transmissions per Message",
            "Avg. channel flip per transmission": "avg_channel_errors_per_transmission",
        }

        self.grid_x_combo["values"] = list(self.GRID_X_OPTIONS.keys())
        self.grid_y_combo["values"] = list(self.GRID_Y_OPTIONS.keys())
        # Default selection
        self.grid_x_var.set("Channel error probability p")
        self.grid_y_var.set("Decoder Success Rate (%)")

        # When X/Y selection changes (in main tab combo) and when used in popup,
        # we want to re-plot the existing results.
        self.grid_x_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._plot_grid_results(self.grid_last_results),
        )
        self.grid_y_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._plot_grid_results(self.grid_last_results),
        )

    @staticmethod
    def _parse_number_list_or_range(text: str, kind: str):
        """
        Parse either:
          - comma-separated list: "0.01, 0.05, 0.1"
          - range in form [start:step:end], e.g. "[0:0.01:1]"
        kind: "float" or "int"
        """
        text = text.strip()
        if not text:
            return []

        def cast(x):
            return float(x) if kind == "float" else int(float(x))

        # Range notation
        if text.startswith("[") and text.endswith("]") and ":" in text:
            body = text[1:-1]
            parts = [p.strip() for p in body.split(":")]
            if len(parts) != 3:
                raise ValueError("Range must be in form [start:step:end].")
            start = cast(parts[0])
            step = cast(parts[1])
            end = cast(parts[2])
            if step <= 0:
                raise ValueError("Step in range must be > 0.")

            values = []
            current = start
            # Little epsilon for floats
            eps = 1e-12 if kind == "float" else 0
            while current <= end + eps:
                if kind == "float":
                    values.append(float(current))
                else:
                    values.append(int(round(current)))
                current += step
            return values

        # Comma-separated list
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return [cast(p) for p in parts]

    def _run_grid_clicked(self):
        if self.code is None or self.decoder is None or self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized.")
            return

        # If a sweep is already running, avoid starting another one
        if self.grid_worker_thread is not None and self.grid_worker_thread.is_alive():
            messagebox.showinfo("Grid Sweep", "A grid sweep is already running.")
            return

        # Parse p list
        try:
            p_list = self._parse_number_list_or_range(self.grid_p_list_var.get(), kind="float")
        except ValueError as e:
            messagebox.showerror("Invalid p list", str(e))
            return

        if not p_list:
            messagebox.showerror("Invalid p list", "At least one channel probability value is required.")
            return

        if any(p < 0 or p > 1 for p in p_list):
            messagebox.showerror(
                "Invalid probability",
                "All channel probabilities p must be in [0, 1].",
            )
            return

        # Parse deliberate errors list
        try:
            delib_list = self._parse_number_list_or_range(self.grid_delib_list_var.get(), kind="int")
        except ValueError as e:
            messagebox.showerror("Invalid deliberate errors list", str(e))
            return

        if not delib_list:
            messagebox.showerror(
                "Invalid deliberate errors list",
                "At least one deliberate errors value is required.",
            )
            return

        if any(d < 0 or d > int(self.code.max_errors_num) for d in delib_list):
            messagebox.showerror(
                "Invalid deliberate errors",
                f"All deliberate errors must be between 0 and t={self.code.max_errors_num}.",
            )
            return

        try:
            num_messages = int(self.grid_num_messages_var.get())
        except ValueError:
            messagebox.showerror("Invalid messages number", "Messages per grid point must be an integer.")
            return
        if num_messages <= 0:
            messagebox.showerror("Invalid messages number", "Messages per grid point must be > 0.")
            return

        # For grid sweeps we always use random plaintexts automatically.
        is_random = True

        # If a previous results window is open, close it before new run
        if self.grid_results_window is not None:
            try:
                self.grid_results_window.destroy()
            except Exception:
                pass
            self.grid_results_window = None

        self.grid_stop_requested = False
        self.grid_run_btn.configure(state="disabled")
        self.grid_stop_btn.configure(state="normal")

        def worker():
            try:
                # Re-implement sweep_simulation_grid logic here,
                # but with the ability to stop between grid points.
                results = []
                for errors_num in delib_list:
                    if self.grid_stop_requested:
                        break
                    for p in p_list:
                        if self.grid_stop_requested:
                            break

                        batch = core.simulate_messages_batch(
                            code=self.code,
                            decoder=self.decoder,
                            k_tag=self.k_tag,
                            error_prob_channel=p,
                            errors_num=errors_num,
                            plaintexts=None,
                            num_messages=num_messages,
                            is_random_message=is_random,
                            max_retries_per_msg=None,
                            max_duration_per_msg_s=None,
                        )
                        metrics = batch.metrics_for_plots()
                        results.append(metrics)
            except Exception:
                results = []
                text = "Error while running grid sweep:\n" + traceback.format_exc()
            else:
                text = self._format_grid_results(results)

            def on_done():
                self.grid_worker_thread = None
                self.grid_run_btn.configure(state="normal")
                self.grid_stop_btn.configure(state="disabled")
                self._update_grid_ui(text, results)

            self.after(0, on_done)

        self.grid_worker_thread = threading.Thread(target=worker, daemon=True)
        self.grid_worker_thread.start()


    def _stop_grid_clicked(self):
        """
        Request stopping the grid sweep.
        The actual stop happens between grid points (no changes needed in Core_new.py).
        """
        if self.grid_worker_thread is not None and self.grid_worker_thread.is_alive():
            self.grid_stop_requested = True

    def _update_grid_ui(self, text: str, results: list[dict]):
        # Save results for re-plot / export and open the popup window with the graph.
        self.grid_last_results = list(results)
        self._open_grid_results_window()

    # ---------- Grid results popup ----------

    def _open_grid_results_window(self):
        if not HAS_MPL:
            messagebox.showerror(
                "Matplotlib missing",
                "Matplotlib is required to display the grid sweep plot.\n"
                "Install it (e.g., 'pip install matplotlib') and restart.",
            )
            return

        if not self.grid_last_results:
            messagebox.showinfo("Grid Sweep", "No results to display (maybe stopped early).")
            return

        # Close previous window if still open
        if self.grid_results_window is not None:
            try:
                self.grid_results_window.destroy()
            except Exception:
                pass
            self.grid_results_window = None

        win = tk.Toplevel(self)
        win.title("Grid Sweep Results")
        win.geometry("900x600")
        self.grid_results_window = win

        def on_close():
            # Closing this window just frees the UI – it does not cancel a running thread.
            self.grid_results_window = None
            self.grid_results_fig = None
            self.grid_results_ax = None
            self.grid_results_canvas = None
            self.grid_point_info_var = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        # Top controls: X/Y selection + Export button
        top = ttk.Frame(win, style="Card.TFrame", padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Grid Sweep Plot", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 5)
        )

        ttk.Label(top, text="X-axis metric:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        x_combo_popup = ttk.Combobox(
            top,
            textvariable=self.grid_x_var,
            values=list(self.GRID_X_OPTIONS.keys()),
            state="readonly",
            width=32,
        )
        x_combo_popup.grid(row=1, column=1, sticky="w", pady=(5, 0))

        ttk.Label(top, text="Y-axis metric:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        y_combo_popup = ttk.Combobox(
            top,
            textvariable=self.grid_y_var,
            values=list(self.GRID_Y_OPTIONS.keys()),
            state="readonly",
            width=32,
        )
        y_combo_popup.grid(row=2, column=1, sticky="w", pady=(5, 0))

        export_btn = ttk.Button(
            top,
            text="Export grid to CSV",
            command=self._export_grid_results_to_csv,
        )
        export_btn.grid(row=1, column=2, rowspan=2, sticky="e", padx=(10, 0), pady=(5, 0))

        # Bind axis changes in popup to re-plot
        x_combo_popup.bind(
            "<<ComboboxSelected>>",
            lambda e: self._plot_grid_results(self.grid_last_results),
        )
        y_combo_popup.bind(
            "<<ComboboxSelected>>",
            lambda e: self._plot_grid_results(self.grid_last_results),
        )

        # Figure + canvas for the plot
        fig_frame = ttk.Frame(win, style="Card.TFrame", padding=5)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 10), padx=10)

        self.grid_results_fig = Figure(figsize=(5, 4), dpi=100)
        self.grid_results_ax = self.grid_results_fig.add_subplot(111)
        self.grid_results_canvas = FigureCanvasTkAgg(self.grid_results_fig, master=fig_frame)
        self.grid_results_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Label to show the last clicked point
        self.grid_point_info_var = tk.StringVar(
            value="Click a point on the curve to see its values."
        )
        info_label = ttk.Label(fig_frame, textvariable=self.grid_point_info_var)
        info_label.pack(side=tk.BOTTOM, anchor="w", pady=(4, 0))

        # Connect pick events for interactive point inspection
        self.grid_results_canvas.mpl_connect("pick_event", self._on_grid_point_picked)

        # Point grid_ax / grid_canvas to the popup widgets so _plot_grid_results can reuse the logic.
        self.grid_ax = self.grid_results_ax
        self.grid_canvas = self.grid_results_canvas

        # Initial plot
        self._plot_grid_results(self.grid_last_results)


    def _plot_grid_results(self, results: list[dict]):
        if not HAS_MPL or not self.grid_ax or not self.grid_canvas:
            return

        self.grid_ax.clear()

        if not results:
            self.grid_canvas.draw()
            return

        x_label = self.grid_x_var.get()
        y_label = self.grid_y_var.get()
        x_key = self.GRID_X_OPTIONS.get(x_label)
        y_key = self.GRID_Y_OPTIONS.get(y_label)

        if not x_key or not y_key:
            # Fallback to defaults
            x_key = "error_prob_channel"
            y_key = "decoder_success_pct"
            x_label = "Channel error probability p"
            y_label = "Decoder Success Rate (%)"

        # Determine grouping: other axis is "series"
        if x_key == "error_prob_channel":
            group_key = "deliberate_errors_requested"
            label_fmt = "errors={}"
        else:
            group_key = "error_prob_channel"
            label_fmt = "p={:.3f}"

        groups = {}
        for entry in results:
            if x_key not in entry or y_key not in entry:
                continue
            x_val = entry[x_key]
            y_val = entry[y_key]
            g_val = entry.get(group_key)
            groups.setdefault(g_val, []).append((x_val, y_val))

        for g_val, pts in sorted(groups.items(), key=lambda kv: kv[0] if kv[0] is not None else -1):
            pts_sorted = sorted(pts, key=lambda t: t[0])
            xs = [p for p, _ in pts_sorted]
            ys = [q for _, q in pts_sorted]

            line, = self.grid_ax.plot(xs, ys, marker="o", label=label_fmt.format(g_val))
            # Make the line pickable so we can click on points
            try:
                line.set_picker(5)  # 5 points tolerance
            except Exception:
                pass

        self.grid_ax.set_xlabel(x_label)
        self.grid_ax.set_ylabel(y_label)
        self.grid_ax.set_title("Grid sweep results")
        self.grid_ax.grid(True)

        # Only call legend if we actually plotted something (avoids the warning)
        if groups:
            legend = self.grid_ax.legend()
            # Make the legend draggable so it can be moved away from the data
            if legend is not None and hasattr(legend, "set_draggable"):
                try:
                    legend.set_draggable(True)
                except Exception:
                    pass

        # Reset point-info text when re-plotting
        if isinstance(getattr(self, "grid_point_info_var", None), tk.StringVar):
            try:
                self.grid_point_info_var.set("Click a point on the curve to see its values.")
            except Exception:
                pass

        self.grid_canvas.draw()

    def _on_grid_point_picked(self, event):
        """
        Matplotlib pick-event handler for the grid sweep plot.

        When the user clicks near a data point on one of the curves,
        this function shows its (x, y) values and the series label.
        """
        if not HAS_MPL:
            return

        info_var = getattr(self, "grid_point_info_var", None)
        if info_var is None:
            return

        artist = getattr(event, "artist", None)
        try:
            from matplotlib.lines import Line2D
        except Exception:
            Line2D = None

        if Line2D is None or not isinstance(artist, Line2D):
            return

        if not event.ind:
            return

        idx = event.ind[0]
        x_data = artist.get_xdata()
        y_data = artist.get_ydata()
        if idx >= len(x_data) or idx >= len(y_data):
            return

        x_val = x_data[idx]
        y_val = y_data[idx]
        series_label = artist.get_label() or ""

        # Format nicely
        try:
            x_text = f"{x_val:.6g}"
        except Exception:
            x_text = str(x_val)
        try:
            y_text = f"{y_val:.6g}"
        except Exception:
            y_text = str(y_val)

        if series_label:
            msg = f"Selected point: series='{series_label}', x={x_text}, y={y_text}"
        else:
            msg = f"Selected point: x={x_text}, y={y_text}"

        try:
            info_var.set(msg)
        except Exception:
            pass


    def _clear_grid_tab(self):
        # Used by "Clear All" – no dedicated Clear button on this tab anymore
        self.grid_last_results = []
        self.grid_stop_requested = True

        if self.grid_results_window is not None:
            try:
                self.grid_results_window.destroy()
            except Exception:
                pass
            self.grid_results_window = None

        if hasattr(self, "grid_run_btn"):
            self.grid_run_btn.configure(state="normal")
        if hasattr(self, "grid_stop_btn"):
            self.grid_stop_btn.configure(state="disabled")

    def _export_grid_results_to_csv(self):
        if not self.grid_last_results:
            messagebox.showinfo("Export to CSV", "No grid results to export.")
            return

        filename = filedialog.asksaveasfilename(
            title="Export grid results to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filename:
            return

        keys = sorted(self.grid_last_results[0].keys())
        try:
            with open(filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in self.grid_last_results:
                    writer.writerow(row)
            messagebox.showinfo("Export to CSV", f"Grid results exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _format_dict_pretty(d: dict) -> str:
        lines = []
        for k, v in d.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    @staticmethod
    def _format_batch_summary(rep: dict) -> str:
        keys_of_interest = [
            "Code Parameters",
            "n",
            "k",
            "t",
            "k_tag",
            "Channel Error Probability (p)",
            "Deliberate Errors (Encoder Noise)",
            "Number of Messages",
            "Total Transmissions",
            "Average Channel Errors per Transmission",
            "Average Total Errors per Transmission",
            "Average Attempts per Message",
            "Decoder Success Rate (%)",
            "CRC Correct Decodability Detection (%)",
            "Average Encoding Time per plaintext bit (us)",
            "Average Decoding Time per code bit (us)",
        ]
        lines = []
        for k in keys_of_interest:
            if k in rep:
                lines.append(f"{k}: {rep[k]}")
        return "\n".join(lines)

    @staticmethod
    def _format_grid_results(results: list[dict]) -> str:
        """
        Currently used mainly for debugging / console printing.
        Display GUI itself only.
        """
        lines = []
        for entry in results:
            p = entry.get("error_prob_channel")
            de = entry.get("deliberate_errors_requested")
            dec_succ = entry.get("decoder_success_pct")
            fp = entry.get("false_positive_pct")
            fn = entry.get("false_negative_pct")
            crc_det = entry.get("crc_detection_pct")
            avg_tx = entry.get("Average transmissions per Message")
            avg_ch_err = entry.get("avg_channel_errors_per_transmission")
            ch_ber = entry.get("Average BER channel Error per Transmission")
            tot_ber = entry.get("Average BER Error per Transmission")

            line = (
                f"p={p:.3f}, deliberate={de} | "
                f"Decoder Success={dec_succ:.2f}%, "
                f"CRC Detection={crc_det:.2f}%, "
                f"FP={fp:.2f}%, FN={fn:.2f}%, "
                f"Avg transmissions/msg={avg_tx:.2f}, "
                f"Avg channel errors/tx={avg_ch_err:.3f}, "
                f"Channel BER={ch_ber:.4e}, Total BER={tot_ber:.4e}"
            )
            lines.append(line)
        return "\n".join(lines)

    def _clear_all_results(self):
        self._clear_single_tab()
        self._clear_batch_tab()
        self._clear_grid_tab()


if __name__ == "__main__":
    app = LinearCodeGUI()
    app.mainloop()
