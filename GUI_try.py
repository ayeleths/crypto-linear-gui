"""
GUI for Linear Code + ARQ Simulator based on Core_new.py

Place this file in the same folder as Core_new.py and run:
    python gui_app.py
"""

import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Optional: matplotlib for plots in the "Grid Sweep" tab
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:
    HAS_MPL = False

import Core_new as core  # Requires galois, sklearn, etc.


class LinearCodeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Linear Code ARQ Simulator")
        self.geometry("1150x750")
        self.minsize(950, 650)

        # State: current code / decoder / k_tag
        self.code = None
        self.decoder = None
        self.k_tag = None

        self._create_style()
        self._build_main_layout()

        # Initialize with default code
        self._on_code_changed()

    # ---------------------------------------------------------
    # Style
    # ---------------------------------------------------------
    def _create_style(self):
        style = ttk.Style()
        # Use a modern theme if available
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#f5f7fb")
        style.configure("Header.TFrame", background="#e3e7f3")
        style.configure("TLabel", background="#f5f7fb", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#e3e7f3", font=("Segoe UI", 11, "bold"))
        style.configure("Title.TLabel", background="#e3e7f3", font=("Segoe UI", 14, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map(
            "Accent.TButton",
            foreground=[("active", "#ffffff")],
            background=[("!disabled", "#4a6cf7"), ("pressed", "#3a57c4")],
        )
        style.configure("Card.TFrame", background="#ffffff", relief="groove", borderwidth=1)

    # ---------------------------------------------------------
    # Main layout
    # ---------------------------------------------------------
    def _build_main_layout(self):
        # Top frame: code selection + parameters
        top = ttk.Frame(self, style="Header.TFrame", padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        title = ttk.Label(top, text="Linear Code ARQ Simulator", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")

        # Code selection
        code_frame = ttk.Frame(top, style="Header.TFrame")
        code_frame.grid(row=1, column=0, pady=(8, 0), sticky="w")

        ttk.Label(code_frame, text="Code:", style="Header.TLabel").grid(row=0, column=0, padx=(0, 5))

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
        self.code_combo.grid(row=0, column=1, padx=(0, 15))
        self.code_combo.bind("<<ComboboxSelected>>", lambda e: self._on_code_changed())

        self.code_info_label = ttk.Label(
            code_frame,
            text="",
            style="Header.TLabel",
        )
        self.code_info_label.grid(row=0, column=2, padx=(0, 10))

        self.k_tag_label = ttk.Label(
            code_frame,
            text="",
            style="Header.TLabel",
        )
        self.k_tag_label.grid(row=0, column=3, padx=(0, 10))

        # Note about BCH
        self.note_label = ttk.Label(
            top,
            text="Note: BCH(63,36,≈11) may be heavy because the decoder builds a coset-leader table.",
            style="Header.TLabel",
        )
        self.note_label.grid(row=2, column=0, pady=(4, 0), sticky="w")

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
                    f"k_tag <= 0. Message is too short for CRC: k={self.code.k}, CRC length={core.CRC.generator_degree}"
                )

            self.code_info_label.config(
                text=f"n={self.code.n}, k={self.code.k}, d_min={self.code.d_min}, t={self.code.max_errors_num}"
            )
            self.k_tag_label.config(text=f"message length (k_tag) = {self.k_tag} bits")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error while building code", str(e))

    # ---------------------------------------------------------
    # Single Run Tab
    # ---------------------------------------------------------
    def _build_single_tab(self):
        # Split: left = parameters, right = output
        left = ttk.Frame(self.single_tab, style="Card.TFrame", padding=10)
        right = ttk.Frame(self.single_tab, style="Card.TFrame", padding=10)

        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)

        # --- Left: parameters ---
        ttk.Label(left, text="Single Run Parameters", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        # Channel error probability
        ttk.Label(left, text="Channel error probability p (0–1):").grid(row=1, column=0, sticky="w")
        self.single_p_var = tk.StringVar(value="0.05")
        ttk.Entry(left, textvariable=self.single_p_var, width=10).grid(row=1, column=1, sticky="w")

        # Deliberate errors
        ttk.Label(left, text="Deliberate errors at encoder:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.single_delib_var = tk.StringVar(value="1")
        ttk.Entry(left, textvariable=self.single_delib_var, width=10).grid(row=2, column=1, sticky="w", pady=(5, 0))

        # Random/manual message selection
        ttk.Label(left, text="Plaintext source:").grid(row=3, column=0, sticky="w", pady=(10, 0))

        self.single_msg_mode = tk.StringVar(value="random")
        random_rb = ttk.Radiobutton(left, text="Random plaintext", value="random", variable=self.single_msg_mode,
                                    command=self._update_single_plaintext_state)
        manual_rb = ttk.Radiobutton(left, text="Manual plaintext", value="manual", variable=self.single_msg_mode,
                                    command=self._update_single_plaintext_state)
        random_rb.grid(row=4, column=0, columnspan=2, sticky="w")
        manual_rb.grid(row=5, column=0, columnspan=2, sticky="w")

        ttk.Label(left, text="Manual plaintext bits (length = k_tag):").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.single_plaintext_var = tk.StringVar()
        self.single_plaintext_entry = ttk.Entry(left, textvariable=self.single_plaintext_var, width=28)
        self.single_plaintext_entry.grid(row=7, column=0, columnspan=2, sticky="w")
        self._update_single_plaintext_state()

        # Run button
        run_btn = ttk.Button(
            left,
            text="Run Single Example",
            style="Accent.TButton",
            command=self._run_single_example_clicked,
        )
        run_btn.grid(row=8, column=0, columnspan=2, pady=(15, 0), sticky="we")

        # --- Right: output ---
        ttk.Label(right, text="Single Run Report", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 5)
        )
        self.single_output = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=25,
        )
        self.single_output.pack(fill=tk.BOTH, expand=True)

    def _update_single_plaintext_state(self):
        mode = self.single_msg_mode.get()
        if mode == "manual":
            self.single_plaintext_entry.configure(state="normal")
        else:
            self.single_plaintext_entry.configure(state="disabled")

    def _run_single_example_clicked(self):
        if self.code is None or self.decoder is None or self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized.")
            return

        try:
            p = float(self.single_p_var.get())
            errors_num = int(self.single_delib_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values for p and deliberate errors.")
            return

        if not (0.0 <= p <= 1.0):
            messagebox.showerror("Invalid p", "Channel error probability p must be in [0, 1].")
            return

        plaintext_vec = None
        is_random = (self.single_msg_mode.get() == "random")

        if not is_random:
            bits_str = self.single_plaintext_var.get().replace(" ", "")
            if len(bits_str) != self.k_tag:
                messagebox.showerror(
                    "Plaintext length error",
                    f"Manual plaintext must have exactly {self.k_tag} bits.",
                )
                return
            if any(c not in "01" for c in bits_str):
                messagebox.showerror("Plaintext error", "Plaintext must contain only 0 and 1.")
                return
            # Convert to GF(2) vector
            arr = [int(b) for b in bits_str]
            plaintext_vec = core.GF2(arr)

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
                text = self._format_dict_pretty(report)
            except Exception as e:
                text = "Error while running single example:\n" + traceback.format_exc()

            # Update UI in main thread
            self.single_output.after(0, lambda: self._update_single_output(text))

        threading.Thread(target=worker, daemon=True).start()

    def _update_single_output(self, text: str):
        self.single_output.delete("1.0", tk.END)
        self.single_output.insert(tk.END, text)

    # ---------------------------------------------------------
    # Batch Simulation Tab
    # ---------------------------------------------------------
    def _build_batch_tab(self):
        # Split horizontally
        top = ttk.Frame(self.batch_tab, style="TFrame")
        top.pack(side=tk.TOP, fill=tk.X)

        params_frame = ttk.Frame(top, style="Card.TFrame", padding=10)
        params_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)

        summary_frame = ttk.Frame(top, style="Card.TFrame", padding=10)
        summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)

        table_frame = ttk.Frame(self.batch_tab, style="Card.TFrame", padding=10)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 5))

        # --- Params ---
        ttk.Label(params_frame, text="Batch Simulation Parameters", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Label(params_frame, text="Number of messages (trials):").grid(row=1, column=0, sticky="w")
        self.batch_num_messages_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.batch_num_messages_var, width=10).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(params_frame, text="Channel error probability p:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.batch_p_var = tk.StringVar(value="0.05")
        ttk.Entry(params_frame, textvariable=self.batch_p_var, width=10).grid(row=2, column=1, sticky="w", pady=(5, 0))

        ttk.Label(params_frame, text="Deliberate errors at encoder:").grid(row=3, column=0, sticky="w", pady=(5, 0))
        self.batch_delib_var = tk.StringVar(value="1")
        ttk.Entry(params_frame, textvariable=self.batch_delib_var, width=10).grid(
            row=3, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(params_frame, text="Max retries per message (optional):").grid(
            row=4, column=0, sticky="w", pady=(5, 0)
        )
        self.batch_max_retries_var = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.batch_max_retries_var, width=10).grid(
            row=4, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(params_frame, text="Max duration per message [s] (optional):").grid(
            row=5, column=0, sticky="w", pady=(5, 0)
        )
        self.batch_max_duration_var = tk.StringVar(value="")
        ttk.Entry(params_frame, textvariable=self.batch_max_duration_var, width=10).grid(
            row=5, column=1, sticky="w", pady=(5, 0)
        )

        # Random / external plaintexts
        ttk.Label(params_frame, text="Plaintext source:").grid(row=6, column=0, sticky="w", pady=(10, 0))
        self.batch_is_random_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            params_frame,
            text="Random plaintexts",
            variable=self.batch_is_random_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w")

        ttk.Label(
            params_frame,
            text="(Optional) manual plaintexts, one per line,\nlength = k_tag bits. If non-empty,\nthis overrides random mode.",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.batch_plaintexts_text = scrolledtext.ScrolledText(
            params_frame,
            wrap=tk.NONE,
            width=28,
            height=6,
            font=("Consolas", 8),
        )
        self.batch_plaintexts_text.grid(row=9, column=0, columnspan=2, sticky="w")

        run_btn = ttk.Button(
            params_frame,
            text="Run Batch Simulation",
            style="Accent.TButton",
            command=self._run_batch_clicked,
        )
        run_btn.grid(row=10, column=0, columnspan=2, pady=(10, 0), sticky="we")

        # --- Summary ---
        ttk.Label(summary_frame, text="Batch Summary Metrics", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 5)
        )
        self.batch_summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            font=("Consolas", 9),
            height=18,
        )
        self.batch_summary_text.pack(fill=tk.BOTH, expand=True)

        # --- Per-message table ---
        ttk.Label(table_frame, text="Per-Message Rollup", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 5)
        )

        columns = (
            "msg_index",
            "attempts",
            "success",
            "reason",
            "avg_channel_flips",
            "avg_total_errors",
        )
        self.batch_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.batch_tree.pack(fill=tk.BOTH, expand=True)

        self.batch_tree.heading("msg_index", text="#")
        self.batch_tree.heading("attempts", text="Attempts")
        self.batch_tree.heading("success", text="Success")
        self.batch_tree.heading("reason", text="Reason")
        self.batch_tree.heading("avg_channel_flips", text="Avg channel flips/attempt")
        self.batch_tree.heading("avg_total_errors", text="Avg total errors/attempt")

        self.batch_tree.column("msg_index", width=40, anchor="center")
        self.batch_tree.column("attempts", width=80, anchor="center")
        self.batch_tree.column("success", width=70, anchor="center")
        self.batch_tree.column("reason", width=230, anchor="w")
        self.batch_tree.column("avg_channel_flips", width=170, anchor="center")
        self.batch_tree.column("avg_total_errors", width=170, anchor="center")

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
            messagebox.showerror("Invalid num messages", "Number of messages must be > 0.")
            return
        if not (0.0 <= p <= 1.0):
            messagebox.showerror("Invalid p", "Channel error probability p must be in [0, 1].")
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

        # Optional manual plaintexts
        manual_text = self.batch_plaintexts_text.get("1.0", tk.END).strip()
        plaintexts = None
        is_random = self.batch_is_random_var.get()

        if manual_text:
            lines = [ln.strip().replace(" ", "") for ln in manual_text.splitlines() if ln.strip()]
            if len(lines) != num_messages:
                messagebox.showerror(
                    "Plaintexts mismatch",
                    f"You provided {len(lines)} plaintext lines but num_messages = {num_messages}.",
                )
                return
            vecs = []
            for i, line in enumerate(lines, start=1):
                if len(line) != self.k_tag:
                    messagebox.showerror(
                        "Plaintext length error",
                        f"Line {i}: expected {self.k_tag} bits, got {len(line)}.",
                    )
                    return
                if any(c not in "01" for c in line):
                    messagebox.showerror(
                        "Plaintext bit error",
                        f"Line {i}: plaintext must contain only 0/1.",
                    )
                    return
                vecs.append(core.GF2([int(b) for b in line]))
            plaintexts = vecs
            is_random = False  # override

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
                summary_text = self._format_batch_summary(rep_dict)

                per_msg_list = rep_dict.get("per_message", [])
            except Exception:
                summary_text = "Error while running batch simulation:\n" + traceback.format_exc()
                per_msg_list = []

            self.batch_summary_text.after(0, lambda: self._update_batch_ui(summary_text, per_msg_list))

        threading.Thread(target=worker, daemon=True).start()

    def _update_batch_ui(self, summary_text: str, per_msg_list):
        # Summary
        self.batch_summary_text.delete("1.0", tk.END)
        self.batch_summary_text.insert(tk.END, summary_text)

        # Table
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)

        for idx, m in enumerate(per_msg_list, start=1):
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
            )

    # ---------------------------------------------------------
    # Grid Sweep Tab
    # ---------------------------------------------------------
    def _build_grid_tab(self):
        # Top parameters + buttons
        params_frame = ttk.Frame(self.grid_tab, style="Card.TFrame", padding=10)
        params_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(params_frame, text="Grid Sweep (sweep_simulation_grid)", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10)
        )

        ttk.Label(params_frame, text="Channel error probabilities p (comma-separated):").grid(
            row=1, column=0, sticky="w"
        )
        self.grid_p_list_var = tk.StringVar(value="0.01, 0.05, 0.1")
        ttk.Entry(params_frame, textvariable=self.grid_p_list_var, width=35).grid(
            row=1, column=1, sticky="w", padx=(5, 25)
        )

        ttk.Label(params_frame, text="Deliberate errors list (comma-separated):").grid(row=2, column=0, sticky="w")
        self.grid_delib_list_var = tk.StringVar(value="0, 1, 2, 3")
        ttk.Entry(params_frame, textvariable=self.grid_delib_list_var, width=35).grid(
            row=2, column=1, sticky="w", padx=(5, 25)
        )

        ttk.Label(params_frame, text="Messages per grid point:").grid(row=3, column=0, sticky="w")
        self.grid_num_messages_var = tk.StringVar(value="50")
        ttk.Entry(params_frame, textvariable=self.grid_num_messages_var, width=10).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Label(params_frame, text="Random plaintexts?").grid(row=4, column=0, sticky="w", pady=(5, 0))
        self.grid_is_random_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Use random plaintexts", variable=self.grid_is_random_var).grid(
            row=4, column=1, sticky="w", pady=(5, 0)
        )

        run_btn = ttk.Button(
            params_frame,
            text="Run Grid Sweep",
            style="Accent.TButton",
            command=self._run_grid_clicked,
        )
        run_btn.grid(row=1, column=2, rowspan=2, padx=(10, 0), pady=(0, 0), sticky="nswe")

        # Main lower area: left = text, right = plot
        lower = ttk.Frame(self.grid_tab, style="TFrame")
        lower.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(lower, style="Card.TFrame", padding=10)
        right = ttk.Frame(lower, style="Card.TFrame", padding=10)

        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=5)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)

        # Text area
        ttk.Label(left, text="Grid Results (per grid point)", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 5)
        )
        self.grid_text = scrolledtext.ScrolledText(
            left,
            wrap=tk.WORD,
            font=("Consolas", 9),
        )
        self.grid_text.pack(fill=tk.BOTH, expand=True)

        # Plot area (matplotlib)
        ttk.Label(right, text="Decoder Success vs Channel Error (per deliberate errors)", font=("Segoe UI", 11, "bold")
                 ).pack(anchor="w", pady=(0, 5))

        if HAS_MPL:
            self.grid_figure = Figure(figsize=(5, 4), dpi=100)
            self.grid_ax = self.grid_figure.add_subplot(111)
            self.grid_canvas = FigureCanvasTkAgg(self.grid_figure, master=right)
            self.grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.grid_figure = None
            self.grid_ax = None
            self.grid_canvas = None
            ttk.Label(
                right,
                text="Matplotlib is not available.\nInstall it to see plots (e.g., 'pip install matplotlib').",
                font=("Segoe UI", 10),
            ).pack(fill=tk.BOTH, expand=True)

    def _run_grid_clicked(self):
        if self.code is None or self.decoder is None or self.k_tag is None:
            messagebox.showerror("Error", "Code is not initialized.")
            return

        # Parse p list
        try:
            p_list = [float(x.strip()) for x in self.grid_p_list_var.get().split(",") if x.strip()]
        except ValueError:
            messagebox.showerror("Invalid p list", "Channel probabilities must be comma-separated floats.")
            return
        if any(p < 0 or p > 1 for p in p_list):
            messagebox.showerror("Invalid p", "All p values must be in [0,1].")
            return

        # Parse deliberate errors list
        try:
            delib_list = [int(x.strip()) for x in self.grid_delib_list_var.get().split(",") if x.strip()]
        except ValueError:
            messagebox.showerror("Invalid deliberate errors list", "Values must be comma-separated integers.")
            return

        try:
            num_messages = int(self.grid_num_messages_var.get())
        except ValueError:
            messagebox.showerror("Invalid messages number", "Messages per grid point must be an integer.")
            return
        if num_messages <= 0:
            messagebox.showerror("Invalid messages number", "Messages per grid point must be > 0.")
            return

        is_random = self.grid_is_random_var.get()

        def worker():
            try:
                results = core.sweep_simulation_grid(
                    code=self.code,
                    decoder=self.decoder,
                    k_tag=self.k_tag,
                    error_probs=p_list,
                    deliberate_errors_values=delib_list,
                    num_messages=num_messages,
                    is_random_message=is_random,
                    max_retries_per_msg=None,
                    max_duration_per_msg_s=None,
                    plaintexts=None,
                )
                text = self._format_grid_results(results)
            except Exception:
                results = []
                text = "Error while running grid sweep:\n" + traceback.format_exc()

            self.grid_text.after(0, lambda: self._update_grid_ui(text, results))

        threading.Thread(target=worker, daemon=True).start()

    def _update_grid_ui(self, text: str, results: list[dict]):
        self.grid_text.delete("1.0", tk.END)
        self.grid_text.insert(tk.END, text)

        if not HAS_MPL or not self.grid_ax:
            return

        # Clear and replot
        self.grid_ax.clear()

        # Group by deliberate_errors_requested
        by_delib = {}
        for entry in results:
            de = entry.get("deliberate_errors_requested")
            by_delib.setdefault(de, []).append(entry)

        for de, entries in sorted(by_delib.items(), key=lambda kv: kv[0]):
            xs = [e["error_prob_channel"] for e in sorted(entries, key=lambda e: e["error_prob_channel"])]
            ys = [e["decoder_success_pct"] for e in sorted(entries, key=lambda e: e["error_prob_channel"])]
            self.grid_ax.plot(xs, ys, marker="o", label=f"delib={de}")

        self.grid_ax.set_xlabel("Channel error probability p")
        self.grid_ax.set_ylabel("Decoder Success Rate (%)")
        self.grid_ax.set_title("Decoder Success vs p")
        self.grid_ax.grid(True)
        self.grid_ax.legend()

        self.grid_canvas.draw()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    @staticmethod
    def _format_dict_pretty(d: dict, indent: int = 2) -> str:
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
            "Average BER channel Error per Transmission",
            "Average BER Error per Transmission",
            "Average Attempts per Message",
            "Decoder Success Rate (%)",
            "False Positives (%)",
            "False Negatives (%)",
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
        lines = []
        for entry in results:
            p = entry.get("error_prob_channel")
            de = entry.get("deliberate_errors_requested")
            dec_succ = entry.get("decoder_success_pct")
            fp = entry.get("false_positive_pct")
            fn = entry.get("false_negative_pct")
            crc_det = entry.get("crc_detection_pct")
            lines.append(
                f"p={p:.3f}, deliberate={de} | "
                f"Decoder Success={dec_succ:.2f}%, "
                f"CRC Detection={crc_det:.2f}%, "
                f"FP={fp:.2f}%, FN={fn:.2f}%"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    app = LinearCodeGUI()
    app.mainloop()
