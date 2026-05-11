from __future__ import annotations

import json
import os
import subprocess
import threading
import traceback
import tempfile
import sys
from pathlib import Path
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from .core import list_image_files, select_before_after
    from .methods import METHODS, get_method_spec
except ImportError:
    from core import list_image_files, select_before_after
    from methods import METHODS, get_method_spec


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _find_conda_exe() -> Path | None:
    candidates = []
    conda_env = os.environ.get("CONDA_EXE")
    if conda_env:
        candidates.append(Path(conda_env))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Scripts" / "conda.exe")
    local_appdata = os.environ.get("LOCALAPPDATA")
    candidates.extend(
        [
            Path(r"C:\ProgramData\anaconda3\Scripts\conda.exe"),
            Path(r"C:\ProgramData\miniconda3\Scripts\conda.exe"),
            Path.home() / "anaconda3" / "Scripts" / "conda.exe",
            Path.home() / "miniconda3" / "Scripts" / "conda.exe",
        ]
    )
    if local_appdata:
        candidates.extend(
            [
                Path(local_appdata) / "anaconda3" / "Scripts" / "conda.exe",
                Path(local_appdata) / "miniconda3" / "Scripts" / "conda.exe",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _current_python_executable() -> Path | None:
    python_exe = Path(sys.executable)
    if python_exe.exists() and any(part.lower() == "envs" for part in python_exe.parts):
        return python_exe
    return None


def _payload_to_result(data: dict):
    try:
        from .core import AnalysisResult
    except ImportError:
        from core import AnalysisResult

    artifacts = {key: Path(value) for key, value in data.get("artifacts", {}).items()}
    preview_image_path = data.get("preview_image_path")
    return AnalysisResult(
        method_id=data["method_id"],
        method_name=data["method_name"],
        summary=data["summary"],
        metrics=data["metrics"],
        before_path=Path(data["before_path"]),
        after_path=Path(data["after_path"]),
        preview_text=data["preview_text"],
        preview_image_path=Path(preview_image_path) if preview_image_path else None,
        artifacts=artifacts,
    )


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Photo Pair Analyzer")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        self.folder_var = tk.StringVar(value="")
        self.method_var = tk.StringVar(value=METHODS[0])
        self.status_var = tk.StringVar(value="Выберите папку с парой фото и запустите анализ.")

        self.before_text_label: tk.Text | None = None
        self.after_text_label: tk.Text | None = None
        self.preview_image_label: tk.Label | None = None
        self.preview_caption_label: ttk.Label | None = None
        self.result_text: tk.Text | None = None
        self.preview_image: tk.PhotoImage | None = None

        self._build_ui()
        # load saved preferences (last folder and method)
        try:
            self._load_prefs()
        except Exception:
            # don't fail GUI startup for prefs issues
            pass
        # save prefs whenever these vars change
        try:
            self.folder_var.trace_add("write", lambda *_args: self._save_prefs())
            self.method_var.trace_add("write", lambda *_args: self._save_prefs())
        except Exception:
            pass

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        top = ttk.Frame(container)
        top.pack(fill="x")

        folder_row = ttk.Frame(top)
        folder_row.pack(fill="x", pady=(0, 8))
        ttk.Label(folder_row, text="Папка с парой фото:", width=20).pack(side="left")
        ttk.Entry(folder_row, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(folder_row, text="Выбрать", command=self.choose_folder).pack(side="left")

        method_row = ttk.Frame(top)
        method_row.pack(fill="x", pady=(0, 8))
        ttk.Label(method_row, text="Метод:", width=20).pack(side="left")
        method_combo = ttk.Combobox(method_row, textvariable=self.method_var, values=METHODS, state="readonly")
        method_combo.pack(side="left", fill="x", expand=True)
        method_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_method_hint())

        action_row = ttk.Frame(top)
        action_row.pack(fill="x", pady=(0, 8))
        ttk.Button(action_row, text="Запустить анализ", command=self.run_analysis).pack(side="left")
        ttk.Label(action_row, textvariable=self.status_var).pack(side="left", padx=12)

        self.method_hint = ttk.Label(top, text="")
        self.method_hint.pack(fill="x", pady=(0, 8))
        self._update_method_hint()

        body = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=2)
        body.add(right, weight=1)

        text_grid = ttk.Frame(left)
        text_grid.pack(fill="both", expand=True)
        text_grid.columnconfigure(0, weight=1)
        text_grid.columnconfigure(1, weight=1)
        text_grid.rowconfigure(0, weight=1)
        text_grid.rowconfigure(1, weight=1)

        self.before_text_label = self._make_text_panel(text_grid, "До", 0, 0)
        self.after_text_label = self._make_text_panel(text_grid, "После", 0, 1)
        self._make_preview_panel(text_grid, "Результат", 1, 0, colspan=2)

        result_frame = ttk.LabelFrame(right, text="Результаты", padding=8)
        result_frame.pack(fill="both", expand=True)
        result_toolbar = ttk.Frame(result_frame)
        result_toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(result_toolbar, text="Скопировать текст", command=self.copy_results_text).pack(side="left")
        self.result_text = tk.Text(result_frame, wrap="word", height=30)
        scroll = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        self.result_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._install_text_copy_support(self.result_text)

    def _make_text_panel(self, parent: ttk.Frame, title: str, row: int, column: int, colspan: int = 1) -> tk.Text:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=row, column=column, columnspan=colspan, sticky="nsew", padx=6, pady=6)
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(column, weight=1)
        widget = tk.Text(frame, wrap="word", height=10)
        widget.pack(fill="both", expand=True)
        widget.configure(state="disabled")
        return widget

    def _make_preview_panel(self, parent: ttk.Frame, title: str, row: int, column: int, colspan: int = 1) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=row, column=column, columnspan=colspan, sticky="nsew", padx=6, pady=6)
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(column, weight=1)

        self.preview_image_label = tk.Label(frame, text="Здесь появится итоговая карта", anchor="center", justify="center", bg="#111111", fg="white")
        self.preview_image_label.pack(fill="both", expand=True)
        self.preview_caption_label = ttk.Label(frame, text="")
        self.preview_caption_label.pack(fill="x", pady=(6, 0))

    def _update_method_hint(self) -> None:
        method_id = self.method_var.get()
        spec = get_method_spec(method_id)
        self.method_hint.configure(text=f"Текущий метод: {spec.label} | conda env: {spec.env_name}")

    def choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Выберите папку с парой фото")
        if folder:
            self.folder_var.set(folder)
            self.status_var.set("Папка выбрана. Можно запускать анализ.")

    def run_analysis(self) -> None:
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("Нет папки", "Сначала выберите папку с изображениями.")
            return

        path = Path(folder)
        if not path.exists():
            messagebox.showerror("Ошибка", "Папка не существует.")
            return

        try:
            image_files = list_image_files(path)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        if len(image_files) < 2:
            messagebox.showwarning("Мало файлов", "Нужны как минимум 2 изображения в папке.")
            return

        before_path, after_path = select_before_after(image_files)
        self.status_var.set("Анализ запущен...")
        self._set_text(
            "Выбрана пара для анализа:\n"
            f"Before: {before_path.name}\n"
            f"After: {after_path.name}\n\n"
        )

        worker = threading.Thread(
            target=self._run_worker,
            args=(before_path, after_path),
            daemon=True,
        )
        worker.start()

    def _run_worker(self, before_path: Path, after_path: Path) -> None:
        try:
            result = self._run_analysis_in_conda(self.method_var.get(), before_path, after_path)
            self.after(0, lambda: self._show_result(result))
        except Exception as exc:
            error_text = traceback.format_exc()
            self.after(0, lambda exc=exc, error_text=error_text: self._show_error(exc, error_text))

    def _run_analysis_in_conda(self, method_id: str, before_path: Path, after_path: Path):
        spec = get_method_spec(method_id)
        result_file = Path(tempfile.gettempdir()) / "projekt_photo_pairs" / "gui_results" / f"{method_id}_{before_path.stem}_{after_path.stem}_{uuid4().hex}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        conda_exe = _find_conda_exe()
        if conda_exe is None:
            raise RuntimeError("Не найден conda.exe для запуска метода в его отдельном окружении.")

        command = [
            str(conda_exe),
            "run",
            "-n",
            spec.env_name,
            "python",
            "-m",
            "launcher.worker",
            "--method-id",
            method_id,
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--output",
            str(result_file),
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")

        if not result_file.exists():
            raise RuntimeError(
                "Worker не создал результат.\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )

        payload = json.loads(result_file.read_text(encoding="utf-8"))
        if completed.returncode != 0 or not payload.get("ok", False):
            error_message = payload.get("error") or completed.stderr or completed.stdout or "Unknown worker error"
            raise RuntimeError(error_message)

        return _payload_to_result(payload["result"])

    def _show_result(self, result) -> None:
        self.status_var.set(f"Готово: {result.method_name}")
        self._set_text(self._format_result(result))
        self._set_panel_text(self.before_text_label, f"Файл: {result.before_path.name}\nПуть: {result.before_path}")
        self._set_panel_text(self.after_text_label, f"Файл: {result.after_path.name}\nПуть: {result.after_path}")
        self._set_preview_image(result.preview_image_path)
        if self.preview_caption_label is not None:
            self.preview_caption_label.configure(text=result.preview_text)
        self._open_results_folder(result)

    def _show_error(self, exc: Exception, error_text: str) -> None:
        self.status_var.set("Ошибка во время анализа")
        self._set_text(f"Ошибка: {exc}\n\n{error_text}")
        messagebox.showerror("Ошибка анализа", str(exc))

    def _open_results_folder(self, result) -> None:
        folder: Path | None = None
        if result.preview_image_path is not None:
            folder = result.preview_image_path.parent
        elif result.artifacts:
            first_artifact = next(iter(result.artifacts.values()))
            folder = first_artifact.parent

        if folder is None or not folder.exists():
            return

        try:
            os.startfile(folder)
        except Exception:
            try:
                subprocess.Popen(["explorer", str(folder)])
            except Exception:
                pass

    def _set_text(self, text: str) -> None:
        if self.result_text is None:
            return
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)

    def copy_results_text(self) -> None:
        if self.result_text is None:
            return
        text = self.result_text.get("1.0", tk.END).rstrip()
        self.clipboard_clear()
        self.clipboard_append(text)

    def _install_text_copy_support(self, widget: tk.Text | None) -> None:
        if widget is None:
            return

        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy selected", command=lambda: self._copy_widget_selection(widget))
        menu.add_command(label="Copy all", command=lambda: self._copy_widget_all(widget))

        def open_menu(event):
            menu.tk_popup(event.x_root, event.y_root)
            return "break"

        widget.bind("<Button-3>", open_menu)
        widget.bind("<Control-c>", lambda _event: self._copy_widget_selection(widget))
        widget.bind("<Control-C>", lambda _event: self._copy_widget_selection(widget))

    def _copy_widget_selection(self, widget: tk.Text) -> str:
        try:
            text = widget.selection_get()
        except tk.TclError:
            text = widget.get("1.0", tk.END).rstrip()
        self.clipboard_clear()
        self.clipboard_append(text)
        return "break"

    def _copy_widget_all(self, widget: tk.Text) -> str:
        text = widget.get("1.0", tk.END).rstrip()
        self.clipboard_clear()
        self.clipboard_append(text)
        return "break"

    def _set_panel_text(self, widget: tk.Text | None, text: str) -> None:
        if widget is None:
            return
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_preview_image(self, image_path: Path | None) -> None:
        if self.preview_image_label is None:
            return
        if image_path is None:
            self.preview_image = None
            self.preview_image_label.configure(image="", text="Нет изображения для отображения")
            return

        try:
            self.preview_image = tk.PhotoImage(file=str(image_path))
            self.preview_image_label.configure(image=self.preview_image, text="")
        except Exception:
            self.preview_image = None
            self.preview_image_label.configure(image="", text=f"Не удалось открыть preview:\n{image_path}")

    def _format_result(self, result) -> str:
        method_spec = get_method_spec(result.method_id)
        cleanup_delta = result.metrics.get("cleanup_delta")
        cleanup_score = result.metrics.get("cleanup_score")
        cleanup_block = []
        if cleanup_delta is not None:
            cleanup_block.append(f"Cleanup delta: {cleanup_delta}")
        if cleanup_score is not None:
            cleanup_block.append(f"Cleanup score: {cleanup_score}")

        lines = [
            f"Метод: {result.method_name}",
            f"Conda env: {method_spec.env_name}",
            f"Env file: {method_spec.env_file}",
            f"Кратко: {result.summary}",
            "",
        ]
        if cleanup_block:
            lines.append("Оценка очистки:")
            for item in cleanup_block:
                lines.append(f"- {item}")
            lines.append("")
        lines.append("Метрики:")
        for key, value in result.metrics.items():
            if key in {"cleanup_delta", "cleanup_score"}:
                continue
            lines.append(f"- {key}: {value}")
        if result.preview_image_path is not None:
            lines.extend(["", f"Preview image: {result.preview_image_path}"])
        if result.artifacts:
            lines.extend(["", "Артефакты:"])
            for key, value in result.artifacts.items():
                lines.append(f"- {key}: {value}")
        lines.extend(
            [
                "",
                "Подсказка:",
                "GUI показывает итоговую картинку результата и список артефактов без смены сценария запуска.",
            ]
        )
        return "\n".join(lines)

    def _prefs_path(self) -> Path:
        try:
            home = Path.home()
        except Exception:
            home = PROJECT_ROOT
        return home / ".projekt_gui_prefs.json"

    def _load_prefs(self) -> None:
        path = self._prefs_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        folder = data.get("folder")
        method = data.get("method")
        if folder:
            self.folder_var.set(folder)
        if method and method in METHODS:
            self.method_var.set(method)

    def _save_prefs(self) -> None:
        path = self._prefs_path()
        data = {"folder": self.folder_var.get() or "", "method": self.method_var.get() or ""}
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # best-effort: ignore write errors
            pass

def main() -> None:
    app = LauncherApp()
    app.mainloop()
