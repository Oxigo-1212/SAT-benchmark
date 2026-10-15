"""Keyboard-driven N-Queens terminal interface using the existing solver workers."""

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
import argparse
import curses
import math
from pathlib import Path
import sys
import textwrap
import traceback
from threading import Event

from benchmark import ENCODINGS, MODEL_ENCODINGS, SAT_SOLVERS, SOLVERS, case_encodings, run_case


def build_cases(fields, solvers, encoders):
    """Validate the form and select CNF encodings or compatible native models."""
    try:
        sizes = list(dict.fromkeys(map(int, fields[0].split())))
        timeout = float(fields[1])
        memory = int(fields[2])
    except ValueError:
        raise ValueError("Enter integer N values and memory; timeout must be a number.") from None
    if not sizes or min(sizes) < 1 or not math.isfinite(timeout) or timeout <= 0 or memory <= 0:
        raise ValueError("N and limits must be positive; timeout must be finite.")
    if not solvers:
        raise ValueError("Select at least one solver.")
    if not encoders:
        raise ValueError("Select at least one encoder or native model.")
    if any(solver in SAT_SOLVERS for solver in solvers) and not set(ENCODINGS) & encoders:
        raise ValueError("Select at least one encoder for SAT solvers.")
    cases = [(encoding, n, solver) for n in sizes for solver in SOLVERS if solver in solvers
             for encoding in case_encodings(solver, [name for name in MODEL_ENCODINGS if name in encoders])]
    return cases, timeout, memory


class NQueensTUI:
    def __init__(self, screen, log_dir=None):
        self.screen = screen
        self.log_dir = log_dir
        self.fields = ["8", "30", "3072"]
        self.solvers = {"glucose"}
        self.encoders = {"binomial"}
        self.focus = 0
        self.editing = False
        self.message = "Ready. Choose parameters and run."
        self.pending = deque()
        self.future = None
        self.current = None
        self.cancel = Event()
        self.results = []
        self.result_index = 0
        self.scroll = 0
        self.total = 0
        self.timeout = 30
        self.memory = 3072

    @property
    def busy(self):
        return self.future is not None or bool(self.pending)

    def start(self):
        try:
            cases, self.timeout, self.memory = build_cases(self.fields, self.solvers, self.encoders)
        except ValueError as error:
            self.message = str(error)
            return
        self.pending = deque(cases)
        self.cancel = Event()
        self.editing = False
        self.results = []
        self.result_index = self.scroll = 0
        self.total = len(cases)
        self.message = f"Running {self.total} cases. Press c to cancel."

    def update(self, pool):
        if self.future is not None and self.future.done():
            encoding, n, solver = self.current
            try:
                result = self.future.result()
            except CancelledError:
                result = {"solver": solver, "encoding": encoding, "n": n, "status": "cancelled"}
            except Exception as error:
                result = {"solver": solver, "encoding": encoding, "n": n,
                          "status": "error", "error": f"{type(error).__name__}: {error}"}
            self.results.append(result)
            self.result_index = len(self.results) - 1
            self.scroll = 0
            self.future = self.current = None
            if not self.pending:
                self.message = ("Cancelled. Results are available." if self.cancel.is_set()
                                else f"Finished {len(self.results)} cases. Change choices or run again.")
        if self.future is None and self.pending:
            self.current = self.pending.popleft()
            encoding, n, solver = self.current
            self.future = pool.submit(run_case, encoding, n, self.timeout, self.memory, solver, self.cancel,
                                      **({"log_dir": self.log_dir} if self.log_dir is not None else {}))

    def handle_key(self, key):
        if key in (ord("q"), 27):
            self.cancel.set()
            self.pending.clear()
            return False
        if key == ord("c") and self.busy:
            self.cancel.set()
            self.pending.clear()
            self.message = "Cancelling active worker ..."
        elif key in (curses.KEY_LEFT, curses.KEY_RIGHT) and self.results:
            self.result_index = (self.result_index + (1 if key == curses.KEY_RIGHT else -1)) % len(self.results)
            self.scroll = 0
        elif key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
            self.scroll = max(0, self.scroll + (5 if key == curses.KEY_NPAGE else -5))
        elif not self.busy:
            last = 3 + len(SOLVERS) + len(MODEL_ENCODINGS)
            if key in (9, curses.KEY_DOWN, curses.KEY_BTAB, curses.KEY_UP):
                step = -1 if key in (curses.KEY_BTAB, curses.KEY_UP) else 1
                self.focus = (self.focus + step) % (last + 1)
                self.editing = False
            elif key == ord("r") or (key in (10, 13, curses.KEY_ENTER) and self.focus == last):
                self.start()
            elif self.focus < 3:
                if key in (curses.KEY_BACKSPACE, 127, 8):
                    self.fields[self.focus] = self.fields[self.focus][:-1] if self.editing else ""
                    self.editing = True
                elif 0 <= key < 256 and chr(key) in ("0123456789 ", "0123456789.eE+-", "0123456789")[self.focus]:
                    self.fields[self.focus] = (self.fields[self.focus] if self.editing else "") + chr(key)
                    self.editing = True
            elif key in (32, 10, 13, curses.KEY_ENTER):
                if self.focus < 3 + len(SOLVERS):
                    name = SOLVERS[self.focus - 3]
                    self.solvers.symmetric_difference_update({name})
                elif self.focus < last:
                    name = MODEL_ENCODINGS[self.focus - 3 - len(SOLVERS)]
                    self.encoders.symmetric_difference_update({name})
        return True

    def put(self, y, x, text, style=0, width=None):
        height, columns = self.screen.getmaxyx()
        if 0 <= y < height and 0 <= x < columns - 1:
            try:
                self.screen.addnstr(y, x, text, min(width or columns, columns - x - 1), style)
            except curses.error:
                pass  # A resize can invalidate coordinates between size checks and drawing.

    def result_lines(self, width):
        if not self.results:
            return ["Solutions will appear here.", "", "SAT: a validated solution", "UNSAT: no solution exists",
                    "Timeout/errors remain incomplete."]
        result = self.results[self.result_index]
        lines = [f"N={result['n']}  {result['solver']} / {result['encoding']}",
                 result["status"].upper()]
        if "wall_s" in result:
            lines.append(f"Wall time: {result['wall_s']:.3f}s")
        if result.get("log_dir"):
            lines += textwrap.wrap(f"Logs: {result['log_dir']}", width)
        if result["status"] == "sat":
            columns = result["queen_columns_1based"]
            lines += ["", "Queen columns by row (1-based):"]
            lines += textwrap.wrap(" ".join(map(str, columns)), width)
            lines += ["", "Board (Q = queen):"]
            visible_columns = min(result["n"], max(1, width // 2))
            if visible_columns < result["n"]:
                lines.append(f"Preview: first {visible_columns} columns")
            lines += [" ".join("Q" if col == queen else "." for col in range(1, visible_columns + 1))
                      for queen in columns]
        elif result["status"] == "unsat":
            lines += ["", "No solution exists."]
        elif result.get("error"):
            lines += ["", *textwrap.wrap(result["error"], width)]
        return lines

    def draw(self):
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 24 or width < 76:
            self.put(0, 0, "Resize terminal to at least 76 columns x 24 rows. q quits.")
            self.screen.refresh()
            return
        self.put(0, 2, "N-QUEENS", curses.A_BOLD)
        self.put(1, 2, "Configure, solve, inspect. Each solver runs in its own process.", curses.A_DIM)
        for row in range(3, height - 4):
            self.put(row, 39, "|", curses.A_DIM)
        for index, label in enumerate(("N values", "Timeout (s)", "Memory (MiB)")):
            self.put(3 + index, 2, f"{label:12s} {self.fields[index]}",
                     curses.A_REVERSE if self.focus == index else 0, 35)
        self.put(7, 2, "SOLVERS", curses.A_BOLD)
        self.put(7, 21, "ENCODINGS", curses.A_BOLD)
        for index, name in enumerate(SOLVERS):
            self.put(8 + index, 2, f"[{'x' if name in self.solvers else ' '}] {name}",
                     curses.A_REVERSE if self.focus == 3 + index else 0, 18)
        for index, name in enumerate(MODEL_ENCODINGS):
            style = curses.A_REVERSE if self.focus == 3 + len(SOLVERS) + index else 0
            self.put(8 + index, 21, f"[{'x' if name in self.encoders else ' '}] {name}", style, 17)
        self.put(17, 2, "[ Run selected cases ]" if not self.busy else "[ Solving ... c cancels ]",
                 curses.A_REVERSE if self.focus == 3 + len(SOLVERS) + len(MODEL_ENCODINGS) else curses.A_BOLD, 35)
        self.put(19, 2, "Type replaces; Backspace edits.", curses.A_DIM, 35)
        self.put(3, 41, f"RESULTS {len(self.results)}/{self.total}", curses.A_BOLD)
        if self.current:
            encoder, n, solver = self.current
            self.put(4, 41, f"Running N={n} {solver}/{encoder}", curses.A_DIM)
        elif self.results:
            self.put(4, 41, f"Viewing result {self.result_index + 1} of {len(self.results)}", curses.A_DIM)
        lines = self.result_lines(width - 43)
        visible = height - 10
        self.scroll = min(self.scroll, max(0, len(lines) - visible))
        for index, line in enumerate(lines[self.scroll:self.scroll + visible]):
            self.put(6 + index, 41, line)
        self.put(height - 3, 2, self.message)
        self.put(height - 2, 2, "Tab/Up/Down focus | Space toggle | r run | c cancel | q/Esc quit", curses.A_DIM)
        self.put(height - 1, 2, "Left/Right result | PgUp/PgDn scroll | Ctrl+C quit", curses.A_DIM)
        self.screen.refresh()

    def run(self):
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self.screen.keypad(True)
        self.screen.timeout(100)
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                while True:
                    self.update(pool)
                    self.draw()
                    if not self.handle_key(self.screen.getch()):
                        break
            finally:
                self.cancel.set()
                self.pending.clear()
        return 0


def run_tui(log_dir=None):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("The TUI needs an interactive terminal. Use command-line flags or --help for scripts.", file=sys.stderr)
        return 2
    try:
        return curses.wrapper(lambda screen: NQueensTUI(screen, log_dir=log_dir).run())
    except BaseException as error:
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            with (Path(log_dir) / "tui-errors.log").open("a") as stream:
                traceback.print_exc(file=stream)
        if isinstance(error, curses.error):
            print(f"Could not initialize the terminal UI: {error}", file=sys.stderr)
            return 2
        if isinstance(error, KeyboardInterrupt):
            return 130
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, help="Write live diagnostic logs for each case")
    sys.exit(run_tui(log_dir=parser.parse_args().log_dir))
