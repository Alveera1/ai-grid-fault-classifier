import os, sys, threading
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from the repo folder
sys.path.append(os.getcwd())

import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from common import *

MEANING = {"NF": "No fault (normal)", "LG": "Line to ground", "LL": "Line to line",
           "LLG": "Two lines to ground", "LLL": "All three lines shorted",
           "LLLG": "Three lines to ground"}
NAMES = ["DecisionTree", "RandomForest", "SVM_RBF"]
PRETTY = {"DecisionTree": "Decision Tree", "RandomForest": "Random Forest", "SVM_RBF": "SVM (RBF)"}
WIN = 160                                   # rows in the waveform window (about 50 ms)
GREEN, RED = "#1a7f37", "#cf222e"

models = {n: joblib.load(f"models/{n}.joblib") for n in NAMES}
_, test = blocked_split(load_clean())

# ---------- blocked cross-validation accuracy (Member 3's tuning table) ----------
CV = {"DecisionTree": 99.78, "RandomForest": 99.89, "SVM_RBF": 96.19}
if os.path.exists("models/tuning_results.csv"):
    tr = pd.read_csv("models/tuning_results.csv")
    CV = {r["model"]: float(r["cv_mean_accuracy"]) * 100 for _, r in tr.iterrows()}

# ---------- noise-test numbers (from results/noise_test.csv if it exists) ----------
NOISE_LEVELS = [0, 5, 10, 20, 50]
NOISE = {"DecisionTree": [100, 93.0, 85.4, 74.1, 50.5],
         "RandomForest": [100, 95.6, 91.7, 83.1, 61.6],
         "SVM_RBF":      [100, 100.0, 99.1, 93.7, 74.7]}
if os.path.exists("results/noise_test.csv"):
    nt = pd.read_csv("results/noise_test.csv")
    piv = nt.pivot(index="noise_pct", columns="model", values="accuracy") * 100
    NOISE_LEVELS = list(piv.index)
    NOISE = {n: list(piv[n]) for n in NAMES if n in piv.columns}

# ---------- test accuracy, computed in the background ----------
test_acc = {}
def compute_test_acc():
    Xt, yt = test[FEATURES], test[TARGET].values
    for n in NAMES:
        test_acc[n] = float((models[n].predict(Xt) == yt).mean()) * 100

threading.Thread(target=compute_test_acc, daemon=True).start()

# ---------- window ----------
root = tk.Tk()
root.title("AI Grid Fault Classifier")
root.geometry("1250x900")
try:
    root.state("zoomed")
except Exception:
    pass

state = {"truth": None}
tally = {n: [0, 0] for n in NAMES}          # [right, seen] for loaded readings
entries, cards = {}, {}

def make_canvas(parent, h=3.2):
    fig = Figure(figsize=(8, h), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    return fig, canvas

# ---------- graphs ----------
def draw_probs(X, pred_by_model):
    fig_p.clear()
    ax = fig_p.add_subplot(111)
    x = np.arange(len(CLASSES))
    w = 0.26
    for i, n in enumerate(NAMES):
        try:
            p = models[n].predict_proba(X)[0]
            vals = pd.Series(p, index=models[n].classes_).reindex(CLASSES).fillna(0).values
            label = PRETTY[n]
        except Exception:                           # SVM has no probabilities
            vals = np.array([1.0 if c == pred_by_model[n] else 0.0 for c in CLASSES])
            label = f"{PRETTY[n]} (answer only)"
        ax.bar(x + (i - 1) * w, vals, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability / vote")
    ax.set_title("What each model thinks of this reading")
    ax.legend()
    fig_p.tight_layout()
    canvas_p.draw()

def draw_wave(win, cls):
    fig_w.clear()
    t = (win["time_stamp"] - win["time_stamp"].iloc[0]) * 1000      # milliseconds
    ax1 = fig_w.add_subplot(211)
    for c in ["Ia", "Ib", "Ic"]:
        ax1.plot(t, win[c], label=c)
    ax1.set_title(f"Current waveforms of a real {cls} reading")
    ax1.set_ylabel("Current")
    ax1.legend(loc="upper right")
    ax2 = fig_w.add_subplot(212)
    for c in ["Va", "Vb", "Vc"]:
        ax2.plot(t, win[c], label=c)
    ax2.set_title("Voltage waveforms")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("Voltage")
    ax2.legend(loc="upper right")
    fig_w.tight_layout()
    canvas_w.draw()

def draw_models():
    fig_m.clear()
    ax1 = fig_m.add_subplot(121)
    for n in NAMES:
        if n in NOISE:
            ax1.plot(NOISE_LEVELS, NOISE[n], marker="o", label=PRETTY[n])
    ax1.set_xlabel("Noise (% of feature std)")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Noise robustness")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2 = fig_m.add_subplot(122)
    imp = pd.Series(models["RandomForest"].feature_importances_, index=FEATURES).sort_values()
    ax2.barh(imp.index, imp.values)
    ax2.set_title("Random Forest feature importance")
    ax2.set_xlabel("Importance")
    fig_m.tight_layout()
    canvas_m.draw()

# ---------- cards ----------
def refresh_accuracy():
    for n in NAMES:
        if n in test_acc:
            cards[n]["acc"].config(text=f"Test accuracy: {test_acc[n]:.2f}%  ({len(test):,} rows)")
        else:
            cards[n]["acc"].config(text="Test accuracy: calculating...")
    if len(test_acc) < len(NAMES):
        root.after(700, refresh_accuracy)

def predict(count=False):
    try:
        vals = [float(entries[f].get()) for f in FEATURES]
    except ValueError:
        result.config(text="Every box must contain a number.", fg=RED)
        return
    X = pd.DataFrame([vals], columns=FEATURES)
    pred_by_model = {n: models[n].predict(X)[0] for n in NAMES}
    truth = state["truth"]
    result.config(text=(f"True class of the loaded reading: {truth} - {MEANING[truth]}"
                        if truth else "Typed values: there is no true class to compare with."),
                  fg="black")
    for n in NAMES:
        p = pred_by_model[n]
        c = cards[n]
        colour = "black"
        verdict = ""
        if truth:
            ok = (p == truth)
            colour = GREEN if ok else RED
            verdict = "Correct" if ok else "Wrong"
            if count:
                tally[n][1] += 1
                tally[n][0] += int(ok)
        c["pred"].config(text=p, fg=colour)
        c["meaning"].config(text=MEANING[p])
        c["verdict"].config(text=verdict, fg=colour)
        try:
            c["conf"].config(text=f"Confidence: {models[n].predict_proba(X)[0].max() * 100:.1f}%")
        except Exception:
            c["conf"].config(text="Confidence: n/a (SVM gives an answer only)")
        r, s = tally[n]
        c["tally"].config(text=f"Right on loaded readings: {r} / {s}")
    draw_probs(X, pred_by_model)

def load_row(cls):
    g = test[test[TARGET] == cls].reset_index(drop=True)
    start = np.random.randint(0, len(g) - WIN)
    win = g.iloc[start:start + WIN]
    row = win.iloc[0]
    for f in FEATURES:
        entries[f].delete(0, tk.END)
        entries[f].insert(0, f"{float(row[f]):.6f}")
    state["truth"] = cls
    draw_wave(win, cls)
    predict(count=True)

def forget_truth(_event=None):
    state["truth"] = None

# ---------- layout ----------
tk.Label(root, text="AI Grid Fault Classifier", font=("Arial", 16, "bold")).pack(pady=4)

bar = tk.Frame(root)
bar.pack(pady=2)
for cls in CLASSES:
    tk.Button(bar, text=f"Load {cls}", width=9,
              command=lambda c=cls: load_row(c)).pack(side=tk.LEFT, padx=3)

grid = tk.Frame(root)
grid.pack(pady=2)
for i, f in enumerate(FEATURES):
    r, c = divmod(i, 6)
    tk.Label(grid, text=f, width=4, anchor="e").grid(row=r, column=c * 2, padx=2, pady=2)
    e = tk.Entry(grid, width=13)
    e.insert(0, "0")
    e.bind("<KeyRelease>", forget_truth)
    e.grid(row=r, column=c * 2 + 1, padx=2, pady=2)
    entries[f] = e

tk.Button(root, text="Predict with all 3 models", width=26, bg="#1f6feb", fg="white",
          command=predict).pack(pady=4)
result = tk.Label(root, text="Load a reading or type values, then press Predict.",
                  font=("Arial", 11))
result.pack()

# three model cards side by side
cards_frame = tk.Frame(root)
cards_frame.pack(fill="x", padx=8, pady=6)
for n in NAMES:
    lf = tk.LabelFrame(cards_frame, text=PRETTY[n], font=("Arial", 12, "bold"), padx=10, pady=4)
    lf.pack(side=tk.LEFT, expand=True, fill="both", padx=6)
    cards[n] = {
        "pred": tk.Label(lf, text="-", font=("Arial", 26, "bold")),
        "meaning": tk.Label(lf, text="", font=("Arial", 10)),
        "verdict": tk.Label(lf, text="", font=("Arial", 11, "bold")),
        "conf": tk.Label(lf, text="Confidence: -", font=("Arial", 10)),
        "acc": tk.Label(lf, text="Test accuracy: calculating...", font=("Arial", 10, "bold")),
        "cv": tk.Label(lf, text=f"Blocked cross-validation accuracy: {CV.get(n, float('nan')):.2f}%",
                       font=("Arial", 10)),
        "tally": tk.Label(lf, text="Right on loaded readings: 0 / 0", font=("Arial", 10)),
    }
    for w_ in cards[n].values():
        w_.pack()

# graphs
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True, padx=8, pady=6)
tab_p, tab_w, tab_m = tk.Frame(nb), tk.Frame(nb), tk.Frame(nb)
nb.add(tab_p, text="Model predictions")
nb.add(tab_w, text="Waveforms")
nb.add(tab_m, text="Noise and importance")

fig_p, canvas_p = make_canvas(tab_p)
fig_w, canvas_w = make_canvas(tab_w)
fig_m, canvas_m = make_canvas(tab_m)
draw_models()
refresh_accuracy()

root.mainloop()