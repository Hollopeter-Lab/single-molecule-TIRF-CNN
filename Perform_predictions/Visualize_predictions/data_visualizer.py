import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TraceViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Colocalized Trace Viewer")

        self.df = None
        self.index = 0

        self.load_button = tk.Button(root, text="Load Trace File", command=self.load_file)
        self.load_button.pack()

        self.step_filter_label = tk.Label(root, text="Filter by Predicted Steps:")
        self.step_filter_label.pack()
        self.step_filter_entry = tk.Entry(root)
        self.step_filter_entry.pack()

        self.apply_filter_button = tk.Button(root, text="Apply Filter", command=self.apply_filter)
        self.apply_filter_button.pack()

        self.next_button = tk.Button(root, text="Next", command=self.next_trace)
        self.next_button.pack()

        self.prev_button = tk.Button(root, text="Previous", command=self.prev_trace)
        self.prev_button.pack()

        self.trace_info = tk.Label(root, text="")
        self.trace_info.pack()

        self.coord_info = tk.Label(root, text="")
        self.coord_info.pack()

        self.file_info = tk.Label(root, text="")
        self.file_info.pack()

        self.figure, self.ax = plt.subplots(2, 1, figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.filtered_df = self.df.copy()
            self.index = 0
            self.plot_trace()

    def apply_filter(self):
        try:
            step_val = int(self.step_filter_entry.get())
            self.filtered_df = self.df[self.df['predicted_steps'] == step_val].reset_index(drop=True)
            self.index = 0
            self.plot_trace()
        except ValueError:
            pass  # ignore non-integer input

    def next_trace(self):
        if self.filtered_df is not None and self.index < len(self.filtered_df) - 1:
            self.index += 1
            self.plot_trace()

    def prev_trace(self):
        if self.filtered_df is not None and self.index > 0:
            self.index -= 1
            self.plot_trace()

    def plot_trace(self):
        if self.filtered_df is None or len(self.filtered_df) == 0:
            return
        row = self.filtered_df.iloc[self.index]

        green_trace = eval(row['green_trace']) if isinstance(row['green_trace'], str) else row['green_trace']
        farred_trace = eval(row['farred_trace']) if isinstance(row['farred_trace'], str) else row['farred_trace']

        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].plot(green_trace)
        self.ax[0].set_title("Green Trace")
        self.ax[1].plot(farred_trace)
        self.ax[1].set_title("Far-Red Trace")

        self.trace_info.config(text=f"Index: {self.index+1}/{len(self.filtered_df)} | Predicted Steps: {row['predicted_steps']}")
        self.coord_info.config(text=f"Far-Red Spot: {row['farRed_spotLocation']} | Green Spot: {row['green_spotLocation']}")
        self.file_info.config(text=f"Far-Red File: {row['farRed_sourceFile']} | Green File: {row['green_sourceFile']}")

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TraceViewerApp(root)
    root.mainloop()
