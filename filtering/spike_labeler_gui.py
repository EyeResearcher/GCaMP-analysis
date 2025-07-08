import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Button, Label

class SpikeLabelerApp:
    def __init__(self, raw_trace, spike_prob_trace, spike_prob_indices, raw_f_indices, on_complete, neuron_id=None):
        self.root = Tk()
        self.root.geometry("1000x700")
        self.raw_trace = raw_trace
        self.spike_prob_trace = spike_prob_trace
        self.spike_prob_indices = spike_prob_indices
        self.raw_f_indices = raw_f_indices
        self.labels = []
        self.index = 0
        self.on_complete = on_complete
        self.neuron_id = neuron_id

        main_frame = Label(self.root)
        main_frame.pack()

        self.fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        self.ax_raw, self.ax_prob = axs
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack()

        controls_frame = Label(self.root)
        controls_frame.pack()

        self.label = Label(controls_frame, text="Is this spike real? (Yes/No/Bad ROI)")
        self.label.pack()

        Button(controls_frame, text="Yes", command=lambda: self.label_spike(1)).pack(side="left", padx=10)
        Button(controls_frame, text="No", command=lambda: self.label_spike(0)).pack(side="left", padx=10)
        Button(controls_frame, text="Bad ROI", command=self.bad_roi).pack(side="left", padx=10)

        self.plot_spike()
        self.root.mainloop()

    def plot_spike(self):
        self.ax_raw.clear()
        self.ax_prob.clear()
        idx_prob = self.spike_prob_indices[self.index]
        idx_raw = self.raw_f_indices[self.index]

        # Plot raw trace
        self.ax_raw.plot(self.raw_trace, color='black', label='Raw F', linewidth=0.7)
        self.ax_raw.axvline(idx_raw, color='red', linestyle='--', label='Spike', linewidth=0.7)
        self.ax_raw.set_ylim(bottom=0)  # Start y-axis at zero
        self.ax_raw.set_title(f"Neuron {self.neuron_id} | Spike {self.index+1}/{len(self.spike_prob_indices)} | Raw F")
        self.ax_raw.legend()

        # Plot spike probability trace
        self.ax_prob.plot(self.spike_prob_trace, color='blue', label='Spike Prob', linewidth=0.7)
        self.ax_prob.axvline(idx_prob, color='red', linestyle='--', label='Spike', linewidth=0.7)
        self.ax_prob.set_ylim(bottom=0)  # Start y-axis at zero
        self.ax_prob.set_title("Spike Probability Trace")
        self.ax_prob.legend()

        self.canvas.draw()

    def label_spike(self, label):
        self.labels.append(label)
        if self.index >= len(self.spike_prob_indices) - 1:
            self.on_complete(self.labels)
            self.root.quit()
            self.root.destroy()
            return
        self.index += 1
        self.plot_spike()

    def bad_roi(self):
        self.on_complete("BAD_ROI")
        self.root.quit()
        self.root.destroy()