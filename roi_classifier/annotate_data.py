from tkinter import Button, Tk, Label, Frame
import numpy as np
import matplotlib.pyplot as plt
import json, os, argparse, random
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class Labeler:
    def __init__(self, roi_key: str, raw_f: np.ndarray, smoothed_sp: np.ndarray, features: dict, current_label: int):
        self.roi_key = roi_key
        self.f_trace = raw_f
        self.spike_prob = smoothed_sp
        self.features = features
        self.current_label = current_label
        self.selected_label = None  # Will be set when user clicks button
        

        parts = roi_key.rsplit('_', 1)
        self.video_name = parts[0]
        self.roi_idx = parts[1]
        
        self.root = Tk()
        self.title = f"ROI {self.roi_idx} from Video {self.video_name}"
        self.root.title(self.title)

        
        # Info panel
        info_frame = Frame(self.root)
        info_frame.pack(side ="top", pady=10)
        
        Label(info_frame, text=f"ROI Key: {roi_key}", font=('Arial', 12, 'bold')).pack()
        Label(info_frame, text=f"Current Label: {current_label} ({'Good' if current_label == 1 else 'Bad/Unlabeled' if current_label == 0 else 'Unlabeled'})", 
              font=('Arial', 10)).pack()
        Label(info_frame, text=f"Derivative Skew: {features['derivative_skew']:.4f}", font=('Arial', 10)).pack()
        Label(info_frame, text=f"Spike Prom Mean: {features['spike_prom_mean']:.4f}", font=('Arial', 10)).pack()
        
    
        
        # Controls
        controls_frame = Frame(self.root)
        controls_frame.pack(pady=10)
        
        Label(controls_frame, text="Label this ROI:", font=('Arial', 12, 'bold')).pack()
        
        button_frame = Frame(controls_frame)
        button_frame.pack()
        
        Button(button_frame, text="Good (1)", command=lambda: self.label_roi(1), 
               bg='green', fg='white', font=('Arial', 12, 'bold'), width=15, height=2).pack(side="left", padx=10)
        Button(button_frame, text="Bad (0)", command=lambda: self.label_roi(0), 
               bg='red', fg='white', font=('Arial', 12, 'bold'), width=15, height=2).pack(side="left", padx=10)
        Button(button_frame, text="Skip", command=self.skip_roi, 
               bg='gray', fg='white', font=('Arial', 12, 'bold'), width=15, height=2).pack(side="left", padx=10)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        # Keyboard shortcuts
        self.root.bind('1', lambda e: self.label_roi(1))
        self.root.bind('0', lambda e: self.label_roi(0))
        self.root.bind('<space>', lambda e: self.skip_roi())
        self.root.bind('<Right>', lambda e: self.skip_roi())
        
        self.plot_traces()
        
    def plot_traces(self):
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot smoothed fluorescence
        self.ax1.plot(self.f_trace, color='blue', linewidth=1)
        self.ax1.set_title(f"Smoothed F Trace - {self.title}")
        self.ax1.set_xlabel("Frame #")
        self.ax1.set_ylabel("Normalized F (smoothed)")
        self.ax1.grid(True, alpha=0.3)
        
        # Plot spike probability
        self.ax2.plot(self.spike_prob, color='red', linewidth=1)
        self.ax2.set_title(f"Spike Probability - {self.title}")
        self.ax2.set_xlabel("Frame #")
        self.ax2.set_ylabel("Cascade Spike Prob (smoothed)")
        self.ax2.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def label_roi(self, label):
        self.selected_label = label
        self.root.quit()
        self.root.destroy()
    
    def skip_roi(self):
        self.selected_label = -1  # No change to label
        self.root.quit()
        self.root.destroy()
    
    def show(self):
        """Show the GUI and wait for user input"""
        self.root.mainloop()
        return self.selected_label
    

def save_data(npy_dict: dict, base_path: Path):
    """Save both .npy and .json files"""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save .npy (with traces)
    npy_file = base_path.with_suffix('.npy')
    np.save(npy_file, npy_dict, allow_pickle=True)
    

    
    print(f"Saved to {npy_file} ")

def main():
    parser = argparse.ArgumentParser(description="Annotate ROI data")
    parser.add_argument("--number_annotations", '-n', type=int, help="Number of annotations to perform", default=1000)
    parser.add_argument("--unlabeled_only", action='store_true', 
                       help="Only annotate ROIs with label=-1 (unlabeled)")
    parser.add_argument("--manual_only", action='store_true',
                       help="Only show ROIs that need manual verification (exclude auto-labeled)")
    args = parser.parse_args()

    base_path = Path("training_data/roi_filtering/all_roi_features")
    npy_dict = np.load(base_path.with_suffix('.npy'), allow_pickle=True).item()

    all_roi_keys = list(npy_dict.keys())
    
    # Helper function to get label value
    def get_label_value(label):
        return label['value'] if isinstance(label, dict) else label
    
    def get_label_source(label):
        return label.get('source', 'unknown') if isinstance(label, dict) else 'unknown'
    
    # Filter for unlabeled ROIs if flag is set
    if args.unlabeled_only:
        unlabeled_keys = [k for k in all_roi_keys if get_label_value(npy_dict[k]['label']) == -1]
        print(f"Filtering for unlabeled ROIs: {len(unlabeled_keys)}/{len(all_roi_keys)} are unlabeled")
        
        if len(unlabeled_keys) == 0:
            print("❌ No unlabeled ROIs found!")
            return
        
        # Sample from unlabeled ROIs only
        n_samples = min(args.number_annotations, len(unlabeled_keys))
        selected_keys = random.sample(unlabeled_keys, n_samples)
        print(f"Sampling {n_samples} unlabeled ROIs")
    elif args.manual_only:
        # Only show ROIs that aren't manually verified
        needs_manual = [k for k in all_roi_keys 
                       if get_label_source(npy_dict[k]['label']) != 'manual']
        print(f"Filtering for ROIs needing manual verification: {len(needs_manual)}/{len(all_roi_keys)}")
        
        if len(needs_manual) == 0:
            print("❌ All ROIs are manually verified!")
            return
        
        n_samples = min(args.number_annotations, len(needs_manual))
        selected_keys = random.sample(needs_manual, n_samples)
        print(f"Sampling {n_samples} ROIs that need manual verification")
    else:
        # Sample from all ROIs
        total_available = len(all_roi_keys)
        n_samples = min(args.number_annotations, total_available)
        selected_keys = random.sample(all_roi_keys, n_samples)
        print(f"Sampling {n_samples} ROIs from all {total_available} ROIs")

    total_rois = len(npy_dict)
    labeled_count = 0
    skipped_count = 0
    updated_count = 0

    for idx, roi_key in enumerate(selected_keys, start=1):
        roi_data = npy_dict[roi_key]
        current_label = get_label_value(roi_data['label'])
        current_source = get_label_source(roi_data['label'])
        
        label_str = 'Good' if current_label == 1 else 'Bad' if current_label == 0 else 'Unlabeled'
        source_str = f"({current_source})" if current_source != 'unknown' else ""
        
        print(f"\n[{idx}/{n_samples}] Annotating ROI: {roi_key} | Current Label: {current_label} {label_str} {source_str}")
        
        labeler = Labeler(roi_key, roi_data['raw_traces'][0], roi_data['smoothed_traces'][1],
                          roi_data['features'], current_label)
        selected_label = labeler.show()
        
        if selected_label is not None:
            if selected_label == -1:
                skipped_count += 1
                print(f"Skipped ROI: {roi_key}")
            else:
                if selected_label != current_label:
                    # Ensure label is dict format
                    if not isinstance(npy_dict[roi_key]['label'], dict):
                        npy_dict[roi_key]['label'] = {'value': selected_label, 'source': 'manual'}
                    else:
                        npy_dict[roi_key]['label']['value'] = selected_label
                        npy_dict[roi_key]['label']['source'] = 'manual'  # Mark as manually verified
                    
                    updated_count += 1
                    print(f"Updated ROI: {roi_key} to Label: {selected_label} ({'Good' if selected_label == 1 else 'Bad'}) [manual]")
                else:
                    # Even if label value unchanged, mark as manually verified
                    if isinstance(npy_dict[roi_key]['label'], dict):
                        npy_dict[roi_key]['label']['source'] = 'manual'
                    else:
                        npy_dict[roi_key]['label'] = {'value': selected_label, 'source': 'manual'}
                    print(f"Confirmed ROI: {roi_key} - Label unchanged but marked as manual")
            labeled_count += 1
        else:
            print(f"No label selected for ROI: {roi_key}")
            
        if (labeled_count + updated_count) % 30 == 0:
            save_data(npy_dict, base_path)
            print(f"📦 Checkpoint: Saved progress ({labeled_count + updated_count} total changes)")
    
    print("\n" + "="*50)
    save_data(npy_dict, base_path)
    
    # Count final statistics
    total_labeled_rois = sum(1 for v in npy_dict.values() if get_label_value(v['label']) != -1)
    total_unlabeled_rois = sum(1 for v in npy_dict.values() if get_label_value(v['label']) == -1)
    manual_rois = sum(1 for v in npy_dict.values() if get_label_source(v['label']) == 'manual')
    auto_rois = sum(1 for v in npy_dict.values() if get_label_source(v['label']) == 'auto')
    
    print(f"\n=== Annotation Complete ===")
    print(f"ROIs sampled: {n_samples}")
    print(f"Newly labeled: {labeled_count}")
    print(f"Updated: {updated_count}")
    print(f"Skipped: {skipped_count}")
    print(f"\nDataset Summary:")
    print(f"  Total labeled:   {total_labeled_rois}/{len(all_roi_keys)}")
    print(f"  Total unlabeled: {total_unlabeled_rois}/{len(all_roi_keys)}")
    print(f"  Manual labels:   {manual_rois}")
    print(f"  Auto labels:     {auto_rois}")


if __name__ == "__main__":
    main()