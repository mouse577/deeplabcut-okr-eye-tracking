import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import dlc_manager
from dlc_manager import generate_eye_tracking_pdf

class DLCGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DLC Training & Analysis GUI")
        self.project_path = None
        self.ttl_path = None

        # === Layout Frames ===
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", padx=20)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", padx=20)

        # === LEFT SIDE: DLC Buttons ===
        self.path_label = tk.Label(left_frame, text="No project selected", fg="gray")
        self.path_label.pack(pady=10)

        tk.Button(left_frame, text="Select DLC Project Folder", command=self.select_project).pack(pady=5)

        self.train_btn = tk.Button(left_frame, text="Train Model", command=self.train_model, state="disabled")
        self.train_btn.pack(pady=5)

        self.analyze_btn = tk.Button(left_frame, text="Analyze Videos", command=self.analyze_videos, state="disabled")
        self.analyze_btn.pack(pady=5)

        self.label_btn = tk.Button(left_frame, text="Create Labeled Videos", command=self.create_labeled, state="disabled")
        self.label_btn.pack(pady=5)

        self.conf_btn = tk.Button(left_frame, text="Create Confidence Overlay Videos", command=self.create_confidence, state="disabled")
        self.conf_btn.pack(pady=5)

        self.retrain_btn = tk.Button(left_frame, text="Reset for Retraining", command=self.reset_retrain, state="disabled")
        self.retrain_btn.pack(pady=5)

        self.reanalyze_btn = tk.Button(left_frame, text="Reset for Reanalysis", command=self.reset_reanalyze, state="disabled")
        self.reanalyze_btn.pack(pady=5)

        # === RIGHT SIDE: TTL Stimulus Buttons ===
        tk.Label(right_frame, text="TTL Stimulus Tools", font=("Arial", 12, "bold")).pack(pady=10)

        self.ttl_select_btn = tk.Button(right_frame, text="Select TTL Folder", command=self.select_ttl_folder)
        self.ttl_select_btn.pack(pady=5)

        self.ttl_extract_btn = tk.Button(right_frame, text="Extract Stimulus Frames", command=self.extract_stimulus_frames)
        self.ttl_extract_btn.pack(pady=5)

        self.plot_pdf_btn = tk.Button(right_frame, text="Generate Eye Tracking PDF", command=self.generate_pdf, state="disabled")
        self.plot_pdf_btn.pack(pady=5)

    def select_project(self):
        path = filedialog.askdirectory(title="Select DLC Project Folder")
        if path:
            self.project_path = Path(path)
            self.path_label.config(text=str(self.project_path), fg="black")
            for btn in [self.train_btn, self.analyze_btn, self.label_btn, self.conf_btn, self.retrain_btn, self.reanalyze_btn, self.plot_pdf_btn]:
                btn.config(state="normal")

    def select_ttl_folder(self):
        path = filedialog.askdirectory(title="Select TTL Folder")
        if path:
            self.ttl_path = Path(path)
            messagebox.showinfo("TTL Folder Set", f"Selected: {self.ttl_path}")

    def extract_stimulus_frames(self):
        if not self.project_path or not self.ttl_path:
            messagebox.showerror("Error", "Select both DLC project folder and TTL folder first.")
            return
        try:
            dlc_manager.extract_stimulus_frames(self.ttl_path, self.project_path / "videos")
            messagebox.showinfo("Done", "Stimulus frame analysis complete.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def train_model(self):
        if not self.project_path: return
        dlc_manager.train(self.project_path)
        messagebox.showinfo("Done", "Training complete.")

    def analyze_videos(self):
        if not self.project_path: return
        dlc_manager.analyze_videos(self.project_path / "config.yaml")
        messagebox.showinfo("Done", "Analysis complete.")

    def create_labeled(self):
        if not self.project_path: return
        dlc_manager.create_labeled_videos(self.project_path)
        messagebox.showinfo("Done", "Labeled videos created.")

    def create_confidence(self):
        if not self.project_path: return
        dlc_manager.create_conf_overlay(self.project_path)
        messagebox.showinfo("Done", "Confidence overlay videos created.")

    def reset_retrain(self):
        if not self.project_path: return
        dlc_manager.reset_for_retraining(self.project_path)
        messagebox.showinfo("Done", "Reset for retraining complete.")

    def reset_reanalyze(self):
        if not self.project_path: return
        dlc_manager.reset_for_reanalysis(self.project_path)
        messagebox.showinfo("Done", "Reset for reanalysis complete.")

    def generate_pdf(self):
        if not self.project_path:
            messagebox.showerror("Error", "No project selected.")
            return
        generate_eye_tracking_pdf(self.project_path)
        messagebox.showinfo("Done", f"PDF saved to:\n{self.project_path}/eye_tracking_summary_plots.pdf")

if __name__ == "__main__":
    root = tk.Tk()
    app = DLCGuiApp(root)
    root.mainloop()
