import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import dlc_manager

class DLCGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DLC Training & Analysis GUI")
        self.project_path = None

        # === GUI Elements ===
        self.path_label = tk.Label(root, text="No project selected", fg="gray")
        self.path_label.pack(pady=10)

        tk.Button(root, text="Select DLC Project Folder", command=self.select_project).pack(pady=5)

        self.train_btn = tk.Button(root, text="Train Model", command=self.train_model, state="disabled")
        self.train_btn.pack(pady=5)

        self.analyze_btn = tk.Button(root, text="Analyze Videos", command=self.analyze_videos, state="disabled")
        self.analyze_btn.pack(pady=5)

        self.label_btn = tk.Button(root, text="Create Labeled Videos", command=self.create_labeled, state="disabled")
        self.label_btn.pack(pady=5)

        self.conf_btn = tk.Button(root, text="Create Confidence Overlay Videos", command=self.create_confidence, state="disabled")
        self.conf_btn.pack(pady=5)

        self.retrain_btn = tk.Button(root, text="Reset for Retraining", command=self.reset_retrain, state="disabled")
        self.retrain_btn.pack(pady=5)

        self.reanalyze_btn = tk.Button(root, text="Reset for Reanalysis", command=self.reset_reanalyze, state="disabled")
        self.reanalyze_btn.pack(pady=5)

    def select_project(self):
        path = filedialog.askdirectory(title="Select DLC Project Folder")
        if path:
            self.project_path = Path(path)
            self.path_label.config(text=str(self.project_path), fg="black")
            for btn in [self.train_btn, self.analyze_btn, self.label_btn, self.conf_btn, self.retrain_btn, self.reanalyze_btn]:
                btn.config(state="normal")

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


if __name__ == "__main__":
    root = tk.Tk()
    app = DLCGuiApp(root)
    root.mainloop()