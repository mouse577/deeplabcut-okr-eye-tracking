import sys
import subprocess
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QFileDialog, QMessageBox
)

class DLCToolkit(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLC Toolkit Launcher")
        self.setGeometry(300, 300, 400, 200)

        self.project_path = None

        self.label = QLabel("No project selected")
        self.select_button = QPushButton("Select DLC Project Folder")
        self.train_button = QPushButton("Train Network")
        self.analyze_button = QPushButton("Analyze Videos")
        self.reset_train_button = QPushButton("Reset for Retraining")
        self.reset_analysis_button = QPushButton("Reset for Reanalysis")

        self.train_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.reset_train_button.setEnabled(False)
        self.reset_analysis_button.setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.reset_train_button)
        layout.addWidget(self.reset_analysis_button)
        self.setLayout(layout)

        self.select_button.clicked.connect(self.select_project)
        self.train_button.clicked.connect(self.train_model)
        self.analyze_button.clicked.connect(self.analyze_videos)
        self.reset_train_button.clicked.connect(self.reset_for_training)
        self.reset_analysis_button.clicked.connect(self.reset_for_analysis)

    def select_project(self):
        path = QFileDialog.getExistingDirectory(self, "Select DLC Project Folder")
        if path:
            self.project_path = Path(path)
            self.label.setText(f"Selected: {self.project_path}")
            self.train_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.reset_train_button.setEnabled(True)
            self.reset_analysis_button.setEnabled(True)

    def run_script(self, script_name):
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please select a project folder first.")
            return

        try:
            subprocess.run(["python", script_name], check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Script {script_name} failed:\n{e}")

    def train_model(self):
        self.run_script("train_dlc.py")

    def analyze_videos(self):
        self.run_script("dlc_video_reanalysis.py")

    def reset_for_training(self):
        reply = QMessageBox.question(
            self, "Reset Confirmation",
            "This will delete training models and labeled results. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.run_script("dlc_reset_for_retraining.py")

    def reset_for_analysis(self):
        reply = QMessageBox.question(
            self, "Reset Confirmation",
            "This will delete analysis results (not training data). Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.run_script("dlc_reset_for_reanalysis.py")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = DLCToolkit()
    win.show()
    sys.exit(app.exec_())