"""
HOME / MENU PAGE: defines the home page or main menu that the user will see first,
with buttons to access different tools
o   this script inherits from QWidget,
o   contains buttons or links to the various tools/pages, and
o   emits signals or calls callbacks to tell MainWindow to switch pages
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

class MainMenu(QWidget):
    def __init__(self, show_grid2table, show_dose_response, show_visualise):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Select a tool:'))

        # add buttons to navigate to tools
        self.grid2table_btn = QPushButton('Grid2Table')
        self.grid2table_btn.clicked.connect(show_grid2table)
        self.dose_response_btn = QPushButton('Measure Dose Response')
        self.dose_response_btn.clicked.connect(show_dose_response)
        self.visualise_btn = QPushButton('Visualise Plot')
        self.visualise_btn.clicked.connect(show_visualise)

        layout.addWidget(self.grid2table_btn)
        layout.addWidget(self.dose_response_btn)
        layout.addWidget(self.visualise_btn)

        self.setLayout(layout)