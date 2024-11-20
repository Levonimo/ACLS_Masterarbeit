# pyqtgraph_style.py

import pyqtgraph as pg

def graph_style_chromatogram(plot_widget):
    # Hintergrundfarbe
    plot_widget.setBackground((255, 255, 255))
    
    # Achsenfarben
    plot_widget.getAxis('left').setPen(pg.mkPen(color=(0, 0, 0)))
    plot_widget.getAxis('bottom').setPen(pg.mkPen(color=(0, 0, 0)))
    
    # Textfarbe für Achsen
    plot_widget.getAxis('left').setTextPen(pg.mkPen(color=(0, 0, 0)))
    plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=(0, 0, 0)))

