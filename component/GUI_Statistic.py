from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton, 
                             QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QMessageBox, QToolTip,QTextEdit)

from PyQt5.QtGui import QCursor, QPixmap

import sys 
import os

from .styles_pyqtgraph import graph_style_chromatogram
import pyqtgraph as pg
from .fun_Groupmaker import GroupMaker
from .GUI_components import CheckableComboBox
import numpy as np
import pandas as pd

from copy import copy

import uuid
import logging
from datetime import datetime

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class StatisticalWindow(QDialog):
    def __init__(self, results, file_names, parent):
        super().__init__(parent)
        # Set Window Settings
        
        self.setWindowTitle('Statistical Analysis')
        self.setMinimumSize(400, 400)
        

        layout = QGridLayout()
        self.setLayout(layout)


        # Set window UI components
        
        # Add Box for Plotting Stuff
        PlottingGroupBox = QGroupBox("Plotting",self)
        PlottingLayout = QGridLayout(PlottingGroupBox)

        # self.PlottingWidget = pg.PlotWidget()
        # graph_style_chromatogram(self.PlottingWidget)
        # PlottingLayout.addWidget(self.PlottingWidget, 0, 0)

        # Create a matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        PlottingLayout.addWidget(self.canvas, 0, 0)

        PlottingGroupBox.setLayout(PlottingLayout)
        layout.addWidget(PlottingGroupBox, 0, 0)


        # Add Box for Text Output
        OutputGroupBox = QGroupBox("Output",self)
        OutputLayout = QGridLayout(OutputGroupBox)

        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)
        OutputLayout.addWidget(self.text_output, 0, 0)

        OutputGroupBox.setLayout(OutputLayout)
        layout.addWidget(OutputGroupBox, 0, 1)


        # Add Box for Bottons/Parameter Selection

        SettingsGroupBox = QGroupBox("Settings",self)
        SettingsLayout = QGridLayout(SettingsGroupBox)

        self.dendrogram_button = QPushButton("Dendrogram", self)
        self.dendrogram_button.clicked.connect(lambda: self.plot_dendrogram(self.score_df, self.file_names))
        SettingsLayout.addWidget(self.dendrogram_button, 0, 0)

        self.kmeans_button = QPushButton("K-Means Clustering", self)
        self.kmeans_button.clicked.connect(lambda: self.plot_kmeans(self.score_df, 4, self.file_names))
        SettingsLayout.addWidget(self.kmeans_button, 1, 0)

        SettingsGroupBox.setLayout(SettingsLayout)
        layout.addWidget(SettingsGroupBox, 0, 2)


        # Initialize the class variables

        self.results = results
        self.score_df = pd.DataFrame.from_dict(self.results['scores'], orient="index")
        self.score_df.columns = [f"PC{i+1}" for i in range(self.score_df.shape[1])]
        print(self.score_df.head())
        print(self.score_df.dtypes)
        self.file_names = file_names

        self.Groups, self.GroupList = GroupMaker(self.file_names, keyword='GroupList')

        print(self.Groups)
        print(self.GroupList)

        for idx, group in self.Groups.items():
            # add each grouplist to the dataframe as a new column as categorical data
            self.score_df[f"Group {idx}"] = pd.Categorical(self.GroupList[idx], categories=group, ordered=True)
            self.text_output.append(f"Group {idx}: {group}")

        print(self.score_df.head())

        print(self.score_df.dtypes)

        filtered_df = self.score_df.select_dtypes(exclude=['category'])
        self.plot_dendrogram(filtered_df, self.file_names)



    def plot_dendrogram(self, data, sample_labels):
        # Clear the previous figure
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        # exclude the categorical data
        data = data.select_dtypes(exclude=['category'])
        
        # Compute distance and linkage matrix
        dist_matrix = pdist(data, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method='ward')
        
        # Plot the dendrogram
        dendrogram(linkage_matrix, labels=sample_labels, ax=ax, leaf_rotation=90)
        ax.set_title("Dendrogram (Ward's Method)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Distance")
        
        # Refresh canvas
        self.canvas.draw()

    def plot_kmeans(self, data, k, sample_labels):
        # clear the previous figure
        self.figure.clf()
        ax = self.figure.add_subplot(111)


        # exclude the categorical data
        data = data.select_dtypes(exclude=['category']).values

        # Compute k-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        y_kmeans = kmeans.predict(data)

        # Plot the k-means clusters
        ax.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')

        # Plot the centroids
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

        # Refresh canvas
        self.canvas.draw()


