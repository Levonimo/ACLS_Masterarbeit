from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton, 
                             QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QMessageBox, QToolTip,QTextEdit, QDoubleSpinBox)

from PyQt5.QtGui import QCursor, QPixmap

import sys 
import os
os.environ["OMP_NUM_THREADS"] = "1"

from .styles_pyqtgraph import graph_style_chromatogram
import pyqtgraph as pg
from .fun_Groupmaker import GroupMaker
from .fun_Statistic import (preprocess_data, compute_cophenetic_correlation, gap_statistic,
                            compute_cluster_stability, compute_silhouette_score, compute_dunn_index)
from .functions import assign_colors
from .GUI_Selection import GroupSelectionWindow
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
    def __init__(self, results, parent):
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

        self.analyze_button = QPushButton("Analyze Clustering", self)
        self.analyze_button.clicked.connect(lambda: self.analyze_clustering(self.score_df))
        SettingsLayout.addWidget(self.analyze_button, 2, 0)

        # add Button for select Group for coloring
        self.select_coloring_button = QPushButton('Select Coloring', self)
        SettingsLayout.addWidget(self.select_coloring_button, 3, 0, 1, 1)
        self.select_coloring_button.clicked.connect(self.select_coloring)


        # add dropdown menu for diffrent supervised maschine learning algorithms
        self.algorithm_label = QLabel("Size Validation Set", self)
        SettingsLayout.addWidget(self.algorithm_label, 4, 0, 1, 1)
        # Using QDoubleSpinBox restricts the input to float values between 0.0 and 0.9
        self.size_validation_set = QDoubleSpinBox(self)
        self.size_validation_set.setRange(0.0, 0.9)
        self.size_validation_set.setSingleStep(0.05)
        SettingsLayout.addWidget(self.size_validation_set, 5, 0, 1, 1)
        
    

        # add dropdown menu for diffrent supervised maschine learning algorithms
        self.algorithm_label = QLabel("Supervised Learning Algorithm", self)
        SettingsLayout.addWidget(self.algorithm_label, 6, 0)
        self.algorithm_combobox = QComboBox(self)

        self.algorithm_combobox.addItem("Naive Bayes")
        self.algorithm_combobox.addItem("K-Nearest Neighbors")
        self.algorithm_combobox.addItem("Linear Discriminant Analysis")

        self.algorithm_combobox.addItem("Decision Tree")
        self.algorithm_combobox.addItem("Random Forest")
        self.algorithm_combobox.addItem("Gradient Boosting")

        SettingsLayout.addWidget(self.algorithm_combobox, 7, 0)

        self.ml_button = QPushButton("Supervised ML", self)
        SettingsLayout.addWidget(self.ml_button, 8, 0)
        self.ml_button.clicked.connect(self.run_ml)

        SettingsGroupBox.setLayout(SettingsLayout)
        layout.addWidget(SettingsGroupBox, 0, 2)


        # Initialize the class variables
        self.parent = parent
        self.colors = None
        
        # Load the results from the PCA analysis and convert to a DataFrame
        self.results = results
        self.score_df = pd.DataFrame.from_dict(self.results['scores'], orient="index")
        self.score_df.columns = [f"PC{i+1}" for i in range(self.score_df.shape[1])]
        self.file_names = self.score_df.index.tolist()

        # Create a dictionary of groups and a list of group names
        self.Groups, self.GroupList = GroupMaker(self.file_names, keyword='GroupList')

        # Add the group columns to the DataFrame
        for idx, group in self.Groups.items():
            # add each grouplist to the dataframe as a new column as categorical data
            self.score_df[f"Group {idx}"] = pd.Categorical(self.GroupList[idx], categories=group, ordered=True)
            self.text_output.append(f"Group {idx}: {group}")

        # Plot the first dendrogram
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

        # coloring the labels with the group colors
        if self.colors is not None:
            first_color = list(self.colors.keys())[0]
            groupname = next((f"Group {key}" for key, group in self.Groups.items() if first_color in group), None)
            
            # Recompute dendrogram to capture the ordering of leaves
            ddata = dendrogram(linkage_matrix, labels=sample_labels, ax=ax, leaf_rotation=90)
            leaves_order = ddata["leaves"]
            
            # Iterate over the leaves in the plotted order to update tick label colors
            for pos, leaf_idx in enumerate(leaves_order):
                sample = sample_labels[leaf_idx]
                tick_label = ax.get_xticklabels()[pos]
                tick_label.set_color(self.colors[self.score_df[groupname].loc[sample]])

        # Reduce white space around the plot
        self.figure.subplots_adjust(top=0.94, bottom=0.2, left=0.05, right=0.99)
        
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
        #
        # coloring the labels with the group colors
        if self.colors is not None:
            # Identify the group column that corresponds to the selected coloring
            first_color = list(self.colors.keys())[0]
            groupname = next((f"Group {key}" for key, group in self.Groups.items() if first_color in group), None)
            
            # Create a list of colors for each sample based on its group assignment
            point_colors = [self.colors[self.score_df[groupname].loc[sample]] for sample in sample_labels]
            
            # Clear the existing scatter plot and redraw the points with the new group colors
            ax.scatter(data[:, 0], data[:, 1], c=point_colors, s=50)
        else:
            ax.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=50, cmap='viridis')


        # Plot the centroids
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        import matplotlib.cm as cm
        import matplotlib.patches as mpatches

        colors = [cm.viridis(float(i)/k) for i in range(k)]
        for i in range(k):
            cluster_points = data[y_kmeans == i]
            # Calculate the maximum distance from the cluster center to its points
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            radius = distances.max()
            circle = mpatches.Circle(centers[i],
                                     radius*0.9,
                                     color=colors[i],
                                     alpha=0.2,
                                     zorder=0)
            ax.add_patch(circle)
        # Refresh canvas
        self.canvas.draw()




    def analyze_clustering(self, df):
        """
        Runs all clustering significance checks and prints results.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing PCA scores.
        """
        # Preprocess data
        pc_data, sample_names = preprocess_data(df)

        # Compute statistics
        cophenetic_corr = compute_cophenetic_correlation(pc_data)
        optimal_clusters = gap_statistic(pc_data)
        stability_score = compute_cluster_stability(pc_data, optimal_clusters)
        silhouette = compute_silhouette_score(pc_data, optimal_clusters)
        dunn_index = compute_dunn_index(pc_data, optimal_clusters)

        # Print results to output
        self.text_output.append(f"Cophenetic Correlation: {cophenetic_corr:.4f}")
        self.text_output.append(f"Optimal Clusters (Gap Statistic): {optimal_clusters}")
        self.text_output.append(f"Cluster Stability Score: {stability_score:.4f}")
        self.text_output.append(f"Silhouette Score: {silhouette:.4f}")
        self.text_output.append(f"Dunn Index: {dunn_index:.4f}")


    def select_coloring(self):
        # open a new window with the group selection
        dialog = GroupSelectionWindow(list(self.Groups.values()), parent = self)
        if dialog.exec_():
            self.group_for_color = dialog.selected_group
            if self.group_for_color is None:
                self.colors = None
            else:
                self.colors = assign_colors(self.group_for_color)

            self.text_output.append(f"Selected Group: {self.colors}")

    
    def run_ml(self):
        # Get the selected algorithm
        algorithm = self.algorithm_combobox.currentText()

        # Get the data
        data = self.score_df.select_dtypes(exclude=['category'])

        first_color = list(self.colors.keys())[0]
        groupname = next((f"Group {key}" for key, group in self.Groups.items() if first_color in group), None)
            
        print(groupname)
        target = self.score_df[groupname]

        if self.size_validation_set.value() > 0:
            from sklearn.model_selection import train_test_split
            data, val_data, target, val_target = train_test_split(data, target, test_size=self.size_validation_set.value(), random_state=42)
            

        # Run the selected algorithm
        if algorithm == "Naive Bayes":
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            self.text_output.append("Running Naive Bayes Classifier")
        elif algorithm == "K-Nearest Neighbors":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
            self.text_output.append("Running K-Nearest Neighbors Classifier")
        elif algorithm == "Linear Discriminant Analysis":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
            self.text_output.append("Running Linear Discriminant Analysis")
        elif algorithm == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
            self.text_output.append("Running Decision Tree Classifier")
        elif algorithm == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()
            self.text_output.append("Running Random Forest Classifier")
        elif algorithm == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier()
            self.text_output.append("Running Gradient Boosting Classifier")

        model.fit(data, target)
        accuracy = model.score(data, target)
        accuracy_val = model.score(val_data, val_target)

        self.text_output.append(f"Accuracy: {accuracy:.4f}")
        self.text_output.append(f"Accuracy Validation: {accuracy_val:.4f}")

        # Print the confusion matrix
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(val_data)
        cm = confusion_matrix(val_target, y_pred)
        # calculate the relative confusion matrix
        #cm = np.round(cm / cm.sum(axis=1)[:, np.newaxis],2)
        self.text_output.append(f"Confusion Matrix:\n{cm}")
        # Plot the confusion matrix
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        cax = ax.matshow(cm, cmap='viridis')
        # add colorbar and adjust it from 0 to 1
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        # set the x and y ticks
        # Use the target group names as tick labels, ensuring all elements are properly positioned
        tick_labels = list(self.score_df[groupname].cat.categories)
        ticks = np.arange(len(tick_labels))
        # Set the extent so that all cells (and thus all labels) are centered correctly.
        #cax = ax.matshow(cm, cmap='viridis', extent=[-0.5, len(tick_labels) - 0.5, len(tick_labels) - 0.5, -0.5])
        self.figure.colorbar(cax)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_yticklabels(tick_labels)
        self.figure.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.99)

        # Refresh canvas
        self.canvas.draw()

        # Print the classification report
        from sklearn.metrics import classification_report

        report = classification_report(val_target, y_pred)
        self.text_output.append(f"Classification Report:\n{report}")

        # Print the feature importances
        if algorithm in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            feature_importances = model.feature_importances_
            self.text_output.append(f"Feature Importances:\n{feature_importances}")

        # Plot the decision boundaries
        if data.shape[1] == 2:
            self.plot_decision_boundaries(val_data, val_target, model)
