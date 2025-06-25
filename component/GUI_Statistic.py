"""GUI for statistical analysis plots."""

from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton,
                             QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QMessageBox, QToolTip,QTextEdit, QDoubleSpinBox, QSpinBox)
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QObject, QEvent

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

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)



class StatisticalWindow(QDialog):
    def __init__(self, results, parent):
        super().__init__(parent)
        # Set Window Settings
        
        self.setWindowTitle('Statistical Analysis')
        self.setMinimumSize(900, 700)  # Größeres Standardfenster
        

        layout = QGridLayout()
        #layout.setSpacing(15)  # Mehr Abstand zwischen den Elementen
        self.setLayout(layout)

        # Spalten-Stretching einstellen - gleiche Breite für Spalten 1 und 2
        layout.setColumnStretch(0, 2)  # Plotting bekommt mehr horizontalen Platz
        layout.setColumnStretch(1, 1)  # Clustering
        layout.setColumnStretch(2, 1)  # ML - gleiche Größe wie Clustering
        
        # Zeilen-Stretching einstellen - macht Output-Fenster vertikal expandierbar
        layout.setRowStretch(0, 3)  # Obere Zeile (mit Plotting und Steuerungen) mehr Platz
        layout.setRowStretch(1, 2)  # Untere Zeile (mit Output) - vertikal expandierbar

        # Gemeinsame Größe für Clustering und ML definieren
        cluster_width = 200
        control_height = 230

        # Add Box for Plotting Stuff
        PlottingGroupBox = QGroupBox("Plotting", self)
        PlottingLayout = QGridLayout(PlottingGroupBox)
        
        # Plotting soll großzügig Platz bekommen und mitwachsen
        PlottingGroupBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        PlottingGroupBox.setMinimumSize(500, 400)  # Mindestgröße festlegen

        # Create a matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        PlottingLayout.addWidget(self.canvas, 0, 0)

        PlottingGroupBox.setLayout(PlottingLayout)
        layout.addWidget(PlottingGroupBox, 0, 0, 3, 1)  # Span über 3 Zeilen


        # Add Box for Text Output
        OutputGroupBox = QGroupBox("Output", self)
        OutputLayout = QGridLayout(OutputGroupBox)
        
        # Output ist in beide Richtungen expandierbar, aber wird quadratisch gehalten
        # OutputGroupBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Breite = Clustering + ML
        
        # # Dieser Event-Filter sorgt dafür, dass die GroupBox quadratisch bleibt
        # class SquareResizeFilter(QObject):
        #     def eventFilter(self, obj, event):
        #         if event.type() == QEvent.Resize:
        #             width = obj.width()
        #             obj.setMinimumHeight(width)  # Höhe an Breite anpassen
        #         return super().eventFilter(obj, event)
                
        # self.square_filter = SquareResizeFilter(self)
        # OutputGroupBox.installEventFilter(self.square_filter)

        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)
        OutputLayout.addWidget(self.text_output, 0, 0)

        OutputGroupBox.setLayout(OutputLayout)
        layout.addWidget(OutputGroupBox, 2, 1, 1, 2)  # Span über 2 Spalten (Clustering + ML)


        # Add Box for Buttons/Parameter Clustering
        ClusteringGroupBox = QGroupBox("Clustering", self)
        ClusterLayout = QGridLayout(ClusteringGroupBox)
        
        # Gemeinsame Größe für Clustering und ML
        ClusteringGroupBox.setFixedSize(200, control_height)

        # add Button for select Group for coloring
        self.select_coloring_button = QPushButton('Select Coloring', self)
        ClusterLayout.addWidget(self.select_coloring_button, 0, 0)
        self.select_coloring_button.clicked.connect(self.select_coloring)

        self.dendrogram_button = QPushButton("Dendrogram", self)
        self.dendrogram_button.clicked.connect(lambda: self.plot_dendrogram(self.score_df, self.file_names))
        ClusterLayout.addWidget(self.dendrogram_button, 1, 0)

        self.kmeans_button = QPushButton("K-Means Clustering", self)
        self.kmeans_button.clicked.connect(lambda: self.plot_kmeans(self.score_df, 4, self.file_names))
        ClusterLayout.addWidget(self.kmeans_button, 2, 0)

        self.analyze_button = QPushButton("Analyze Clustering", self)
        self.analyze_button.clicked.connect(lambda: self.analyze_clustering(self.score_df))
        ClusterLayout.addWidget(self.analyze_button, 3, 0)

        # Add button to save the current plot
        self.save_button = QPushButton("Save Plot", self)
        self.save_button.clicked.connect(self.save_current_plot)
        ClusterLayout.addWidget(self.save_button, 4, 0)

        ClusteringGroupBox.setLayout(ClusterLayout)
        layout.addWidget(ClusteringGroupBox, 0, 1)


        # Add Box for Buttons/Parameter Machine Learning
        MLGroupBox = QGroupBox("Machine Learning", self)
        MLLayout = QGridLayout(MLGroupBox)
        
        # Exakt gleiche Größe wie Clustering
        MLGroupBox.setFixedSize(cluster_width, control_height)

        # add dropdown menu for different supervised machine learning algorithms
        self.algorithm_label = QLabel("Algorithm:", self)
        MLLayout.addWidget(self.algorithm_label, 0, 0)
        self.algorithm_combobox = QComboBox(self)

        self.algorithm_combobox.addItem("Naive Bayes")
        self.algorithm_combobox.addItem("K-Nearest Neighbors")
        self.algorithm_combobox.addItem("Linear Discriminant Analysis")
        self.algorithm_combobox.addItem("Decision Tree")
        self.algorithm_combobox.addItem("Random Forest")
        self.algorithm_combobox.addItem("Gradient Boosting")

        MLLayout.addWidget(self.algorithm_combobox, 0, 1)

        # Add selection tool for number of PCs
        self.pc_label = QLabel("Number PC's:", self)
        MLLayout.addWidget(self.pc_label, 1, 0)
        self.pc_spinbox = QSpinBox(self)
        self.pc_spinbox.setRange(1, len(results['scores'][list(results['scores'].keys())[0]]))
        self.pc_spinbox.setValue(len(results['scores'][list(results['scores'].keys())[0]]))
        self.pc_spinbox.valueChanged.connect(self.update_pc_selection)
        MLLayout.addWidget(self.pc_spinbox, 1, 1)

        # add dropdown menu for different supervised machine learning algorithms
        self.algorithm_label = QLabel("Size Val. Set:", self)
        MLLayout.addWidget(self.algorithm_label, 2, 0, 1, 1)
        # Using QDoubleSpinBox restricts the input to float values between 0.0 and 0.9
        self.size_validation_set = QDoubleSpinBox(self)
        self.size_validation_set.setRange(0.0, 0.9)
        self.size_validation_set.setSingleStep(0.05)
        MLLayout.addWidget(self.size_validation_set, 2, 1, 1, 1)
        
        # add button to run the selected supervised machine learning algorithm
        self.ml_button = QPushButton("Run ML", self)
        MLLayout.addWidget(self.ml_button, 3, 0, 1, 2)
        self.ml_button.clicked.connect(self.run_ml)

        MLGroupBox.setLayout(MLLayout)
        layout.addWidget(MLGroupBox, 0, 2, 1, 1)
        

         # Add Box for Buttons/Parameter Machine Learning
        ClusterTestGroupBox = QGroupBox("Test Cluster Assignments", self)
        # set fixed height
        ClusterTestGroupBox.setFixedHeight(100)
        ClusterTestLayout = QGridLayout(ClusterTestGroupBox)

        # add dropdown menu for different supervised machine learning algorithms
        self.cluster_label = QLabel("Clustering:", self)
        ClusterTestLayout.addWidget(self.cluster_label, 0, 0)
        self.cluster_combobox = QComboBox(self)

        self.cluster_combobox.addItem("K-Means")
        self.cluster_combobox.addItem("Hierarchical")
        self.cluster_combobox.addItem("DBSCAN")
        self.cluster_combobox.addItem("Gaussian Mixture Model")

        ClusterTestLayout.addWidget(self.cluster_combobox, 0, 1)


        # add button to run the selected supervised machine learning algorithm
        self.cluster_button = QPushButton("Run Assignment Test", self)
        ClusterTestLayout.addWidget(self.cluster_button, 0, 2, 1, 1)
        self.cluster_button.clicked.connect(self.run_cluster_test)

        # Add button to save the current plot
        self.save_csv_button = QPushButton("  Test all Assignments - Save as CSV  ", self)
        ClusterTestLayout.addWidget(self.save_csv_button, 1, 2, 1, 1)
        self.save_csv_button.clicked.connect(self.save_as_csv)

        ClusterTestGroupBox.setLayout(ClusterTestLayout)
        layout.addWidget(ClusterTestGroupBox, 1, 1, 1, 2)  # Über beide Zeilen (Clustering + ML)

        # Initialize the class variables
        self.parent = parent
        self.colors = None
        
        # Load the results from the PCA analysis and convert to a DataFrame
        self.results = results
        self.score_df = pd.DataFrame.from_dict(self.results['scores'], orient="index")
        self.score_df.columns = [f"PC{i+1}" for i in range(self.score_df.shape[1])]
        self.file_names = self.score_df.index.tolist()

        # Create a dictionary of groups and a list of group names
        # self.Groups, self.GroupList = GroupMaker(self.file_names, keyword='GroupList')

        # # Add the group columns to the DataFrame
        # for idx, group in self.Groups.items():
        #     # add each grouplist to the dataframe as a new column as categorical data
        #     self.score_df[f"Group {idx}"] = pd.Categorical(self.GroupList[idx], categories=group, ordered=True)
        #     self.text_output.append(f"Group {idx}: {group}")

        # Adjust the number of PCs based on the selection
        self.update_pc_selection()
        

        # Plot the first dendrogram
        filtered_df = self.score_df.select_dtypes(exclude=['category'])
        self.plot_dendrogram(filtered_df, self.file_names)



    def plot_dendrogram(self, data, sample_labels):
        self.update_pc_selection()
        # Clear the previous figure
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        # Exclude categorical data
        data = data.select_dtypes(exclude=['category'])
        
        # Compute distance and linkage matrix
        dist_matrix = pdist(data, metric='euclidean')
        linkage_matrix = linkage(dist_matrix, method='ward')
        
        # Plot the dendrogram and capture leaf ordering
        ddata = dendrogram(linkage_matrix, labels=sample_labels, ax=ax, leaf_rotation=90)
        
        ax.set_title("Dendrogram (Ward's Method)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Distance")

        # Color the tick labels if coloring information is available
        if self.colors is not None:
            first_color = list(self.colors.keys())[0]
            groupname = next((f"Group {key}" for key, group in self.Groups.items() if first_color in group), None)
            leaves_order = ddata["leaves"]
            for pos, leaf_idx in enumerate(leaves_order):
                sample = sample_labels[leaf_idx]
                tick_labels = ax.get_xticklabels()
                if pos < len(tick_labels):  # Ensure index is within bounds
                    tick_labels[pos].set_color(self.colors[self.score_df[groupname].loc[sample]])
        
        self.figure.subplots_adjust(top=0.94, bottom=0.2, left=0.05, right=0.99)
        self.canvas.draw()

        self.plot_type = "Dendrogram"

    def plot_kmeans(self, data, k, sample_labels):
        self.update_pc_selection()
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

        self.plot_type = "KMeans"




    def analyze_clustering(self, df):
        self.update_pc_selection()
        """
        Runs all clustering significance checks and prints results.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing PCA scores.
        """
        # Preprocess data
        pc_data, sample_names = preprocess_data(df)

        # number of cluster based on diffrent type of current group
        if self.colors is not None:
            number_of_clusters = len(self.colors)
        else:
            QMessageBox.warning(self, "Warning", "No coloring group selected. Please select a group for coloring first.")
            return

        # Compute statistics
        cophenetic_corr = compute_cophenetic_correlation(pc_data)
        optimal_clusters = gap_statistic(pc_data)
        stability_score = compute_cluster_stability(pc_data, number_of_clusters)
        silhouette = compute_silhouette_score(pc_data, number_of_clusters)
        optimal_clusters_dunn, dunn_index = compute_dunn_index(pc_data)

        self.text_output.append(f"Cluster based on current Groups: {number_of_clusters} "
                                "(Suggests the ideal number of clusters based on expected separation)")
        # Print results to output with descriptions
        self.text_output.append(f"Cophenetic Correlation: {cophenetic_corr:.4f} "
                                "(Measures how well the dendrogram preserves the pairwise distances)")
        self.text_output.append(f"Optimal Clusters (Gap Statistic): {optimal_clusters} "
                                "(Suggests the ideal number of clusters based on expected separation)")
        self.text_output.append(f"Cluster Stability Score: {stability_score:.4f} "
                                "(Indicates the robustness and reproducibility of the clusters)")
        self.text_output.append(f"Silhouette Score: {silhouette:.4f} "
                                "(Evaluates the quality of the clustering; higher values indicate better consistency)")
        self.text_output.append(f"Dunn Index: {dunn_index:.4f} (Optimal Clusters: {optimal_clusters_dunn}) "
                                "(Ratio of minimum inter-cluster distance to maximum intra-cluster distance; higher is better)"
                                " (Optimal number of clusters based on Dunn Index)")


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

    def get_model(self, algorithm):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        model_mapping = {
            "Naive Bayes": (GaussianNB, "Running Naive Bayes Classifier"),
            "K-Nearest Neighbors": (KNeighborsClassifier, "Running K-Nearest Neighbors Classifier"),
            "Linear Discriminant Analysis": (LinearDiscriminantAnalysis, "Running Linear Discriminant Analysis"),
            "Decision Tree": (DecisionTreeClassifier, "Running Decision Tree Classifier"),
            "Random Forest": (RandomForestClassifier, "Running Random Forest Classifier"),
            "Gradient Boosting": (GradientBoostingClassifier, "Running Gradient Boosting Classifier")
        }
        
        model_info = model_mapping.get(algorithm)
        if model_info is None:
            self.text_output.append("Selected algorithm is not supported.")
            return None, None
        model_class, message = model_info
        return model_class(), message
    
    def run_ml(self):
        self.update_pc_selection()
        # Get the selected algorithm and the data with numerical features only
        algorithm = self.algorithm_combobox.currentText()
        data = self.score_df.select_dtypes(exclude=['category'])

        # Check if colors (and thus group selection) have been defined
        if not self.colors:
            self.text_output.append("Coloring group not selected. Please select a group for coloring first.")
            return

        first_color = list(self.colors.keys())[0]
        groupname = next((f"Group {key}" 
                          for key, group in self.Groups.items() 
                          if first_color in group), None)
        target = self.score_df[groupname]

        # Split data if a validation set size has been specified
        if self.size_validation_set.value() > 0:
            from sklearn.model_selection import train_test_split
            data, val_data, target, val_target = train_test_split(
                data, target, test_size=self.size_validation_set.value(), random_state=42)
        else:
            # If no validation set is specified, use the training set for validation (warning: use with care)
            val_data, val_target = data, target

        # Get the model and print the selected message
        model, message = self.get_model(algorithm)
        if model is None:
            return  # Exit if there was a problem with model selection

        self.text_output.append(message)

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
        cm = np.round(cm / cm.sum(axis=1)[:, np.newaxis],2)
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
        # set colorbar limits from 0 to 1
        cax.set_clim(0, 1)
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

        self.plot_type = "CM_" + self.algorithm_combobox.currentText()

    def update_pc_selection(self):
        num_pcs = self.pc_spinbox.value()
        max_pcs = len(self.results['scores'][list(self.results['scores'].keys())[0]])
        if num_pcs > max_pcs:
            num_pcs = max_pcs
        self.score_df = pd.DataFrame.from_dict(self.results['scores'], orient="index")
        self.score_df = self.score_df.iloc[:, :num_pcs]
        self.score_df.columns = [f"PC{i+1}" for i in range(num_pcs)]

        self.Groups, self.GroupList = GroupMaker(self.file_names, keyword='GroupList')

        # Add the group columns to the DataFrame
        for idx, group in self.Groups.items():
            # add each grouplist to the dataframe as a new column as categorical data
            self.score_df[f"Group {idx}"] = pd.Categorical(self.GroupList[idx], categories=group, ordered=True)
    
    def save_current_plot(self):            
        self.output_folder = os.path.join(self.parent.selected_folder, 'output', f'statistical_analysis_{self.parent.run_id}')
        os.makedirs(self.output_folder, exist_ok=True)
        # save plot- add timestamp and what kind of plot it is
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # save the plot with a timestamp and type
        filename = os.path.join(self.output_folder, f"{self.plot_type}_{timestamp}.png")
        self.figure.savefig(filename)

        self.text_output.append(f"Plot saved as {filename}")


    
    def cluster_test(self, data=None, algorithm=None, number_of_clusters=None):
        if algorithm == "K-Means":
            kmeans = KMeans(n_clusters=number_of_clusters)
            labels = kmeans.fit_predict(data)

        elif algorithm == "Hierarchical":
            # Compute the linkage matrix
            dist_matrix = pdist(data, metric='euclidean')
            linkage_matrix = linkage(dist_matrix, method='ward')
            # Form flat clusters
            labels = fcluster(linkage_matrix, t=number_of_clusters, criterion='maxclust')

        elif algorithm == "DBSCAN":
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(data)
            
            
        elif algorithm == "Gaussian Mixture Model":
            gmm = GaussianMixture(n_components=4)
            gmm.fit(data)
            labels = gmm.predict(data)
            
        
        else:
            QMessageBox.warning(self, "Warning", "Selected clustering algorithm is not supported.")

        return labels, algorithm

    def run_cluster_test(self):
        self.update_pc_selection()
        # Get the selected algorithm
        algorithm = self.cluster_combobox.currentText()
        data = self.score_df.select_dtypes(exclude=['category'])

        # Check if colors (and thus group selection) have been defined
        if not self.colors:
            self.text_output.append("Coloring group not selected. Please select a group for coloring first.")
            return

        first_color = list(self.colors.keys())[0]
        groupname = next((f"Group {key}" 
                          for key, group in self.Groups.items() 
                          if first_color in group), None)
        target = self.score_df[groupname]

        if self.colors is not None:
            number_of_clusters = len(self.colors)
        else:
            QMessageBox.warning(self, "Warning", "No coloring group selected. Please select a group for coloring first.")
            return


        # Run the clustering test and get the target labels and algorithm
        labels, algorithm = self.cluster_test(data=data, algorithm=algorithm, number_of_clusters=number_of_clusters)

        # Test Clustering with Adjusted Rand Index, Normalized Mutual Information, Purity Score, Homogeneity, Completeness, and V-Measure
        ari, nmi, purity, homogeneity, completeness, v_measure = self.evaluate_clustering(target, labels, algorithm)
        # Print the evaluation results
        self.print_evaluation_results(ari, nmi, purity, homogeneity, completeness, v_measure, algorithm)

    def purity_score(self, y_true, y_pred):
        matrix = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

    def evaluate_clustering(self, y_true, y_pred, algorithm):
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        purity = self.purity_score(y_true, y_pred)
        homogeneity = homogeneity_score(y_true, y_pred)
        completeness = completeness_score(y_true, y_pred)
        v_measure = v_measure_score(y_true, y_pred)


        return ari, nmi, purity, homogeneity, completeness, v_measure

    def print_evaluation_results(self, ari, nmi, purity, homogeneity, completeness, v_measure, algorithm):
        self.text_output.append(f"\n🔍 Clustering Evaluation with {algorithm}:\n")

        self.text_output.append(f"Adjusted Rand Index (ARI): {ari:.4f}")
        if ari > 0.8:
            self.text_output.append(" → Very strong agreement with true labels.")
        elif ari > 0.5:
            self.text_output.append(" → Moderate agreement.")
        elif ari > 0.2:
            self.text_output.append(" → Weak agreement.")
        else:
            self.text_output.append(" → Little to no agreement.")

        self.text_output.append(f"\nNormalized Mutual Information (NMI): {nmi:.4f}")
        if nmi > 0.8:
            self.text_output.append(" → Clusters preserve most of the label information.")
        elif nmi > 0.5:
            self.text_output.append(" → Clusters capture some label information.")
        else:
            self.text_output.append(" → Clusters carry little information about the labels.")

        self.text_output.append(f"\nPurity Score: {purity:.4f}")
        if purity > 0.9:
            self.text_output.append(" → Nearly all clusters contain only one class.")
        elif purity > 0.7:
            self.text_output.append(" → Good purity, but possibly many small clusters.")
        else:
            self.text_output.append(" → Many clusters mix multiple classes.")

        self.text_output.append(f"\nHomogeneity: {homogeneity:.4f}")
        self.text_output.append(f"Completeness: {completeness:.4f}")
        self.text_output.append(f"V-Measure: {v_measure:.4f}")
        if homogeneity > 0.8 and completeness > 0.8:
            self.text_output.append(" → Clusters are both homogeneous and complete – excellent structure.")
        elif v_measure > 0.5:
            self.text_output.append(" → Moderate balance between homogeneity and completeness.")
        else:
            self.text_output.append(" → Clusters poorly match the true label distribution.")        

    def save_as_csv(self):
        self.output_folder = os.path.join(self.parent.selected_folder, 'output', f'statistical_analysis_{self.parent.run_id}')
        os.makedirs(self.output_folder, exist_ok=True)
        # save plot- add timestamp and what kind of plot it is
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Go through all groups and all clusters algorithms and save the evaluation results in a csv file
        filename = os.path.join(self.output_folder, f"clustering_evaluation_{timestamp}.csv")
        # Create a DataFrame to hold the results
        results_df = pd.DataFrame(columns=[
            "Group", "Algorithm", "Adjusted Rand Index", "Normalized Mutual Information",
            "Purity Score", "Homogeneity", "Completeness", "V-Measure"
        ])
        # Add the results to the DataFrame
        data = self.score_df.select_dtypes(exclude=['category'])

        results_list = []
        for group in list(self.Groups.values()):
            color = assign_colors(group)

            first_color = list(color.keys())[0]
            groupname = next((f"Group {key}" 
                            for key, group in self.Groups.items() 
                            if first_color in group), None)
            target = self.score_df[groupname]

            for algorithm in ["K-Means", "Hierarchical", "DBSCAN", "Gaussian Mixture Model"]:
                # Run the clustering test and get the target labels and algorithm
                labels, algorithm = self.cluster_test(data=data, algorithm=algorithm, number_of_clusters=len(color))

                # Test Clustering with Adjusted Rand Index, Normalized Mutual Information, Purity Score, Homogeneity, Completeness, and V-Measure
                ari, nmi, purity, homogeneity, completeness, v_measure = self.evaluate_clustering(target, labels, algorithm)

                group_name = ' '.join(color.keys())
                
                # Collect the results in a list of dicts
                results_list.append({
                    "Group": group_name,
                    "Algorithm": algorithm,
                    "Adjusted Rand Index": ari,
                    "Normalized Mutual Information": nmi,
                    "Purity Score": purity,
                    "Homogeneity": homogeneity,
                    "Completeness": completeness,
                    "V-Measure": v_measure
                })
        # Convert the list of dicts to a DataFrame and save to CSV
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(filename, index=False)
        self.text_output.append(f"Clustering evaluation results saved as {filename}")

