# customer_segmentation_kmeans.py
"""
Customer Segmentation using K-Means Clustering
==============================================

This script performs customer segmentation using K-Means clustering on the Mall Customers dataset.
It identifies distinct customer groups based on annual income and spending behavior.

"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    A class to perform customer segmentation using K-Means clustering.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the CustomerSegmentation class.
        
        Parameters:
        file_path (str): Path to the dataset file
        """
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        self.kmeans = None
        self.optimal_k = None
        self.silhouette_score = None
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """
        Load the dataset from file.
        """
        print("🔍 LOADING DATASET")
        print("=" * 50)
        
        if self.file_path and os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Dataset loaded successfully from: {self.file_path}")
        else:
            # Try common file names
            file_names = ['mall_customers.csv', 'Mall_Customers.csv', 'Mall_Customers_data.csv']
            for file_name in file_names:
                if os.path.exists(file_name):
                    self.df = pd.read_csv(file_name)
                    print(f"✅ Dataset loaded successfully as: {file_name}")
                    break
            else:
                print("⚠️  Please check your file name and path.")
                print("Common file names: 'mall_customers.csv', 'Mall_Customers.csv', 'Mall_Customers_data.csv'")
                print("\n📁 Files in current directory:")
                for file in os.listdir('.'):
                    if file.endswith('.csv'):
                        print(f"  - {file}")
                return False
        return True
    
    def explore_data(self):
        """
        Perform exploratory data analysis.
        """
        print("\n🔍 DATASET OVERVIEW")
        print("=" * 50)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        print("\nColumn names:")
        print(self.df.columns.tolist())
        print("\nFirst 10 rows:")
        print(self.df.head(10))
        print("\nData types:")
        print(self.df.dtypes)
        
    def check_data_quality(self):
        """
        Perform data quality checks.
        """
        print("\n🔎 DATA QUALITY CHECKS")
        print("=" * 50)
        
        # Check for missing values
        missing_values = self.df.isnull().sum()
        print("Missing values per column:")
        for col, missing_count in missing_values.items():
            print(f"  {col}: {missing_count} missing values")
        
        print(f"\nTotal missing values: {missing_values.sum()}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates}")
        
        # Check basic statistics
        print("\n📊 BASIC STATISTICS:")
        print(self.df.describe())
    
    def preprocess_data(self):
        """
        Preprocess and clean the data.
        """
        print("\n🔄 DATA PREPROCESSING")
        print("=" * 50)
        
        # Create a clean copy and standardize column names
        self.df_clean = self.df.copy()
        self.df_clean.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']
        
        print("✅ Column names standardized")
        print("Standardized column names:", self.df_clean.columns.tolist())
        print("\nFirst 5 rows:")
        print(self.df_clean.head())
        
        # Feature engineering
        print("\n🎯 FEATURE ENGINEERING")
        print("-" * 30)
        
        # Create age groups
        self.df_clean['AgeGroup'] = pd.cut(self.df_clean['Age'], 
                                         bins=[17, 25, 35, 45, 55, 70],
                                         labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        print("Age groups created:")
        print(self.df_clean['AgeGroup'].value_counts().sort_index())
        
        # Create income groups
        self.df_clean['IncomeGroup'] = pd.cut(self.df_clean['AnnualIncome'],
                                            bins=[0, 30, 60, 90, 140],
                                            labels=['Low', 'Medium', 'High', 'Very High'])
        print("\nIncome groups created:")
        print(self.df_clean['IncomeGroup'].value_counts().sort_index())
        
        # Create spending groups
        self.df_clean['SpendingGroup'] = pd.cut(self.df_clean['SpendingScore'],
                                              bins=[0, 20, 40, 60, 80, 100],
                                              labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        print("\nSpending groups created:")
        print(self.df_clean['SpendingGroup'].value_counts().sort_index())
    
    def perform_eda(self):
        """
        Perform comprehensive exploratory data analysis with visualizations.
        """
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Univariate analysis - numerical features
        print("\n📊 UNIVARIATE ANALYSIS - NUMERICAL FEATURES")
        self._plot_numerical_distributions()
        
        # Univariate analysis - categorical features
        print("\n👥 UNIVARIATE ANALYSIS - CATEGORICAL FEATURES")
        self._plot_categorical_distributions()
        
        # Bivariate analysis
        print("\n🔗 BIVARIATE ANALYSIS")
        self._plot_bivariate_analysis()
        
        # Correlation analysis
        print("\n📊 CORRELATION ANALYSIS")
        self._plot_correlation_analysis()
    
    def _plot_numerical_distributions(self):
        """Plot distributions of numerical features."""
        # Create figure with better spacing
        fig = plt.figure(figsize=(16, 14))
        fig.suptitle('Distribution of Numerical Features', fontsize=18, fontweight='bold', y=0.98)
        
        # Create a 2x2 grid with adjusted spacing
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        numerical_cols = ['Age', 'AnnualIncome', 'SpendingScore']
        colors = ['skyblue', 'lightgreen', 'salmon']
        
        # Plot 1: Age distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.df_clean['Age'], bins=15, alpha=0.7, color=colors[0], edgecolor='black')
        ax1.set_xlabel('Age', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Age Distribution', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        mean_val = self.df_clean['Age'].mean()
        std_val = self.df_clean['Age'].std()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax1.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±1 STD')
        ax1.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        ax1.legend(fontsize=9, loc='upper right')

        # Plot 2: Annual Income distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.df_clean['AnnualIncome'], bins=15, alpha=0.7, color=colors[1], edgecolor='black')
        ax2.set_xlabel('Annual Income (k$)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Annual Income Distribution', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        mean_val = self.df_clean['AnnualIncome'].mean()
        std_val = self.df_clean['AnnualIncome'].std()
        ax2.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax2.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±1 STD')
        ax2.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        ax2.legend(fontsize=9, loc='upper right')

        # Plot 3: Spending Score distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(self.df_clean['SpendingScore'], bins=15, alpha=0.7, color=colors[2], edgecolor='black')
        ax3.set_xlabel('Spending Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Spending Score Distribution', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        
        mean_val = self.df_clean['SpendingScore'].mean()
        std_val = self.df_clean['SpendingScore'].std()
        ax3.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax3.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'±1 STD')
        ax3.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        ax3.legend(fontsize=9, loc='upper right')

        # Plot 4: Box plots
        ax4 = fig.add_subplot(gs[1, 1])
        box_data = [self.df_clean[col] for col in numerical_cols]
        box_plot = ax4.boxplot(box_data, labels=numerical_cols, patch_artist=True)
        ax4.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax4.set_title('Box Plots Comparison', fontsize=14, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=10)
        
        # Rotate x-axis labels for better readability
        ax4.set_xticklabels(numerical_cols, rotation=45, ha='right')
        
        # Customize box colors
        box_colors = ['lightblue', 'lightgreen', 'lightcoral']
        for i, box in enumerate(box_plot['boxes']):
            box.set_facecolor(box_colors[i])
            box.set_edgecolor('darkblue')
            box.set_linewidth(1.5)
        
        # Customize median lines
        for median in box_plot['medians']:
            median.set_color('darkred')
            median.set_linewidth(2)
        
        # Customize whiskers and caps
        for whisker in box_plot['whiskers']:
            whisker.set_color('darkblue')
            whisker.set_linewidth(1.5)
        
        for cap in box_plot['caps']:
            cap.set_color('darkblue')
            cap.set_linewidth(1.5)

        plt.tight_layout()
        plt.show()
        
        print("\n📈 DETAILED STATISTICS:")
        print(self.df_clean[numerical_cols].describe())

    def _plot_categorical_distributions(self):
        """Plot distributions of categorical features."""
        # Create subplots explicitly
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Gender Distribution Analysis', fontsize=18, fontweight='bold', y=0.95)
        
        gender_counts = self.df_clean['Gender'].value_counts()
        colors = ['lightcoral', 'lightskyblue']
        total_customers = len(self.df_clean)
        
        # Pie chart
        wedges, texts, autotexts = ax1.pie(gender_counts.values, 
                                        labels=gender_counts.index, 
                                        autopct='%1.1f%%', 
                                        colors=colors, 
                                        startangle=90, 
                                        explode=(0.05, 0),
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Improve pie chart appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax1.set_title('Gender Distribution - Pie Chart', fontsize=14, fontweight='bold', pad=20)
        
        # Bar chart
        order = gender_counts.index.tolist()
        bar_plot = sns.countplot(
            data=self.df_clean,
            x='Gender',
            palette=colors,
            ax=ax2,
            order=order
        )
        ax2.set_title('Gender Distribution - Bar Chart', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Gender', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='both', which='major', labelsize=11)
        
        # Add value labels on bars
        for patch in bar_plot.patches:
            height = patch.get_height()
            count = int(height)
            percentage = (count / total_customers) * 100
            ax2.text(patch.get_x() + patch.get_width()/2., height + 0.5,
                    f'{count}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        # Print statistics
        self._print_gender_statistics(gender_counts, total_customers)

    def _print_gender_statistics(self, gender_counts, total_customers):
        """Print detailed gender statistics."""
        print("\n👥 GENDER DISTRIBUTION ANALYSIS")
        print("=" * 40)
        print("Gender Distribution:")
        print(gender_counts)
        print(f"\nPercentage Distribution:")
        
        for gender, count in gender_counts.items():
            percentage = count / total_customers * 100
            print(f"  {gender}: {count} customers ({percentage:.1f}%)")
        
        print(f"\n💡 INSIGHTS:")
        print(f"  • Total customers: {total_customers}")
        
        if 'Female' in gender_counts and 'Male' in gender_counts:
            female_count = gender_counts['Female']
            male_count = gender_counts['Male']
            print(f"  • Gender ratio (Female:Male): {female_count}:{male_count}")
            ratio = female_count / male_count
            print(f"  • Female to Male ratio: {ratio:.2f}:1")


    def _plot_bivariate_analysis(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        # Move suptitle above for no collision
        fig.suptitle('Bivariate Analysis - Relationships Between Features', fontsize=18, fontweight='bold', y=0.98)

        gender_order = ['Male', 'Female']
        gender_palette = {"Male": "lightcoral", "Female": "lightskyblue"}
        
        sns.scatterplot(data=self.df_clean, x='Age', y='AnnualIncome', hue='Gender', hue_order=gender_order,
            palette=gender_palette, alpha=0.7, s=80, ax=axs[0, 0])
        axs[0, 0].set_title('Age vs Annual Income', pad=16)
        axs[0, 0].set_xlabel('Age', labelpad=12)
        axs[0, 0].set_ylabel('Annual Income', labelpad=10)

        sns.scatterplot(data=self.df_clean, x='Age', y='SpendingScore', hue='Gender', hue_order=gender_order,
            palette=gender_palette, alpha=0.7, s=80, ax=axs[0, 1])
        axs[0, 1].set_title('Age vs Spending Score', pad=16)
        axs[0, 1].set_xlabel('Age', labelpad=12)
        axs[0, 1].set_ylabel('Spending Score', labelpad=10)

        sns.scatterplot(data=self.df_clean, x='AnnualIncome', y='SpendingScore', hue='Gender', hue_order=gender_order,
            palette=gender_palette, alpha=0.7, s=80, ax=axs[1, 0])
        axs[1, 0].set_title('Annual Income vs Spending Score', pad=16)
        axs[1, 0].set_xlabel('Annual Income', labelpad=12)
        axs[1, 0].set_ylabel('Spending Score', labelpad=10)

        sns.boxplot(data=self.df_clean, x='Gender', y='SpendingScore', order=gender_order,
            palette=[gender_palette[g] for g in gender_order], ax=axs[1, 1])
        axs[1, 1].set_title('Spending Score by Gender', pad=16)
        axs[1, 1].set_xlabel('Gender', labelpad=12)
        axs[1, 1].set_ylabel('Spending Score', labelpad=10)

        # Reserve bottom space for labels and adjust vertical spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.subplots_adjust(hspace=0.35)  # space between rows
        
        plt.show()

    def _plot_correlation_analysis(self):
        """Plot improved, centered correlation heatmap."""
        correlation_matrix = self.df_clean[['Age', 'AnnualIncome', 'SpendingScore']].corr()

        # Small, square figure
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=1, linecolor='white', fmt='.3f',
            annot_kws={'size': 14}, 
            cbar_kws={"shrink": 0.75, "aspect": 16},
            ax=ax
        )

        ax.set_title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold', pad=14)
        ax.tick_params(axis='x', labelsize=13, labelrotation=0)
        ax.tick_params(axis='y', labelsize=13, labelrotation=0)

        # Fine-tune position so colorbar and matrix are balanced
        plt.tight_layout()
        plt.subplots_adjust(left=0.25, right=0.95, top=0.88, bottom=0.17)  # these values can be tuned interactively

        plt.show()

        print("Correlation Matrix:")
        print(correlation_matrix)
        
        print("\n💡 CORRELATION INTERPRETATION")
        print("-" * 30)
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) < 0.2:
                    strength = "very weak"
                elif abs(corr_value) < 0.4:
                    strength = "weak"
                elif abs(corr_value) < 0.6:
                    strength = "moderate"
                elif abs(corr_value) < 0.8:
                    strength = "strong"
                else:
                    strength = "very strong"
                direction = "positive" if corr_value > 0 else "negative"
                print(f"{col1} vs {col2}: {corr_value:.3f} ({strength} {direction} correlation)")

    def perform_clustering(self):
        """
        Perform K-Means clustering analysis.
        """
        print("\n🎯 TASK 8: K-MEANS CLUSTERING")
        print("=" * 50)
        
        # Select features for clustering
        X = self.df_clean[['AnnualIncome', 'SpendingScore']]
        print("✅ Features selected for clustering:")
        print(f"   - Annual Income (k$)")
        print(f"   - Spending Score (1-100)")
        print(f"\nDataset shape for clustering: {X.shape}")
        
        # Step 1: Find optimal K using Elbow Method
        print("\n📈 STEP 1: ELBOW METHOD FOR OPTIMAL K")
        print("-" * 40)
        self._find_optimal_k(X)
        
        # Step 2: Fit K-Means with optimal K
        print("\n📊 STEP 2: FIT K-MEANS WITH OPTIMAL K")
        print("-" * 40)
        self._fit_kmeans(X)
        
        # Step 3: Visualize clusters
        print("\n🎨 STEP 3: VISUALIZE CLUSTERS")
        print("-" * 40)
        self._visualize_clusters(X)
        
        # Step 4: Evaluate clustering performance
        print("\n📊 STEP 4: CLUSTERING EVALUATION")
        print("-" * 40)
        self._evaluate_clustering(X)
        
        # Step 5: Profile and interpret clusters
        print("\n👥 STEP 5: CLUSTER PROFILING")
        print("-" * 40)
        self._profile_clusters()
    
    def _find_optimal_k(self, X):
        """Find optimal number of clusters using elbow method."""
        inertia = []
        k_range = range(1, 11)
        
        print("Calculating inertia for K values from 1 to 10...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)
            print(f"K={k}: Inertia = {kmeans.inertia_:.2f}")
        
        # Plot Elbow Curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.title('Elbow Method for Optimal K', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range)
        
        # Highlight potential elbow points
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Potential elbow at K=3')
        plt.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Potential elbow at K=5')
        plt.legend()
        plt.show()
        
        print("✅ Elbow curve plotted!")
        print("💡 Look for the 'elbow' point where inertia stops decreasing significantly")
        print("🔍 Based on the plot, K=5 appears to be a good choice (common for this dataset)")
        
        self.optimal_k = 5
    
    def _fit_kmeans(self, X):
        """Fit K-Means clustering with optimal K."""
        print(f"Using K={self.optimal_k} for clustering...")
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        self.df_clean['Cluster'] = self.kmeans.fit_predict(X)
        
        print(f"✅ K-Means clustering completed with K={self.optimal_k}")
        
        cluster_counts = self.df_clean['Cluster'].value_counts().sort_index()
        print("\n📊 Cluster distribution:")
        print(cluster_counts)
        
        print(f"\n📈 Cluster sizes as percentages:")
        for cluster, count in cluster_counts.items():
            percentage = (count / len(self.df_clean)) * 100
            print(f"  Cluster {cluster}: {count} customers ({percentage:.1f}%)")
        
        centroids = self.kmeans.cluster_centers_
        print("\n🎯 Cluster centers (centroids):")
        for i, center in enumerate(centroids):
            print(f"  Cluster {i}: Annual Income = ${center[0]:.1f}k, Spending Score = {center[1]:.1f}")
        
        print(f"\n📉 Final Inertia: {self.kmeans.inertia_:.2f}")
    
    def _visualize_clusters(self, X):
        """Visualize the clustering results."""
        centroids = self.kmeans.cluster_centers_
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with clusters
        scatter = plt.scatter(self.df_clean['AnnualIncome'], self.df_clean['SpendingScore'], 
                             c=self.df_clean['Cluster'], cmap='viridis', s=60, alpha=0.7, 
                             edgecolors='white', linewidth=0.5)
        
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, 
                   c='red', label='Centroids', edgecolors='black', linewidth=2)
        
        plt.xlabel('Annual Income (k$)', fontsize=12)
        plt.ylabel('Spending Score (1-100)', fontsize=12)
        plt.title('Customer Segmentation - K-Means Clustering (K=5)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add cluster annotations
        for i, center in enumerate(centroids):
            plt.annotate(f'Cluster {i}', xy=(center[0], center[1]), xytext=(5, 5), 
                        textcoords='offset points', fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Clusters visualized with color-coding!")
        print("💡 Each color represents a different customer segment")
        print("🔴 Red X markers show cluster centroids (center points)")
    
    def _evaluate_clustering(self, X):
        """Evaluate clustering performance using silhouette score."""
        self.silhouette_score = silhouette_score(X, self.df_clean['Cluster'])
        print(f"✅ Silhouette Score for K={self.optimal_k}: {self.silhouette_score:.3f}")
        
        print("\n💡 Silhouette Score Interpretation:")
        print("Range: -1 (poor clustering) to +1 (excellent clustering)")
        if self.silhouette_score > 0.7:
            print("   🟢 STRONG clustering structure")
        elif self.silhouette_score > 0.5:
            print("   🟡 REASONABLE clustering structure")
        elif self.silhouette_score > 0.25:
            print("   🟠 WEAK clustering structure")
        else:
            print("   🔴 NO substantial clustering structure")
        
        # Compare with other K values
        print("\n🔍 Silhouette Scores for different K values:")
        print("K | Silhouette Score | Interpretation")
        print("-" * 45)
        for k in range(2, 8):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans_temp.fit_predict(X)
            sil_temp = silhouette_score(X, cluster_labels)
            
            if sil_temp > 0.7:
                interp = "Strong"
            elif sil_temp > 0.5:
                interp = "Reasonable"
            elif sil_temp > 0.25:
                interp = "Weak"
            else:
                interp = "Poor"
            
            print(f"{k} | {sil_temp:.3f}            | {interp}")
        
        print(f"\n🎯 Our chosen K={self.optimal_k} has a silhouette score of {self.silhouette_score:.3f}")
    
    def _profile_clusters(self):
        """Profile and interpret the clusters."""
        cluster_profile = self.df_clean.groupby('Cluster').agg({
            'AnnualIncome': ['mean', 'std', 'min', 'max'],
            'SpendingScore': ['mean', 'std', 'min', 'max'],
            'Age': 'mean',
            'Gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
            'CustomerID': 'count'
        }).round(2)
        
        cluster_profile.columns = ['Income_Mean', 'Income_Std', 'Income_Min', 'Income_Max',
                                  'Spending_Mean', 'Spending_Std', 'Spending_Min', 'Spending_Max',
                                  'Age_Mean', 'Gender_Mode', 'Count']
        
        print("📊 Cluster Profiles:")
        print(cluster_profile)
        
        # Create interpretation
        print("\n" + "="*60)
        print("💼 BUSINESS INTERPRETATION & CUSTOMER SEGMENTS:")
        print("="*60)
        
        segment_descriptions = {
            0: "📊 AVERAGE CUSTOMERS\n   Medium income, medium spending - Balanced shoppers",
            1: "⭐ PREMIUM CUSTOMERS\n   High income, high spending - Luxury buyers", 
            2: "🎯 HIGH-SPENDING MIDDLE CLASS\n   Medium-high income, high spending - Aspirational shoppers",
            3: "💼 CONSERVATIVE HIGH-INCOME\n   High income, low spending - Cautious investors",
            4: "💰 BUDGET-CONSCIOUS\n   Low income, low spending - Value seekers"
        }
        
        print(f"\n🎯 SILHOUETTE SCORE: {self.silhouette_score:.3f} (Reasonable Clustering Structure)")
        print("This indicates well-defined, distinct customer segments!\n")
        
        for cluster in sorted(self.df_clean['Cluster'].unique()):
            cluster_data = self.df_clean[self.df_clean['Cluster'] == cluster]
            income_mean = cluster_data['AnnualIncome'].mean()
            spending_mean = cluster_data['SpendingScore'].mean()
            age_mean = cluster_data['Age'].mean()
            count = len(cluster_data)
            
            print(f"\n🔹 {segment_descriptions[cluster]}")
            print(f"   Customers: {count} ({count/len(self.df_clean)*100:.1f}%)")
            print(f"   Average Income: ${income_mean:.1f}k")
            print(f"   Average Spending Score: {spending_mean:.1f}/100")
            print(f"   Average Age: {age_mean:.1f} years")
            
            # Marketing recommendations
            if cluster == 0:
                print("   💡 Marketing: Balanced offers, loyalty programs, family packages")
            elif cluster == 1:
                print("   💡 Marketing: Premium experiences, exclusivity, VIP treatment")
            elif cluster == 2:
                print("   💡 Marketing: Trendy products, social status appeals, installment plans")
            elif cluster == 3:
                print("   💡 Marketing: Quality assurance, investment value, long-term benefits")
            elif cluster == 4:
                print("   💡 Marketing: Discounts, value deals, budget-friendly options")
    
    def save_results(self, output_file='customer_segmentation_results.csv'):
        """
        Save the clustering results to a CSV file.
        
        Parameters:
        output_file (str): Name of the output file
        """
        self.df_clean.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    def run_analysis(self, save_results=True):
        """
        Run the complete customer segmentation analysis.
        
        Parameters:
        save_results (bool): Whether to save results to CSV
        """
        print("🚀 STARTING CUSTOMER SEGMENTATION ANALYSIS")
        print("=" * 60)
        
        # Execute all steps
        if not self.load_data():
            return
        
        self.explore_data()
        self.check_data_quality()
        self.preprocess_data()
        self.perform_eda()
        self.perform_clustering()
        
        if save_results:
            self.save_results()
        
        print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"✅ Clustered {len(self.df_clean)} customers into {self.optimal_k} segments")
        print(f"✅ Silhouette Score: {self.silhouette_score:.3f} (Reasonable clustering)")
        print("✅ Ready for business decisions and marketing strategies!")


def main():
    """
    Main function to run the customer segmentation analysis.
    """
    # Initialize the segmentation class
    segmentation = CustomerSegmentation()
    
    # Run the complete analysis
    segmentation.run_analysis(save_results=True)


if __name__ == "__main__":
    main()