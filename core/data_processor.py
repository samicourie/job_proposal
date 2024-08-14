import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic


class DataProcessor(object):
    def __init__(self, origin_file, accepted_file, refused_file, verbose=True):
        self.origin = pd.read_csv(origin_file)
        self.accepted = pd.read_csv(accepted_file)
        self.refused = pd.read_csv(refused_file)
        self.origin_coord = (self.origin.iloc[0]['latitude'], self.origin.iloc[0]['longitude'])
        self.data = None
        self.verbose = verbose

    def calculate_distance(self, row):
        return geodesic(self.origin_coord, (row['latitude'], row['longitude'])).kilometers

    def describe_data(self):
        """
        Describe the dataset for a better insight, whether there are duplicates or NaN rows and how many
        """
        if self.verbose:
            print('- Data description with pandas ...')
        print(self.data.describe())
        print()
        # Checking for missing values
        print('- Rows that contain NaN ...')
        print(self.data.isnull().sum())
        print()
        # Checking for duplicates
        print('- Duplicated:', self.data.duplicated().sum())
        print()

    def preprocess_data(self):
        """
        Calculating the distance in KM relative to the original point of interest.
        """

        if self.verbose:
            print('- Calculating the distance in KM ...')
        self.accepted['distance'] = self.accepted.apply(self.calculate_distance, axis=1)
        self.refused['distance'] = self.refused.apply(self.calculate_distance, axis=1)

        self.accepted['accepted'] = 1
        self.refused['accepted'] = 0

        if self.verbose:
            print('- Merging accepted and refused dataset ...')
        self.data = pd.concat([self.accepted, self.refused], ignore_index=True)

        if self.verbose:
            print('- Calculating the price in euros instead of cents ...')
        self.data['prix'] = self.data['prix'] / 100

    def feature_engineering(self):
        """
        Adding one more feature Euro/KM to the dataset.
        """

        if self.verbose:
            print('- Adding euro_per_km feature to the dataset ...')
        self.data['euro_per_km'] = self.data['prix'] / self.data['distance']

    def visualize_data(self):
        """
        Visualising the data with histogram and scatter plot relative to the original point.
        """

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.data[self.data['accepted'] == 1]['distance'], color='green', kde=True, label='Accepted')
        sns.histplot(self.data[self.data['accepted'] == 0]['distance'], color='red', kde=True, label='Refused')
        plt.xlabel('Distance (km)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution of Distances')

        plt.subplot(1, 2, 2)
        sns.histplot(self.data[self.data['accepted'] == 1]['prix'], color='green', kde=True, label='Accepted')
        sns.histplot(self.data[self.data['accepted'] == 0]['prix'], color='red', kde=True, label='Refused')
        plt.xlabel('Price (euros)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution of Prices')

        plt.tight_layout()
        plt.show()

        # Create a 2D scatter plot
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot accepted proposals in green
        accepted_proposals = self.data[self.data['accepted'] == 1]
        ax.scatter(accepted_proposals['distance'], accepted_proposals['prix'], color='green', label='Accepted')

        # Plot refused proposals in red
        refused_proposals = self.data[self.data['accepted'] == 0]
        ax.scatter(refused_proposals['distance'], refused_proposals['prix'], color='red', label='Refused')

        # Plot lines connecting each point to the origin
        for _, row in self.data.iterrows():
            if row['accepted'] == 1:
                ax.plot([0, row['distance']], [0, row['prix']], color='green', alpha=0.5)
            else:
                ax.plot([0, row['distance']], [0, row['prix']], color='red', alpha=0.5)

        # Set labels
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Price (euros)')

        # Set title
        ax.set_title('2D Scatter Plot of Distance and Price with Acceptance Status')

        # Add legend
        ax.legend()

        # Show plot
        plt.show()
