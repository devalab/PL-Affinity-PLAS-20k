from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np

import argparse
parser = argparse.ArgumentParser(
    description='Train 3D colnvolutional neural network on affinity data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--num_frames', type=int, default=25, help='Number of frames to select')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--total_num_frames', type=int, default=200, help='Total number of frames')
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)

# Define total number of points to select
x = args.num_frames

total_num_frames = args.total_num_frames

with open("list-pdbids.txt", "r") as f:
    pdbids = f.readlines()
pdbids = [x.strip() for x in pdbids]
pdbids.sort()

best_cluster_count = {}

for pdbid in pdbids:

    print(f"Processing {pdbid}...", flush=True)

    # Load your RMSD matrix here
    filename = f"rmsd/rmsd_matrix_{pdbid}.dat"

    # Check if file is empty
    with open(filename, 'r') as f:
        if not f.readline():
            print(f"Empty file for {pdbid}. Skipping...", flush=True)
            with open(f"selected_points_{x}.txt", "a") as f:
                f.write(f"{pdbid} 0\n")
            continue

    matrix = []
    with open(filename, 'r') as f:
        text = f.readlines()
        count = 0
        for line in text:
            temp = []
            temp += [0] * count
            temp += [round(float(x), 9) for x in line.split()]
            matrix.append(temp)
            count += 1

    if (count != total_num_frames):
        print(f"Invalid file for {pdbid}. Skipping...", flush=True)
        with open(f"selected_points_{x}.txt", "a") as f:
            f.write(f"{pdbid} -1\n")
        continue

    rmsd_matrix = np.array(matrix)
    for i in range(len(rmsd_matrix)):
        for j in range(i):
            rmsd_matrix[i][j] = rmsd_matrix[j][i]

    # Define a range of values for n_clusters
    n_clusters_range = range(4, 19)

    # Initialize variables to store the best silhouette score and corresponding number of clusters
    best_silhouette_score = -1
    best_n_clusters = None

    # Iterate over the range of values for n_clusters
    for n_clusters in n_clusters_range:
        # Initialize KMedoids with custom distance metric and K-Medoids++ initialization
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', init='k-medoids++', random_state=seed)
        
        # Fit KMedoids using dissimilarity matrix
        kmedoids.fit(rmsd_matrix)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(rmsd_matrix, kmedoids.labels_, metric='precomputed')
        print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
        
        # Update best silhouette score and corresponding number of clusters if needed
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters

    # Print the best number of clusters
    print()
    print(f"The best number of clusters is: {best_n_clusters} with silhouette score: {best_silhouette_score}")

    best_cluster_count[best_n_clusters] = best_cluster_count.get(best_n_clusters, 0) + 1

    # Initialize KMedoids with custom distance metric and K-Medoids++ initialization
    kmedoids = KMedoids(n_clusters=best_n_clusters, metric='precomputed', init='k-medoids++', random_state=seed)

    # Fit KMedoids using dissimilarity matrix
    kmedoids.fit(rmsd_matrix)

    # Get cluster labels
    labels = kmedoids.labels_
    print("\nCluster labels:")
    print(labels[:20], labels[20:40], labels[40:60], labels[60:80], labels[80:100], labels[100:120], labels[120:140], labels[140:160], labels[160:180], labels[180:200], sep='\n')
    print()

    # Get cluster medoids (indices)
    medoid_indices = kmedoids.medoid_indices_
    print("Cluster medoid indices:", medoid_indices)
    print()

    # Initialize list to temporarily store selected points and their frequencies
    selected_points_freq = []

    total_sum = 0 

    # Iterate over each cluster
    for cluster_index in range(best_n_clusters):
        # Get indices of points in the current cluster
        cluster_points_indices = np.where(labels == cluster_index)[0]
        
        # Calculate number of points to select from this cluster based on its proportion to total points and cluster size
        num_points_cluster = len(cluster_points_indices)
        num_points_to_select = int(x * (num_points_cluster / len(rmsd_matrix))) + 1
        print(f"Cluster {cluster_index} has {num_points_cluster} points and will select {num_points_to_select} points")

        # Select points randomly from the cluster
        selected_points_indices = np.random.choice(cluster_points_indices, num_points_to_select, replace=False) + 1

        print(f"Selected points from cluster {cluster_index}:", selected_points_indices)

        # Add selected points to the list of selected points
        selected_points_freq.append([len(selected_points_indices), [point for point in selected_points_indices]])

        total_sum += len(selected_points_indices)

    # Sort selected points based on their frequency
    selected_points_freq.sort(reverse=True)

    # Initialize list to store selected points
    selected_points = []

    for i in range(len(selected_points_freq)):
        if (total_sum > x):
            selected_points.extend(selected_points_freq[i][1][:-1])
            total_sum -= 1
        else:
            selected_points.extend(selected_points_freq[i][1])

    # Take a random permutation of selected points to shuffle their order
    np.random.shuffle(selected_points)

    print()
    print("Final selected points:", selected_points)
    print("Number of selected points:", len(selected_points), flush=True)

    with open(f"selected_points_{x}.txt", "a") as f:
        f.write(f"{pdbid} {len(selected_points)} ")
        for point in selected_points:
            f.write(f"{point} ")
        f.write("\n")
    print()

print("Best cluster counts:")
print(best_cluster_count)
print()