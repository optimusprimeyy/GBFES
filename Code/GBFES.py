# Granular-ball fuzzy entropy-based anomaly detection (GBFES) algorithm
# Please refer to the following papers:
# Granular-Ball Fuzzy Entropy Based Multi-Granularity Fusion Model For Anomaly Detection.
# Uploaded by Sihan Wang on April. 12, 2025. E-mail:wangsihan0713@foxmail.com.
import numpy as np
from pyod.models.lof import LOF
from sklearn.preprocessing import MinMaxScaler
import GettingGranularBalls as GettingGranularBalls
import warnings
warnings.filterwarnings("ignore")


class GB:
    def __init__(self, data, index):  # Data is labeled data, the penultimate column is label, and the last column is index

        self.data = data[:, :-1]  # Delete the indexed column
        self.index = index # Label each GB with an index number
        self.xuhao = list(data[:, -1])  # Save the serial number of each sample in the GB
        self.center = self.data.mean(0) # Get the center of the GB.
        self.radius = self.calculate_radius()   # Get the radius of the GB.

    def calculate_radius(self):
        distances = np.mean(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)
        return distances


# Wrap GB into a class and add it to the set gb_dist.
def Wrap_class(gb_list):
    # input:
    # gb_list is the list of GBs, where each element represents the array of all the samples in the GB.
    gb_dist = []
    for i in range(0, len(gb_list)):
        gb = GB(gb_list[i], i)
        gb_dist.append(gb)
    return gb_dist


# Calculate the distance matrix for the number of overlaps.
def get_Dist_GB(gb_dist, gb_len):
    # input:
    # gb_dist is the set of GBs, where each element represents the class of the GB.
    # gb_len is the number of GBs.
    distance_matrix = np.zeros((gb_len, gb_len))
    for i in range(gb_len):
        for j in range(i, gb_len):
            distance_matrix[i, j] = np.linalg.norm(gb_dist[i].center - gb_dist[j].center) - gb_dist[i].radius - gb_dist[
                j].radius
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


# Calculate the distance between GBs for fuzzy similarity.
def get_Dist(center1, center2, radius1, radius2):
    # input:
    # center1 and radius1 denote the center and radius of the first GB.
    # center2 and radius2 denote the center and radius of the second GB.
    dis_GB = np.linalg.norm(center1 - center2) + radius1 + radius2
    return dis_GB

# Compute fuzzy similarity between GBs using Gaussian kernel function.
def gaussian_kernel(Dis, sigma):
    # input:
    # Dis is the distance matrix based on GBs.
    # sigma is the Gaussian kernel parameter.
    Dis = Dis ** 2
    return np.exp(-Dis / (sigma))


def merge_GBs(distance_matrix, gb_dist):
    # Input:
    # distance_matrix is data matrix between GBs.
    # gb_dist is the set of all GB classes.

    merge_pairs = []  # Record merge pairs, eg:(i,j), indicating that i and j are to be merged
    merge_times = np.zeros(len(distance_matrix))  # Number of overlaps

    # 1.1 Calculate the number of overlaps
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix[i])):
            if distance_matrix[i][j] < 0:
                merge_times[i] = merge_times[i] + 1
                merge_times[j] = merge_times[j] + 1

    # 1.2 Record the GB pairs that will be merged.
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix[i])):
            if distance_matrix[i][j] < (
                    min(gb_dist[i].radius, gb_dist[j].radius) / (min(merge_times[i], merge_times[j]) + 1)):
                merge_pairs.append((i, j))

    # 2.1 Record the GB indexes which each GB was last merged into.
    index_list = {}  # The dict to record the GB indexes
    merged_indices = []  # The list to record GB indexes that have been merged.
    for i, j in merge_pairs:
        if i not in merged_indices:
            index_list[i] = i
            merged_indices.append(i)  # Record merged gb_i
        if j not in merged_indices:
            index_list[j] = index_list[i]
            merged_indices.append(j)  # Record merged gb_j

    # 2.2 Add the last GBs for which no merger occurred.
    for i in range(len(distance_matrix)):
        if i not in merged_indices:
            index_list[i] = i
    return index_list


# Calculate GBLOF
def LOF_GBG(gb_now):
    # Input:
    # gb_now is a list of all GBs contained in the current MGB。
    data_matrix = gb_now[0].data
    data_xuhao_matrix = gb_now[0].xuhao  # Serial number of each sample
    gb_index = [gb_now[0].index]  # Index of each GB

    # 1. Splice all samples in GBs.
    for i in range(1, len(gb_now)):
        data_matrix = np.vstack((data_matrix, gb_now[i].data))
        data_xuhao_matrix.extend(gb_now[i].xuhao)
        gb_index.append(gb_now[i].index)

    # 2. Set the k-value of LOF.
    k = min(8, len(data_matrix))
    if k == 1:
        return [-1 for i in range(len(data_xuhao_matrix))], data_xuhao_matrix

    # 3. Call pyod.LOF
    clf = LOF(n_neighbors=k)
    clf.fit(data_matrix)
    LOF_score = clf.decision_scores_
    LOF_score = LOF_score / max(LOF_score)
    return LOF_score, data_xuhao_matrix


def GBFES(data, sigma, lammda):
    # Input:
    # data is data matrix without decisions, where rows for samples and columns for attributes.
    # epsilon is a given parameter for calculating the fuzzy similarity relation.
    # lammda is a trade-off factor between the LOF of the sample and the fuzzy relative entropy of the GB to which the sample belongs.

    # 0.min-max normalization
    n = data.shape[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    # 1.1 GB generation
    gb_list = GettingGranularBalls.GettingGranularBalls(data)
    gb_len = len(gb_list)
    gb_dist = Wrap_class(gb_list)

    # 2.1 Distance between GBs
    Dis = np.zeros((gb_len, gb_len))
    for x in range(gb_len):
        for y in range(x + 1, gb_len):
            Dis[x, y] = get_Dist(gb_dist[x].center, gb_dist[y].center, gb_dist[x].radius, gb_dist[y].radius)
            Dis[y, x] = Dis[x, y]

    # 2.2 GB-based fuzzy similarity matrix
    R = gaussian_kernel(Dis, sigma)

    # 3.1 GB-based fuzzy entropy
    H = -np.mean(np.log2(np.sum(R, axis=1) / gb_len))

    # 3.2 GB-based relative fuzzy entropy
    H_as_x = np.zeros(gb_len)
    for j in range(gb_len):
        R_x = R
        R_x = np.delete(R_x, j, axis=0)
        R_x = np.delete(R_x, j, axis=1)
        H_x = -np.mean(np.log2(np.sum(R_x, axis=1) / (gb_len - 1)))
        if H_x > H:
            H_as_x[j] = 0
            continue
        H_as_x[j] = 1 - H_x / H

    # 4.1 Find the overlapping GB
    gb_Dis = get_Dist_GB(gb_dist, gb_len)  # ||ci-cj||-ri-rj

    # 4.2 Merge GB
    index_list = merge_GBs(gb_Dis, gb_dist)
    new_gb_item_list = list(set(index_list.values()))  # Get the last merged GB index

    # 4.3 calculate LOF(xi) in GBG
    LOF_list = []
    data_xuhao_list = []
    for new_gb_item in new_gb_item_list:
        # Find the GBs in this last merged GB.
        indexes_with_value = {i for i, value in index_list.items() if value == new_gb_item}
        gb_now = [gb_dist[i] for i in indexes_with_value]

        # Calculate LOF
        LOF, data_xuhao = LOF_GBG(gb_now)
        LOF_list.extend(LOF)
        data_xuhao_list.extend(data_xuhao)

    # 5.1 Normalize LOF and relative entropy
    LOF_list_max = max(LOF_list)
    LOF_list_min = min(LOF_list)
    LOF_list = [LOF_list_max if i == -1 else i for i in LOF_list]
    if LOF_list_min != LOF_list_max:
        LOF_list = [(i - LOF_list_min) / (LOF_list_max - LOF_list_min) for i in LOF_list]

    H_as_x_max = max(H_as_x)
    H_as_x_min = min(H_as_x)
    if H_as_x_min != H_as_x_max:
        H_as_x = [(i - H_as_x_min) / (H_as_x_max - H_as_x_min) for i in H_as_x]

    # 5.2 Get the Anomaly Score
    data_xuhao_list = [int(i) for i in data_xuhao_list]
    samples_scores = np.zeros(n)
    for x in range(0, len(data)):
        for j in range(0, len(gb_dist)):
            if x in gb_dist[j].xuhao:  # If sample x is in the GB_j
                samples_scores[x] = lammda * H_as_x[j] * (1 - (len(gb_list[j]) / n) ** (1 / 3)) + (1 - lammda) * \
                                    LOF_list[data_xuhao_list.index(x)]

    return samples_scores.reshape(-1)
