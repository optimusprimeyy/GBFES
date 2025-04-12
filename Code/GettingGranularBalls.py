# The file is called to generate the GBs by main function.
import numpy as np
from sklearn.cluster import k_means

# The split stage according to DM.
def splitting_ball(gb_list, L):
    # input:
    # gb_list is the list of GBs, where each element represents the array of all the samples in the GB.
    # L is the parameter used to set the minimum number of samples in GB.
    gb_list_new = []
    for gb in gb_list:
        if gb.shape[0] >= L:
            ball_1, ball_2 = kmeans_ball(gb)  # Split the GB using 2-means
            len_original = gb.shape[0]
            len_1 = ball_1.shape[0]
            len_2 = ball_2.shape[0]
            DM_original = get_DM(gb)  # Calculate the SD of the original GB

            if len(ball_1) == 0:  # The samples were all counted in ball_2
                gb_list_new.append(ball_2)
                continue
            elif len(ball_2) == 0:  # The samples were all counted in ball_1
                gb_list_new.append(ball_1)
                continue

            DM_k_1 = get_DM(ball_1)  # Calculate the DM of sub-GB 1
            DM_k_2 = get_DM(ball_2)  # Calculate the DM of sub-GB 2

            DM_weight = DM_k_1 * ( len_1 / len_original ) + DM_k_2 * ( len_2 / len_original ) # Calculate the DM_weight of sub-GBs

            # Splitting criterion
            if DM_weight < DM_original:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb)
        else:
            gb_list_new.append(gb)
    return gb_list_new


# Calculate the Sum Distance SD of GBs.
def get_DM(gb):
    # input:
    # gb is a array including all samples in the gb.
    data = gb[:, :-1]  # Delete serial number column
    center = data.mean(0)
    DM = np.mean(((data - center) ** 2).sum(axis=1) ** 0.5)
    return DM

# Split GBs using 2-means.
def kmeans_ball(gb):
    # input:
    # gb is a array including all samples in the gb.
    data = gb[:, :-1]
    cluster = k_means(X=data, init='k-means++', n_clusters=2)[1]
    ball1 = gb[cluster == 0, :]
    ball2 = gb[cluster == 1, :]
    return [ball1, ball2]


def GettingGranularBalls(data):
    # input:
    # data is the data without labels

    # Join index in last column
    L = 8 # Used to set the minimum number of samples in GB
    index = np.arange(0, data.shape[0], 1)
    data_index = np.insert(data, data.shape[1], values=index, axis=1)  # Add index to last column for each sample.
    gb_list_temp = [data_index]  # Start by treating the entire dataset as a GB.

    # Split until the number of GBs no longer changes.
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = splitting_ball(gb_list_temp, L)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    # Get the final GB list.
    gb_list_final = gb_list_temp

    return gb_list_final
