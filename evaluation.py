import numpy as np

def calculate_l2_distance(predictions):
    targets = np.load('datasets/data/TCLD/TEST_LABEL.npy').astype(np.float)
    predictions = predictions.astype(np.float)
    targets = targets[1:,:]
    predictions = predictions[1:,:]
    targets = targets * 512.
    predictions = predictions * 512.

    dist = np.power((predictions-targets),2)
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    dist = np.mean(dist)

    return dist




def evaluate_detailed(predictions):
    predictions = predictions *512.
    targets = np.load('datasets\\data\\TCLD\\TEST_LABEL.npy').astype(np.float)*512.
    easy_points = np.load("datasets\\data\\TCLD\\TestEasyIndex.npy")
    hard_points = np.load("datasets\\data\\TCLD\\TestHardIndex.npy")

    mean_easy = 0.0
    mean_hard = 0.0
    mean_overall = 0.0

    for i in range(len(easy_points)):
        mean_easypont = np.power((targets[easy_points[i]] - predictions[easy_points[i]]), 2)
        mean_easypont = np.sqrt(np.sum(mean_easypont))
        mean_easy = mean_easy + mean_easypont

    for m in range(len(hard_points)):
        mean_hardpoint = np.power((targets[hard_points[m]] - predictions[hard_points[m]]), 2)
        mean_hardpoint = np.sqrt(np.sum(mean_hardpoint))
        mean_hard = mean_hard + mean_hardpoint

    for k in range(len(predictions)-1):
        mean_overpoint = np.power((targets[k+1] - predictions[k+1]), 2)
        mean_overpoint = np.sqrt(np.sum(mean_overpoint))
        mean_overall = mean_overall + mean_overpoint



    mean_easy = mean_easy / len(easy_points)
    mean_hard = mean_hard / len(hard_points)
    mean_overall = mean_overall / (len(predictions)-1)

    return mean_overall,mean_easy,mean_hard