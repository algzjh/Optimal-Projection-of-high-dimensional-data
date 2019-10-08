import numpy as np


def getDSCValue(c0, c1, w):
    proj_c0 = c0.dot(w)
    proj_c1 = c1.dot(w)
    center_c0 = np.mean(proj_c0)
    center_c1 = np.mean(proj_c1)
    energy = 0
    tot_c0 = len(proj_c0)
    tot_c1 = len(proj_c1)
    for i in range(0, tot_c0):
        ay = abs(center_c0 - proj_c0[i])
        by = abs(center_c1 - proj_c0[i])
        if ay < by:
            energy += 1
    for i in range(0, tot_c1):
        ay = abs(center_c1 - proj_c1[i])
        by = abs(center_c0 - proj_c1[i])
        if ay < by:
            energy += 1
    DSC = float(energy) / (tot_c0 + tot_c1)
    return DSC


def getdDSCValue(c0, c1, w):
    proj_c0 = c0.dot(w)
    proj_c1 = c1.dot(w)
    center_c0 = np.mean(proj_c0)
    center_c1 = np.mean(proj_c1)
    energy = 0
    tot_c0 = len(proj_c0)
    tot_c1 = len(proj_c1)
    for i in range(0, tot_c0):
        ay = abs(center_c0 - proj_c0[i])
        by = abs(center_c1 - proj_c0[i])
        energy += float((by - ay)) / max(ay, by)
    for i in range(0, tot_c1):
        ay = abs(center_c1 - proj_c1[i])
        by = abs(center_c0 - proj_c1[i])
        energy += float((by - ay)) / max(ay, by)
    # print("energy: ", energy)
    # print("tot_c0: ", tot_c0)
    # print("tot_c1: ", tot_c1)
    dDSC = float(energy) / (tot_c0 + tot_c1)
    return dDSC


if __name__ == "__main__":
    c0 = np.array([[2, 3], [3, 4], [5, 6]])
    c1 = np.array([[7, 8], [9, 10], [11, 12]])
    w = np.array([7, 10])
    print("DSC: \n", getDSCValue(c0, c1, w))
    print("dDSC: \n", getdDSCValue(c0, c1, w))
