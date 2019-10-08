import numpy as np
from . import LDAfun_test
from . import NetworksxTest
import matplotlib.pyplot as plt
import networkx as nx
from . import OtherMeasurement


def readDataFromFile(filename):
    """
    :param filename: the data file, the format is class_label, dim0, dim1, ...
    :return: data and label
    """
    newfilename = './Data/' + filename  # because my data is in the Data folder
    data = []
    with open(newfilename, 'r') as f:
        for line in f:
            data.append(list(eval(line)))
    data = np.array(data)
    dim = data.shape[1]
    return data[:, 1:dim], data[:, 0]


def getOptVecFromLDA(c0, c1):
    discriminating_vectors = LDAfun_test.lda(c0, c1)
    opt_vec = np.real(discriminating_vectors[0])  # ndarray
    return opt_vec


def getRightDir(vec, last_axis):
    if np.dot(vec, last_axis) < 0:
        return -vec
    return vec


def getBetweenClassOptVec(data, label):
    """
    :param data: the input data
    :param label: The label of the data
    :return: the optimal discriminant vector corresponding to (class a, class b)
    """
    class_list = list(set(label))
    # print(class_list)
    result_dict = {}
    class_num = len(class_list)
    dim = data.shape[1]
    last_axis = np.array([0 for i in range(dim)])
    last_axis[dim - 1] = 2
    for i in range(class_num):
        for j in range(i+1, class_num, 1):
            opt_vec = getOptVecFromLDA(data[label == class_list[i]],
                                       data[label == class_list[j]])
            opt_vec = getRightDir(opt_vec, last_axis)
            result_dict[(class_list[i], class_list[j])] = opt_vec
            # drawTwoClassProjection(data[label == class_list[i]],
            #                        data[label == class_list[j]],
            #                        opt_vec)
            # print(class_list[i], class_list[j])
    return result_dict


def drawTwoClassProjection(c0, c1, w):
    new_c0 = c0.dot(w)
    new_c1 = c1.dot(w)
    # plt.hist(new_c0, 30, density=True, color='r', alpha=0.5, label="class0")
    # plt.hist(new_c1, 30, density=True, color='y', alpha=0.5, label="class1")
    # plt.legend()
    # plt.show()


def getPointOnPlane(p):
    # the last component of the corresponding point on the plane has a value of 2
    new_p = []
    dim = len(p)
    for i in range(dim-1):
        new_p.append(2.0/p[dim-1]*p[i])
    return np.array(new_p)


def getPointOnSphere(p):
    """
    already know the point on plane
    return the corresponding point on sphere
    in fact, it is just to normalize the vector
    """
    new_p = []
    sum = 0
    dim = len(p)
    for i in range(dim):
        sum += p[i]*p[i]
    sum += 2.0*2.0
    # print("sum: ", sum)
    sum = np.sqrt(sum)
    for i in range(dim):
        new_p.append(p[i]/sum)
    new_p.append(2.0/sum)
    return np.array(new_p)


def getTwoClassMean(c0, c1):
    """
    get the mean vector of two class
    """
    mean0_vectors = np.mean(c0, axis=0)
    mean1_vectors = np.mean(c1, axis=0)
    return mean0_vectors, mean1_vectors


def getClassScatterMatrix(c0, c1):
    """
    input two class data
    return the scatter for each class
    and with-in sc_mat and between sc_mat
    """
    mean0_vectors, mean1_vectors = getTwoClassMean(c0, c1)
    num0 = c0.shape[0]
    num1 = c1.shape[0]
    dim = c0.shape[1]
    class0_sc_mat = np.zeros((dim, dim))
    mv = mean0_vectors.reshape((dim, 1))
    for row in c0:
        row = row.reshape((dim, 1))
        class0_sc_mat += (row-mv).dot((row-mv).T)
    class1_sc_mat = np.zeros((dim, dim))
    mv = mean1_vectors.reshape((dim, 1))
    for row in c1:
        row = row.reshape((dim, 1))
        class1_sc_mat += (row-mv).dot((row-mv).T)
    S_W = class0_sc_mat + class1_sc_mat
    c = np.concatenate((c0, c1))  # Vertical stacking, default axis is 0
    overall_mean = np.mean(c, axis=0)
    overall_mean = overall_mean.reshape((dim, 1))
    mean_vec = mean0_vectors.reshape((dim, 1))
    S_B = np.zeros((dim, dim))
    S_B += num0 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    mean_vec = mean1_vectors.reshape((dim, 1))
    S_B += num1 * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return class0_sc_mat, class1_sc_mat, S_W, S_B


def getLDAValue(S_W, S_B, w):
    result_cost = 1.0 * w.T.dot(S_B).dot(w) / w.T.dot(S_W).dot(w)
    return result_cost


def getRangeOnAxis(border):
    """
    :param border: Boundary vector set
    :return: The maximum or minimum value on each dimension
    """
    R = []
    dim = len(border)
    # print("dim: \n", dim)
    for i in range(dim):
        R.append([border[i][0][i], border[i][1][i]])
    return R


def getTwoClassBoundingBox(c0, c1, w, op_str, threshold):
    """
    :param c0: class zero
    :param c1: class one
    :param w:  The optimal vector
    :param op_str: The optimal metric to use
    :param threshold: The threshold used to calculate the bounding box
    :return: Thresholds for two categories on a high-dimensional box
             and two categories on a projection plane
    """
    w_plane = getPointOnPlane(w)  # the optimal discriminant vector on projection plane
    plane_dim = len(w_plane)  # the dimension of projection plane
    result_border = []  # the border on the sphere
    result_border_on_plane = []  # the border on the projection plane
    if op_str == "LDA":
        # mean0, mean1 = getTwoClassMean(c0, c1)
        class0_sc_mat, class1_sc_mat, S_W, S_B = getClassScatterMatrix(c0, c1)
        # num0 = c0.shape[0]
        # num1 = c1.shape[0]
        pre_cost = getLDAValue(S_W, S_B, w)
        # print("pre_cost: \n", pre_cost)
        for i in range(plane_dim):
            one_dim_border = []
            one_dim_border_on_plane = []
            for sign in range(-1, 2, 2):
                step = 0.1
                new_w_plane = w_plane.copy()
                iterations_num = 0
                pre_w_plane = new_w_plane.copy()
                while True:
                    iterations_num += 1
                    pre_w_plane = new_w_plane.copy()
                    # print("----------------")
                    # print("new_w_plane: \n", new_w_plane)
                    # print("step: \n", step)
                    # print("sign: \n", sign)
                    add_vec = np.zeros(plane_dim)
                    add_vec[i] = sign * step
                    # print("add_vec: \n", add_vec)
                    new_w_plane += add_vec
                    # new_w_plane += sign*step
                    # print("new_w_plane: \n", new_w_plane)
                    # print("-----------------")
                    # break
                    new_w_on_s = getPointOnSphere(new_w_plane)
                    now_cost = getLDAValue(S_W, S_B, new_w_on_s)
                    # print("now_cost: \n", now_cost)
                    if now_cost < threshold:
                        # print("====== new_cost < threshold ========")
                        # print("now_cost: \n", now_cost)
                        # print("step: \n", step)
                        # print("iterations_num: \n", iterations_num)
                        # print("new_w_on_s: \n", new_w_on_s)
                        # print("====== over ========")
                        break
                    elif now_cost > pre_cost:
                        # print("====== now_cost > pre_cost ====")
                        # print("====== over  ===========")
                        break
                    elif iterations_num > 2000:
                        # print("===== iterations_num exceeding ==========")
                        # print("===== over ========")
                        break
                    step += 0.2
                # print("iterations_num: \n", iterations_num)
                # print("~~~~~~~~~~~: pre_w_plane: ", pre_w_plane)
                one_dim_border.append(getPointOnSphere(pre_w_plane).tolist())
                one_dim_border_on_plane.append(pre_w_plane.tolist())
                # print("successful!!!")
            result_border.append(one_dim_border)
            result_border_on_plane.append(one_dim_border_on_plane)
    elif op_str == "DSC":
        pre_cost = OtherMeasurement.getdDSCValue(c0, c1, w)
        for i in range(plane_dim):
            one_dim_border = []
            one_dim_border_on_plane = []
            for sign in range(-1, 2, 2):
                step = 0.1
                new_w_plane = w_plane.copy()
                iterations_num = 0
                pre_w_plane = new_w_plane.copy()
                while True:
                    iterations_num += 1
                    pre_w_plane = new_w_plane.copy()
                    add_vec = np.zeros(plane_dim)
                    add_vec[i] = sign * step
                    new_w_plane += add_vec
                    new_w_on_s = getPointOnSphere(new_w_plane)
                    now_cost = OtherMeasurement.getDSCValue(c0, c1, new_w_on_s)
                    # print("now_cost: ", now_cost)
                    if now_cost < threshold:
                        break
                    elif iterations_num > 2000:
                        break
                    step += 0.1
                one_dim_border.append(getPointOnSphere(pre_w_plane).tolist())
                one_dim_border_on_plane.append(pre_w_plane.tolist())
            result_border.append(one_dim_border)
            result_border_on_plane.append(one_dim_border_on_plane)
    elif op_str == "dDSC":
        pre_cost = OtherMeasurement.getdDSCValue(c0, c1, w)
        for i in range(plane_dim):
            one_dim_border = []
            one_dim_border_on_plane = []
            for sign in range(-1, 2, 2):
                step = 0.1
                new_w_plane = w_plane.copy()
                iterations_num = 0
                pre_w_plane = new_w_plane.copy()
                while True:
                    iterations_num += 1
                    pre_w_plane = new_w_plane.copy()
                    add_vec = np.zeros(plane_dim)
                    add_vec[i] = sign * step
                    new_w_plane += add_vec
                    new_w_on_s = getPointOnSphere(new_w_plane)
                    now_cost = OtherMeasurement.getdDSCValue(c0, c1, new_w_on_s)
                    if now_cost < threshold:
                        break
                    elif iterations_num > 100000:
                        break
                    step += 0.1
                one_dim_border.append(getPointOnSphere(pre_w_plane).tolist())
                one_dim_border_on_plane.append(pre_w_plane.tolist())
            result_border.append(one_dim_border)
            result_border_on_plane.append(one_dim_border_on_plane)
    return result_border, result_border_on_plane


def getAllBoundingBox(vec_dict, data, label, threshold, measure_option="LDA"):
    """
    :param vec_dict: the optimal discriminant vector corresponding to (class a, class b)
    :param data: the input data
    :param label: The label of the data
    :param threshold: The threshold used to calculate the bounding box
    :param measure_option: the measure used for find bounding box
    :return: On the projection plane, the bounding boxes between two categories in each dimension
    """
    result_dict = {}
    for class_tuple in vec_dict:
        c0 = data[label == class_tuple[0]]
        c1 = data[label == class_tuple[1]]
        opt_vec = vec_dict[class_tuple]
        # print("-------------Class_tuple:---------", class_tuple);
        temporary_ignored, result_border_on_plane = getTwoClassBoundingBox(c0, c1, opt_vec, measure_option, threshold)
        # print("result_border_on_plane: \n", result_border_on_plane)
        range_on_every_dim = getRangeOnAxis(result_border_on_plane)
        # print("range_on_every_dim: \n", range_on_every_dim)
        result_dict[class_tuple] = range_on_every_dim
    return result_dict


def traverseDict(mydict):
    for key in mydict:
        print(key)
        print(mydict[key])


def isBoxIntersect(R0, R1):
    dim = len(R0)
    for i in range(dim):
        a1 = R0[i][0]
        a2 = R0[i][1]
        b1 = R1[i][0]
        b2 = R1[i][1]
        if a2 < b1 or a1 > b2:
            return False
    return True


def boxIntersectRange(R0, R1):
    """
    :param R0: A bounding box
    :param R1: Another bounding box
    :return: Intersection of bounding boxes
    """
    R = []
    dim = len(R0)
    for i in range(dim):
        a1 = R0[i][0]
        a2 = R0[i][1]
        b1 = R1[i][0]
        b2 = R1[i][1]
        if a1 <= b1 <= a2 <= b2:
            R.append([b1, a2])
        elif b1 <= a1 <= b2 <= a2:
            R.append([a1, b2])
        elif a1 <= b1 <= b2 <= a2:  # elif a1 < b1 and b2 < a2:
            R.append([b1, b2])
        elif b1 <= a1 <= a2 <= b2:  # elif b1 < a1 and a2 < b2:
            R.append([a1, a2])
        else:
            R.append([])
    return R


def getPlaneRangetoSphere(boxRange):
    plane_p = []
    for r in boxRange:
        plane_p.append((r[0] + r[1]) / 2.0)
    sphere_p = getPointOnSphere(plane_p)
    # print("plane_p: \n", plane_p)
    return sphere_p.tolist()


def testIntersect(box_dict):
    combined_box_dict = box_dict.copy()
    key_list = list(box_dict.keys())
    # print("key_list: \n", key_list)
    print("============= Below is the bounding box pair with intersections =========================")
    key_num = len(key_list)
    for i in range(key_num):
        for j in range(i+1, key_num, 1):
            # print(key_list[i], key_list[j])
            if isBoxIntersect(box_dict[key_list[i]], box_dict[key_list[j]]):
                # print("have Intersection!")
                print(key_list[i], key_list[j])
                boxRange = boxIntersectRange(box_dict[key_list[i]], box_dict[key_list[j]])
                print("Intersection range: \n", boxRange)
                print("Corresponding vector in the original dimension： \n", getPlaneRangetoSphere(boxRange))
    # if isBoxIntersect(box_dict[(1, 4)], box_dict[(1, 5)]):
    #     combined_box_dict[(1, 4, 1, 5)] = boxIntersectRange(box_dict[(1, 4)], box_dict[(1, 5)])
    #     del combined_box_dict[(1, 4)]
    #     del combined_box_dict[(1, 5)]
    #     print("There is an intersection!")
    # else:
    #     print("There is no intersection！")
    return combined_box_dict


def isIntersection(s1, s2):
    """
    :param s1: a set
    :param s2: anothre set
    :return: if the two set have intersection return true otherwise return false
    """
    s3 = s1.intersection(s2)
    return len(s3) != 0


def getMinSetList(my_dict, tot_node_num):
    """
    :param my_dict: the dictionary of the clique
    :param tot_node_num: the total number of nodes
    :return: The minimum number of clique that can cover all the points
    """
    result_list = []  # the result list of minimum set
    key_list = list(my_dict.keys())  # the list of the number of vertexes in clique
    key_list.sort(reverse=True)  # sort the list from large to small
    print("key_list: \n", key_list)
    k_num = key_list[0]  # the largest vertexes number
    k_list = my_dict[k_num]  # the list corresponding to the largest vertexes number

    max_ver_num = k_num
    st = 0
    if k_num != tot_node_num:
        result_list.append(k_list[0])  # the largest clique which not include all nodes
    else:
        st += 1  # the largest acceptable clique is the smaller clique, so jump to the next
        max_ver_num -= 1
    k_total = len(key_list)  # the tot different number of clique vertexes
    for i in range(st, k_total):
        k_num = key_list[i]  # Number of vertices being processed
        k_list = my_dict[k_num]  # the corresponding clique
        for it1 in k_list:
            isnew = True
            for it2 in result_list:
                if isIntersection(it1, it2):
                    isnew = False
                    break
            if isnew:
                result_list.append(it1)
    if len(result_list) % 2 == 1:
        if max_ver_num != 1:
            ts = result_list[0]
            ts_len = len(ts)
            ts1 = set()
            ts2 = set()
            cnt = 1
            for it in ts:
                if cnt <= int(ts_len / 2):
                    ts1.add(it)
                else:
                    ts2.add(it)
                cnt += 1
            result_list[0] = ts1
            result_list.append(ts2)
        else:
            ts = result_list[0]
            result_list.append(ts)
    return result_list


def nodeMapping(box_dict):
    """
    :param box_dict: The bounding box corresponding to two categories on the projection plane
    :return: Two categories to id, id to two categories, a list of category pairs
    """
    pair2id = {}
    id2pair = {}
    id = 0
    key_list = list(box_dict.keys())
    print("key_list: \n", key_list)
    for it in key_list:
        pair2id[it] = id
        id2pair[id] = it
        id += 1
    return pair2id, id2pair, key_list


def getEdge(pair2id, id2pair, class_pair_list, box_dict):
    """
    :param pair2id: Two categories to id
    :param id2pair: id to two categories
    :param class_pair_list:
    :param box_dict:
    :return: A list of edge sets consisting of binary pairs of nodes
    """
    my_add_edges = []
    class_pair_tot = len(class_pair_list)
    for i in range(class_pair_tot):
        for j in range(i+1, class_pair_tot):
            Ri = box_dict[class_pair_list[i]]
            Rj = box_dict[class_pair_list[j]]
            if isBoxIntersect(Ri, Rj):
                my_add_edges.append([pair2id[class_pair_list[i]],
                                     pair2id[class_pair_list[j]]])
    return my_add_edges


def getCliqueDict(my_add_nodes, my_add_edges):
    """
    :param my_add_nodes: Point set
    :param my_add_edges: Edge set
    :return: Returns the dictionary of the clique.
             See the explanation of this variable for details
    """
    graph = nx.Graph()
    graph.add_nodes_from(my_add_nodes)
    graph.add_edges_from(my_add_edges)
    # plt.figure(figsize=(12, 12))
    # plt.figure()
    nx.draw_networkx(graph)
    # plt.show()
    clique_dict = NetworksxTest.print_cliques(graph)
    one_list = [{it} for it in my_add_nodes]
    clique_dict[1] = one_list
    return clique_dict


def getMinAxis(minsetlist, box_dict, pair2id, id2pair):
    """
    :param minsetlist: The minimum number of clique that can cover all the points
    :param box_dict: The corresponding bounding box on the projection plane
    :param pair2id:  Two categories to id
    :param id2pair:  id to two categories
    :return: Take the center of the bounding box and return to the high dimensional sphere
    """
    result_axis_list = []
    for s1 in minsetlist:
        first = True
        for it1 in s1:
            if first:
                tmp_plane_range = box_dict[id2pair[it1]]
                first = False
            else:
                tmp_plane_range = boxIntersectRange(tmp_plane_range, box_dict[id2pair[it1]])
        result_axis_list.append(getPlaneRangetoSphere(tmp_plane_range))
    return result_axis_list


def getCorrespondingColor(id):
    color_list = ['black', 'gray', 'firebrick', 'red', 'chocolate',
                  'orange', 'peru', 'darkgoldenrod', 'olive',
                  'yellow', 'green', 'turquoise', 'deepskyblue',
                  'blue', 'violet', 'purple', 'deeppink']
    return color_list[id]


def getOneAxisClass(s1, id2pair):
    result_set = set()
    for it1 in s1:
        tmp_pair = id2pair[it1]
        result_set.add(tmp_pair[0])
        result_set.add(tmp_pair[1])
    return result_set


def drawPlotSet(result_axis_list, minsetlist, data, label, id2pair):
    assert len(result_axis_list) == len(minsetlist), "Something went wrong！！"
    if len(result_axis_list) & 1:
        result_axis_list.append(result_axis_list[0])
        minsetlist.append(minsetlist[0])
    print("result_axis_list: \n", result_axis_list)
    print("minsetlist: \n", minsetlist)
    plot_tot = int(len(result_axis_list) / 2)
    row_num = 2
    col_num = int(np.ceil(plot_tot / 2.0))
    print("row_num: \n", row_num)
    print("col_num: \n", col_num)


    # row_num -= 1

    fig, ax = plt.subplots(nrows=row_num, ncols=col_num, squeeze=False)
    print("ax: \n", ax)
    pos = 0
    plot_num_now = 0
    for i in range(row_num):
        for j in range(col_num):
            print("i: \n", i)
            print("j: \n", j)
            s1 = getOneAxisClass(minsetlist[pos], id2pair)
            print("s1: \n", s1)
            ax[i][j].set_xlabel(str(s1))
            s2 = getOneAxisClass(minsetlist[pos + 1], id2pair)
            print("s2: \n", s2)
            ax[i][j].set_ylabel(str(s2))
            plot_class_set = s1 | s2
            print("plot_class_set: \n", plot_class_set)
            for it in plot_class_set:
                x = np.dot(data[label == it], result_axis_list[pos])
                y = np.dot(data[label == it], result_axis_list[pos + 1])
                print("x: \n", len(x))
                print("y: \n", len(y))
                print("it: \n", it)
                print("getCorrespondingColor(it): \n", type(getCorrespondingColor(it)))
                # ax[i, j].subplot(i, j)
                ax[i][j].scatter(x, y, color=getCorrespondingColor(it), label=str(it), alpha=0.5, s=10)
            ax[i][j].legend()
            pos += 2
            plot_num_now += 1
            if plot_num_now == plot_tot:
                break
        if plot_num_now == plot_tot:
            break
            # col.plot(x, y)

    if plot_tot & 1:
        # pass
        fig.delaxes(ax[1, col_num - 1])

    # plt.legend()
    plt.suptitle(datafilename)
    plt.show()


def getLabelRight(label):
    class_label = list(set(label))
    label_dict = {}
    re_label = 0
    for it in class_label:
        label_dict[it] = re_label
        re_label += 1
    new_label = []
    for it in label:
        new_label.append(label_dict[it])
    return np.array(new_label)

def update_drawPlotSet(result_axis_list, minsetlist, data, label, id2pair):
    assert len(result_axis_list) == len(minsetlist), "Something went wrong！！"
    if len(result_axis_list) & 1:
        result_axis_list.append(result_axis_list[0])
        minsetlist.append(minsetlist[0])
    print("result_axis_list: \n", result_axis_list)
    print("minsetlist: \n", minsetlist)
    plot_tot = int(len(result_axis_list) / 2)
    row_num = 2
    col_num = int(np.ceil(plot_tot / 2.0))
    print("row_num: \n", row_num)
    print("col_num: \n", col_num)


    # row_num -= 1

    # fig, ax = plt.subplots(nrows=row_num, ncols=col_num, squeeze=False)
    # print("ax: \n", ax)
    result_list = []
    pos = 0
    plot_num_now = 0
    for i in range(row_num):
        for j in range(col_num):
            # print("i: \n", i)
            # print("j: \n", j)
            s1 = getOneAxisClass(minsetlist[pos], id2pair)
            # print("s1: \n", s1)
            # ax[i][j].set_xlabel(str(s1))
            s2 = getOneAxisClass(minsetlist[pos + 1], id2pair)
            # print("s2: \n", s2)
            # ax[i][j].set_ylabel(str(s2))
            plot_class_set = s1 | s2
            print("plot_class_set: \n", plot_class_set)
            tmp_x = []
            tmp_y = []
            tmp_label = []
            for it in plot_class_set:
                x = np.dot(data[label == it], result_axis_list[pos])
                y = np.dot(data[label == it], result_axis_list[pos + 1])
                print("x: \n", len(x))
                print("y: \n", len(y))
                print("it: \n", it)
                tmp_x = tmp_x + list(x)
                tmp_y = tmp_y + list(y)
                tmp_label = tmp_label + [it for i in range(len(x))]
                print("getCorrespondingColor(it): \n", type(getCorrespondingColor(it)))
                # ax[i, j].subplot(i, j)
                # ax[i][j].scatter(x, y, color=getCorrespondingColor(it), label=str(it), alpha=0.5, s=10)
            # ax[i][j].legend()
            result_list.append([tmp_x, tmp_y, tmp_label])
            pos += 2
            plot_num_now += 1
            if plot_num_now == plot_tot:
                break
        if plot_num_now == plot_tot:
            break
            # col.plot(x, y)

    # if plot_tot & 1:
    #     # pass
    #     fig.delaxes(ax[1, col_num - 1])

    # plt.legend()
    # plt.suptitle(datafilename)
    # plt.show()
    return result_list


def getpLDA_pass(data, label, threshold, measure_option, evaluation):
    vec_dict = getBetweenClassOptVec(data, label)  # the optimal projection direction between each class

    box_dict = getAllBoundingBox(vec_dict, data, label, threshold, measure_option)  # the bounding box between each class

    pair2id, id2pair, class_pair_list = nodeMapping(box_dict)

    # print("pair2id: \n", pair2id)
    # print("id2pair: \n", id2pair)
    my_add_nodes = [pair2id[it] for it in class_pair_list]
    my_add_edges = getEdge(pair2id, id2pair, class_pair_list, box_dict)
    print("my_add_nodes: \n", my_add_nodes)
    print("my_add_edges: \n", my_add_edges)
    clique_dict = getCliqueDict(my_add_nodes, my_add_edges)

    print("clique_dict: \n", clique_dict)
    minsetlist = getMinSetList(clique_dict, len(my_add_nodes))
    print("minset: \n", minsetlist)
    print("minset_len: \n", len(minsetlist))
    result_axis_list = getMinAxis(minsetlist, box_dict, pair2id, id2pair)
    print("result_axis_list: \n", result_axis_list)
    result_list = update_drawPlotSet(result_axis_list, minsetlist, data, label, id2pair)
    return result_list


# if __name__ == "__main__":
    """
    LDA
    bbdm13_origClassLabels.data   0.74
    world_11d.data  1.69  DSC 0.97
    worldmap_origClassLabels.data  2.07
    ms_interleaved_40_80_3d_0.data   2
    DSC:
    ms_interleaved_40_80_3d_0.data  0.99
    bbdm13_origClassLabels.data 0.96
    not  world_11d.data  0.97
    not  worldmap_origClassLabels.data 0.98
    not out_boston_origClassLabels.data 0.99
    n100-d10-c3-spr0.1-out0.data 1.0
    ForestTypes.data 0.95
    hiv.data 1.0
    """

    """
    # Change the following two lines to fit your data
    datafilename = "iris_2.csv"  # the class data file
    threshold = 0.95  # the threshold of class distinction, now use lda, later will use DSC etc..
    measure_option = "DSC"
    # ============================================
    data, label = readDataFromFile(datafilename)

    label = np.array(list(map(int, label)))
    label = getLabelRight(label)

    vec_dict = getBetweenClassOptVec(data, label)  # the optimal projection direction between each class

    box_dict = getAllBoundingBox(vec_dict, data, label, threshold, measure_option)  # the bounding box between each class

    pair2id, id2pair, class_pair_list = nodeMapping(box_dict)

    # print("pair2id: \n", pair2id)
    # print("id2pair: \n", id2pair)
    my_add_nodes = [pair2id[it] for it in class_pair_list]
    my_add_edges = getEdge(pair2id, id2pair, class_pair_list, box_dict)
    print("my_add_nodes: \n", my_add_nodes)
    print("my_add_edges: \n", my_add_edges)
    clique_dict = getCliqueDict(my_add_nodes, my_add_edges)

    print("clique_dict: \n", clique_dict)
    minsetlist = getMinSetList(clique_dict, len(my_add_nodes))
    print("minset: \n", minsetlist)
    print("minset_len: \n", len(minsetlist))
    result_axis_list = getMinAxis(minsetlist, box_dict, pair2id, id2pair)
    print("result_axis_list: \n", result_axis_list)
    drawPlotSet(result_axis_list, minsetlist, data, label, id2pair, datafilename)
    # traverseDict(box_dict)
    # combined_box_dict = testIntersect(box_dict)
    """
