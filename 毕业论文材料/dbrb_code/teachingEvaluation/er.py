def evidential_reasoning(attribute_weights, belief_matrix, D):
    """
    ER推理
    :param attribute_weights: 参数权重[criterion weight 1, criterion weight 2, ..., criterion weight N]
    :param belief_matrix: 置信分布矩阵[criterion, evaluation]，二维
    :param D: 评级等级集合[evaluation 1, evaluation 2,...,evaluation M]
    :return: [belief 1, ..., belief M, belief_H]
    """
    evaluation_beta_sum = []  # 存放每个criterion的等级置信度之和
    for i in range(len(belief_matrix)):
        criterion = belief_matrix[i]
        sum_tmp = 0.0
        for k in range(len(criterion)):
            sum_tmp += criterion[k]
        evaluation_beta_sum.append(sum_tmp)

    fire_pare = [1.0] * len(D)
    fire_pare_sum = 0.0
    for i in range(len(D)):
        for k in range(len(belief_matrix)):
            fire_pare[i] *= attribute_weights[k]*belief_matrix[k][i] + 1 - attribute_weights[k] * evaluation_beta_sum[k]
        fire_pare_sum += fire_pare[i]

    sec_para = 1.0
    wec_para = 1.0
    for i in range(len(belief_matrix)):
        sec_para *= (1 - attribute_weights[i] * evaluation_beta_sum[i])
        wec_para *= (1 - attribute_weights[i])
    belief = []
    sum = 0
    var1 = 0
    for i in range(len(D)):
        var1 = (fire_pare[i] - sec_para) / (fire_pare_sum - (len(D) - 1) * sec_para - wec_para)
        sum += var1
        belief.append(var1)
    belief.append(1-sum)
    return belief


def computing_utility(D, belief_matrix):
    """
    计算效用
    :param D: 评价结果集，array-like
    :param belief_matrix: 2D array-like
    :return:
    """
    result_set = []
    for row in range(len(belief_matrix)):
        u_i = 0.0
        u = 0.0
        for data_index in range(len(D)):
            u_i += belief_matrix[row][data_index] * D[data_index]
            u += D[data_index]
        result_set.append(u_i + u * belief_matrix[row][-1] / len(D))
    return result_set
