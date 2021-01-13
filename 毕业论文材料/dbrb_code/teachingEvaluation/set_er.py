from datasets.load_data import load_csv_by
from teachingEvaluation.er import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

D = [100, 80, 70, 60, 40]
X_data_1 = load_csv_by('jscp_pre_semester_all.csv', path='data/student_evaluation_teacher')
X_data_2 = load_csv_by('jscp_201801_all.csv', path='data/student_evaluation_teacher')
left_1 = 100
right_1 = 105
right_2 = 120
# X = np.concatenate((X_data_1[left_1:right_1, :-2], X_data_2[right_1:right_2, :-2]), axis=0)
# X_amount = np.concatenate((X_data_1[left_1:right_1, -2:], X_data_2[right_1:right_2, -2:]), axis=0)
X = X_data_1[:, :-5]
X_amount = X_data_1[:, -5:]  # 人数
attribute_weight = [1 / len(D) for i in range(len(D))]
belief_matrix = []
for i in range(len(X)):
    criterion_i = X[i]
    criterion_belief = []
    interval = []
    for j in range(len(criterion_i)):
        interval.append(criterion_i[j])
        if (j + 1) % 5 == 0:
            criterion_belief.append([x / X_amount[i][0] for x in interval])
            interval = []
    belief_matrix.append(criterion_belief)
result_set = []
for i in range(len(belief_matrix)):
    result = evidential_reasoning(attribute_weight, belief_matrix[i], D)
    result_set.append(result)

X = X.tolist()
'''
    得到具有置信分布的数据集
'''
# print("准备写入数据...")
# out = []
# out_utility = computing_utility(D, result_set)
# for i in range(len(X_amount)):
#     out.append([out_utility[i], X_amount[i][-3], X_amount[i][-1]])
# write_csv_file_by_name("jscp_pre_semester_utility.csv", out)
# print("写入数据完成---")

# print("输入数据集的记录个数为：" + str(X.shape))
# print("ER推理得到的置信分布个数为：" + str(np.array(result_set).shape))

# 计算效用
utility = computing_utility(D, result_set)
# 计算平均值
X_mean = X
score_mean = []
for i in range(len(X_mean)):
    row = X_mean[i]
    loc = 0
    score = 0.0
    score_total = 0.0
    num = 0
    for j in range(len(row)):
        score += D[loc] * row[j]
        loc += 1
        num += row[j]
        if (j + 1) % 5 == 0:
            score_total += score / num
            num = 0
            loc = 0
            score = 0.0
    score_mean.append(score_total / len(D))

index = np.arange(len(utility))
plt.figure()
plt.plot(index, utility, color='r', linestyle='', marker='o', label='score1')
plt.bar(index, utility, color='r')
plt.plot(index, score_mean, color='b', linestyle='-', marker='^', label='score2')
plt.ylim(40, 100)
plt.title("evaluation of teaching")
plt.xlabel("course ID")
plt.ylabel("score")
plt.legend()
plt.show()
