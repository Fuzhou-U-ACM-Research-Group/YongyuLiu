import numpy as np

from datasets.load_data import load_x_data_by_name, write_csv_file_by_name
from BRB.er import *
# data = load_x_data_by_name("jscp_201801_all.csv")
#
# s = ''
# for i in range(len(data)):
#     d = str(data[i][-3])
#     for j in range(8 - len(d)):
#         d = "0" + d
#     s += "\""+d+"\","
# print(s)


# all = load_x_data_by_name("jscp_201801_all_precondition.csv")
# sem = load_x_data_by_name("jscp_pre_semester_utility.csv")
#
# res = []
# for i in range(len(all)):
#     for j in range(len(sem)):
#         if all[i][-3] == sem[j][-2] and all[i][-1] == sem[j][-1]:
#             print((all[i][-2]))
#             print(all[i][-3])
#             print(all[i][-1])
#             row = [x for x in all[i, :]]
#             row.insert(-3, sem[j][0])
#             row.remove(row[-3])
#             res.append(row)
#
# write_csv_file_by_name("jscp_201801_pre_semester_all_precondition.csv", res)

a = np.zeros((2, 3))
a[0] = [1, 2, 3]
a[1] = [4, 5, 6]
b = np.zeros((2, 3))
b[0] = [0, 1, 1]
b[1] = [2, 2, 2]
c = np.zeros((2, 3))
c[0] = [3, 3, 3]
c[1] = [3, 2, 3]
c[:, 1] = a[:, 1] + b[:, 1]
# print(c)
# np.place(a[:, 1], a[:, 1] > 1, -99)
# print(a)
# print(a[:, 1] + b[:, 1])
# print(a[:, 0: 2])
# print(np.sum(a[:, 0: 2], axis=1))

for i in range(0, 10, 3):
    print(i)
