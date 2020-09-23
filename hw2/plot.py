import numpy as np

curve_sb1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_sb_no_rtg_dsa.npy")
curve_sb2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_sb_rtg_dsa.npy")
curve_sb3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_sb_rtg_na.npy")

# print(curve_sb1)
# print(curve_sb2)
# print(curve_sb3)

curve_lb1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_lb_no_rtg_dsa.npy")
curve_lb2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_lb_rtg_dsa.npy")
curve_lb3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_lb_rtg_na.npy")

from matplotlib import pyplot as plt

plt.figure()
plt.ylim([0,220])
plt.plot(curve_sb1, label='q1_sb_no_rtg_dsa')
plt.plot(curve_sb2, '--', label='q1_sb_rtg_dsa')
plt.plot(curve_sb3, '.', label='q1_sb_rtg_na')
plt.legend()
# plt.show()

plt.figure()
plt.ylim([0,220])
plt.plot(curve_lb1, label='q1_lb_no_rtg_dsa')
plt.plot(curve_lb2, '--', label='q1_lb_rtg_dsa')
plt.plot(curve_lb3, '-.', label='q1_lb_rtg_na')
plt.legend()
plt.show()