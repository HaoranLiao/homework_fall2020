import numpy as np

curve_sb1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_sb_no_rtg_dsa.npy")
curve_sb2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_sb_rtg_dsa.npy")
curve_sb3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_sb_rtg_na.npy")

curve_lb1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_lb_no_rtg_dsa.npy")
curve_lb2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_lb_rtg_dsa.npy")
curve_lb3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q1_lb_rtg_na.npy")

curve_q2_1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q2_b3000_r0.01.npy")
curve_q2_2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q2_b4000_r0.008.npy")
curve_q2_3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q2_b5000_r0.008.npy")

curve_q3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q3_b40000_r0.005_2.npy")

curve_q4_1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b10000_lr0.005_rtg_nnbaseline.npy")
curve_q4_2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b10000_lr0.01_rtg_nnbaseline.npy")
curve_q4_3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b10000_lr0.02_rtg_nnbaseline.npy")
curve_q4_4 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b30000_lr0.005_rtg_nnbaseline.npy")
curve_q4_5 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b30000_lr0.01_rtg_nnbaseline.npy")
curve_q4_6 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b30000_lr0.02_rtg_nnbaseline.npy")
curve_q4_7 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b50000_lr0.005_rtg_nnbaseline.npy")
curve_q4_8 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b50000_lr0.01_rtg_nnbaseline.npy")
curve_q4_9 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_search_b50000_lr0.02_rtg_nnbaseline.npy")

curve_q4_run_1 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_b30000_r0.02.npy")
curve_q4_run_2 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_b30000_r0.02_rtg.npy")
curve_q4_run_3 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_b30000_r0.02_nnbaseline.npy")
curve_q4_run_4 = np.load("/Users/haoran/Documents/GitHub/homework_fall2020/hw2/data/q4_b30000_r0.02_rtg_nn_baseline.npy")

from matplotlib import pyplot as plt

################## q1 #################
# plt.figure()
# plt.ylim([0,220])
# plt.plot(curve_sb1, label='q1_sb_no_rtg_dsa')
# plt.plot(curve_sb2, label='q1_sb_rtg_dsa')
# plt.plot(curve_sb3, label='q1_sb_rtg_na')
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Average Evaluation Returns")
# plt.show()

# plt.figure()
# plt.ylim([0,220])
# plt.plot(curve_lb1, label='q1_lb_no_rtg_dsa')
# plt.plot(curve_lb2, label='q1_lb_rtg_dsa')
# plt.plot(curve_lb3, label='q1_lb_rtg_na')
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Average Evaluation Returns")
# plt.show()

################## q2 #################
# plt.figure()
# # plt.plot(curve_q2_1, label='b3000_r0.01')
# # plt.plot(curve_q2_2, label='b4000_r0.008')
# plt.plot(curve_q2_3, label='b5000_r0.008')
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Average Evaluation Returns")
# plt.show()


################# q3 #################
# plt.figure()
# plt.plot(curve_q3, label='q3')
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Average Evaluation Returns")
# plt.show()


################# q4 search ##################
# plt.figure()
# plt.plot(curve_q4_1, label='b10000_lr0.005')
# plt.plot(curve_q4_2, label='b10000_lr0.01')
# plt.plot(curve_q4_3, label='b10000_lr0.02')
# plt.plot(curve_q4_4, label='b30000_lr0.005')
# plt.plot(curve_q4_5, label='b30000_lr0.01')
# plt.plot(curve_q4_6, label='b30000_lr0.02')
# plt.plot(curve_q4_7, label='b50000_lr0.005')
# plt.plot(curve_q4_8, label='b50000_lr0.01')
# plt.plot(curve_q4_9, label='b50000_lr0.02')
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Average Evaluation Returns")
# plt.show()

################ q4 #########################
plt.figure()
plt.plot(curve_q4_run_1, label='none')
plt.plot(curve_q4_run_2, label='rtg')
plt.plot(curve_q4_run_3, label='baseline')
plt.plot(curve_q4_run_4, label='rtg and baseline')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Average Evaluation Returns")
plt.show()

