import matplotlib.pyplot as plt
import numpy as np

# RBF with test set
# pre_score=[0.025, 0.07, 0.07, 0.011, 0.012, 0.33, 0.37, 0.38, 0.38, 0.38]
# C_value=[10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# gamma_value=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
# plt.plot(C_value,pre_score,'s-',color='r',label="C Value")
# plt.plot(gamma_value,pre_score,'o-', color='b',label="gamma Value")
# plt.xlabel("Parameter")
# plt.ylabel("Accuracy")
# # plt.ylim((0.00001,10000))
# # y_ticks=np.linspace(0.00001,10000,10)
# # plt.yticks(y_ticks)
# plt.xscale('log')
# plt.legend(loc = "best")
# plt.show()

pre_score=[0.37, 0.26, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]
C_value=[10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
gamma_value=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
plt.plot(C_value,pre_score,'s-',color='r',label="C Value")
plt.plot(gamma_value,pre_score,'o-', color='b',label="gamma Value")
plt.xlabel("Parameter")
plt.ylabel("Accuracy")
# plt.ylim((0.00001,10000))
# y_ticks=np.linspace(0.00001,10000,10)
# plt.yticks(y_ticks)
plt.xscale('log')
plt.legend(loc = "best")
plt.show()
