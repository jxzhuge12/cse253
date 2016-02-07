import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_log = pd.read_csv("./lenet_train.log.train")
test_log = pd.read_csv("./lenet_train.log.test")
_, ax1 = plt.subplots(figsize=(15, 10))
ax2 = ax1.twinx()
ax1.plot(train_log["NumIters"], train_log["TrainingLoss"], alpha=0.4)
ax1.plot(test_log["NumIters"], test_log["TestLoss"], 'g')
ax2.plot(test_log["NumIters"], test_log["TestAccuracy"], 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')

plt.show()