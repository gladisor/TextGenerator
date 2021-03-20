import pandas as pd
import matplotlib.pyplot as plt

## Loss and Accuracy
fontsize = 15
# data = pd.read_csv('test_loss.csv')
# x = list(range(1, 6))
# for i in range(2, 8):
#     y = data[f'{str(i)}redo ({str(i)} word) - test_loss']
#     plt.plot(x, y, label=f'N = {i}')
#
# plt.xlabel('Epoch number', fontsize=fontsize)
# plt.ylabel('Loss', fontsize=fontsize)
# plt.xlim(left=1, right=5)
# plt.xticks(ticks=range(1, 6))
# plt.legend(fontsize=fontsize)
# plt.grid()
# plt.show()

## Perplexity
data = pd.read_csv('results.csv')[:6]
x = data['num_predict_words']
y = data['perplexity']

plt.plot(x, y)
plt.xlabel('Number of Prediction Words', fontsize=fontsize)
plt.ylabel('Perplexity', fontsize=fontsize)
plt.xlim(left=2, right=7)
# plt.xticks(ticks=range(1, 6))
plt.grid()
plt.show()

# data = pd.read_csv('score_eval.csv', index_col=0)
#
# # sample = data[0:4]
# sample = data[0:len(data)-1]
# tolstoy = data.loc[len(data)-1]
#
# fontsize = 15
# for test in data.columns[1:]:
#     x = sample['num_predict']
#     plt.plot(x, sample[test], label='LSTM Models')
#     plt.plot(x, [tolstoy[test]] * len(x), label='Tolstoy')
#     plt.xlabel('Number of Prediction Words', fontsize=fontsize)
#     plt.xlim(left=0, right=5)
#     plt.ylabel(test, fontsize=fontsize)
#
#     plt.grid()
#     plt.legend(fontsize=fontsize)
#     plt.show()
