from data import create_dataset
import time

if __name__ == '__main__':
    # for i in range(2, 6):
    # for i in [1]:
    #     start = time.time()
    #     create_dataset(num_predict_words=2**i)
    #     print(time.time() - start)
    create_dataset(num_predict_words=7)
