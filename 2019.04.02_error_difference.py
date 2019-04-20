import tensorflow as tf
import numpy as np

def mean_squred_error(y, t):
    return 0.5 * np.sum((y-t)**2);

def cross_entropy_error(y, t):
    delta = 1e-7;

    batch_size = y.shape[0];
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size;

    return -np.sum(t * np.log(y + delta));

#실제데이터 = '2'
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
#정답일경우 y1[2] = 0.6
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0];
#오답일경우 y2[2] = 0.1
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0];

print("CASE 1 : 2가 정답일때 신경망의 정답이 맞을경우");
print("평균 제곱 오차    : %f" % mean_squred_error(np.array(y1), np.array(t)));
print("교차 엔트로피 오차 : %f" % cross_entropy_error(np.array(y1), np.array(t)));

print("");
print("CASE 2 : 2가 정답일때 신경망의 정답이 틀릴경우");
print("평균 제곱 오차    : %f" % mean_squred_error(np.array(y2), np.array(t)));
print("교차 엔트로피 오차 : %f" % cross_entropy_error(np.array(y2), np.array(t)));
