import numpy as np

def step_function(x):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1;
        else:
            x[i] = 0;
    return x;

x1 = [0.0, 0.0, 1.0, 1.0];
x2 = [0.0, 1.0, 0.0, 1.0];
t = [0, 1, 1, 1];

w0 = 0.6;
w1 = 0.5;
w2 = 1.0;

x0 = -1;

tmp = [0, 0, 0, 0];

print("before train : w0 = {}, w1 = {}, w2 = {}".format(w0, w1, w2));
for _ in range(10):
    for i in range(4):
        tmp[i] = x1[i] * w1 + x2[i] * w2 + w0 * x0;

    y = step_function(tmp);
    for i in range(4):
        if y[i] != t[i]:
            if x1 == 0 and x2 == 0:
                w0 = w0 + 0.1 * x0 * (0 - y[i]);
                w1 = w1 + 0.1 * x1[i] * (0 - y[i]);
                w2 = w2 + 0.1 * x2[i] * (0 - y[i]);
            else:
                w0 = w0 + 0.1 * x0 * (1 - y[i]);
                w1 = w1 + 0.1 * x1[i] * (1 - y[i]);
                w2 = w2 + 0.1 * x2[i] * (1 - y[i]);

print("after train : w0 = {}, w1 = {}, w2 = {}".format(w0, w1, w2));

print("t = {}".format(t));
print("predict = {}".format(y));
