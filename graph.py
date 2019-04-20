"""  #step_function

import matplotlib.pyplot as plt
import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=np.int);

c
plt.plot(x, y);
plt.plot(-0.1, 1.1);
plt.show();

"""

""" #sigmoid_function

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x));


x = np.arange(-5.0, 5.0, 0.1);
y = sigmoid(x);
plt.plot(x, y);
plt.ylim(-0.1, 1.1);
plt.show();

"""

""" #ReLU_function

import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return np.maximum(0,x);

x = np.arange(-5.0, 5.0, 0.1);
y = relu(x);
plt.plot(x, y);
#plt.ylim();
plt.show();

"""

""" #softmax

import numpy as np
import matplotlib.pyplot as plt

def softmax(add_inputs):
  #Softmax Equation
  return np.exp(add_inputs) / float(sum(np.exp(add_inputs)))

def line_graph(x, y, x_title, y_title):
  plt.plot(x, y)
  plt.xlabel(x_title)
  plt.ylabel(y_title)
  plt.show()

x = range(0, 10)
y = softmax(x)
line_graph(x, y, "Inputs", "Softmax Probability")
"""

""" # -logx
"""
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
    return math.log(x);


x = np.arange(-5.0, 5.0, 0.1);
y = sigmoid(x);
plt.plot(x, y);
plt.ylim(-0.1, 1.1);
plt.show();
