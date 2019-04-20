import matplotlib.pyplot as plt
import numpy as np
x = [0,0,1,1];
y = [0,1,0,1];

plt.plot([1,1,0],[1,0,1],'bo',label = 'False');
plt.plot([0],[0],'ro',label = 'True');
plt.xticks(np.arange(0,1.1,1));
plt.yticks(np.arange(0,1.1,1));
plt.title('OR');
plt.show();
