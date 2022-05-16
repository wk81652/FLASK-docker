import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
iris['data']



df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names']+['target'])
df.head()



class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                update_xi = update[0,0]*xi
                update_xi = np.array(update_xi).flatten()
                self.w_[1:] += update_xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)



model = Perceptron()
X_iris = np.matrix(df.iloc[:100,[0,2]])
#print(X_iris)
y_iris = np.matrix(df.iloc[:100,4]).reshape(100,1)
#print(y_iris)




model.fit(X_iris, y_iris)
print(model.w_)




import pickle
with open('model.pkl', 'wb') as moj_model:
    pickle.dump(model, moj_model)



with open('model.pkl', 'rb') as mm:
    perc_model = pickle.load(mm)


