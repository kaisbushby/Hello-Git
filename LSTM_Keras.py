import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow import random_shuffle
from sklearn.utils import shuffle

M = 2           # Input Dimension
K = 3           # Class Output
n = 100         # Data Count
N = n*K         # Total Data Size
batch_size = 50 # Mini Batch Size
n_batches = N

model = Sequential([
    Dense(input_dim=M, units=K),
    Activation('sigmoid')
])

X1 = np.random.randn(n, M) + np.array([0, 10])
X2 = np.random.randn(n, M) + np.array([5, 5])
X3 = np.random.randn(n, M) + np.array([10, 0])
Y1 = np.array([[1, 0, 0] for i in range(n)])
Y2 = np.array([[0, 1, 0] for i in range(n)])
Y3 = np.array([[0, 0, 1] for i in range(n)])

X = np.concatenate((X1, X2, X3), axis=0)
Y = np.concatenate((Y1, Y2, Y3), axis=0)
X_, Y_ = shuffle(X, Y)


model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.1))
model.fit(X, Y, epochs=20, batch_size=batch_size)

classes = model.predict_classes(X_[0:10], batch_size=batch_size)
prob = model.predict_proba(X_[0:10], batch_size=1)

np.set_printoptions(suppress=True)

print("Classified: ")
print(np.argmax(model.predict(X_[0:10]), axis=1) == classes)
print()
print("Output Probability")
print(prob)
