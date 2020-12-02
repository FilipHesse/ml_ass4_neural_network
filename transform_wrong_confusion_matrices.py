import pickle
import numpy as np

f = open('iris_perceptron_C.pckl', 'rb')
C_wrong = pickle.load(f)
f.close()

C_correct = {}

for key in C_wrong:
    C = C_wrong[key]
    C_correct[key] = np.array([[ C[1,1], C[1,0]], [ C[0,1], C[0,0] ]])

for element in C_wrong:
    print("eta={}, k={}:\n C={}".format(element[0], element[1], C_wrong[element]))

for element in C_correct:
    print("eta={}, k={}:\n C={}".format(element[0], element[1], C_correct[element]))

f = open('iris_perceptron_C.pckl', 'wb')
pickle.dump(C_correct, f)
f.close()

f = open('iris_adaline_C.pckl', 'rb')
C_wrong = pickle.load(f)
f.close()

C_correct = {}

for key in C_wrong:
    C = C_wrong[key]
    C_correct[key] = np.array([[ C[1,1], C[1,0]], [ C[0,1], C[0,0] ]])

for element in C_wrong:
    print("eta={}, k={}:\n C={}".format(element[0], element[1], C_wrong[element]))

for element in C_correct:
    print("eta={}, k={}:\n C={}".format(element[0], element[1], C_correct[element]))

f = open('iris_adaline_C.pckl', 'wb')
pickle.dump(C_correct, f)
f.close()