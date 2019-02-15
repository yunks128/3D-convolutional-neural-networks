import matplotlib.pyplot as plt
import pickle
import numpy as np

# define empty list
fpr = []

# open file and read the content in a list
with open('objects_all.pkl', 'rb') as f:  
    fpr, tpr, auc_keras_val = pickle.load(f)
with open('objects_region1.pkl', 'rb') as f:  
    fpr1, tpr1, auc_keras_val1 = pickle.load(f)
with open('objects_region2.pkl', 'rb') as f:  
    fpr2, tpr2, auc_keras_val2 = pickle.load(f)
with open('objects_region3.pkl', 'rb') as f:  
    fpr3, tpr3, auc_keras_val3 = pickle.load(f)
with open('objects_region4.pkl', 'rb') as f:  
    fpr4, tpr4, auc_keras_val4 = pickle.load(f)
with open('objects_region5.pkl', 'rb') as f:  
    fpr5, tpr5, auc_keras_val5 = pickle.load(f)
with open('objects_region6.pkl', 'rb') as f:  
    fpr6, tpr6, auc_keras_val6 = pickle.load(f)
with open('objects_region7.pkl', 'rb') as f:  
    fpr7, tpr7, auc_keras_val7 = pickle.load(f)
with open('objects_region8.pkl', 'rb') as f:  
    fpr8, tpr8, auc_keras_val8 = pickle.load(f)
np.savetxt('fpr.txt', (fpr), fmt="%5.2f")
np.savetxt('tpr.txt', (tpr), fmt="%5.2f")
np.savetxt('fpr1.txt', (fpr1), fmt="%5.2f")
np.savetxt('tpr1.txt', (tpr1), fmt="%5.2f")
np.savetxt('fpr2.txt', (fpr2), fmt="%5.2f")
np.savetxt('tpr2.txt', (tpr2), fmt="%5.2f")
np.savetxt('fpr3.txt', (fpr3), fmt="%5.2f")
np.savetxt('tpr3.txt', (tpr3), fmt="%5.2f")
np.savetxt('fpr4.txt', (fpr4), fmt="%5.2f")
np.savetxt('tpr4.txt', (tpr4), fmt="%5.2f")
np.savetxt('fpr5.txt', (fpr5), fmt="%5.2f")
np.savetxt('tpr5.txt', (tpr5), fmt="%5.2f")
np.savetxt('fpr6.txt', (fpr6), fmt="%5.2f")
np.savetxt('tpr6.txt', (tpr6), fmt="%5.2f")
np.savetxt('fpr7.txt', (fpr7), fmt="%5.2f")
np.savetxt('tpr7.txt', (tpr7), fmt="%5.2f")
np.savetxt('fpr8.txt', (fpr8), fmt="%5.2f")
np.savetxt('tpr8.txt', (tpr8), fmt="%5.2f")

print(auc_keras_val)
print(auc_keras_val1)
print(auc_keras_val2)
print(auc_keras_val3)
print(auc_keras_val4)
print(auc_keras_val5)
print(auc_keras_val6)
print(auc_keras_val7)
print(auc_keras_val8)


fig=plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='3DCNN St Mary''s Hospital (AUROC = {:.3f})'.format(auc_keras_val))
plt.plot(fpr3, tpr3, label='3DCNN R Frontal (AUROC = {:.3f})'.format(auc_keras_val3))
plt.plot(fpr4, tpr4, label='3DCNN L Frontal (AUROC = {:.3f})'.format(auc_keras_val4))
plt.plot(fpr2, tpr2, label='3DCNN R Temporal (AUROC = {:.3f})'.format(auc_keras_val2))
plt.plot(fpr5, tpr5, label='3DCNN L Temporal (AUROC = {:.3f})'.format(auc_keras_val5))
plt.plot(fpr1, tpr1, label='3DCNN R Parietal (AUROC = {:.3f})'.format(auc_keras_val1))
plt.plot(fpr6, tpr6, label='3DCNN L Parietal (AUROC = {:.3f})'.format(auc_keras_val6))
plt.plot(fpr8, tpr8, label='3DCNN R Occipital (AUROC = {:.3f})'.format(auc_keras_val8))
plt.plot(fpr7, tpr7, label='3DCNN L Occipital (AUROC = {:.3f})'.format(auc_keras_val7))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
fig.savefig('ROC_curve_all.png')