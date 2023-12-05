import keras
import keras.backend as K
from keras import initializers
from keras.models import load_model, clone_model
import sys
import h5py
import numpy as np
from keras import Model
from keras.layers import Lambda
import keras.backend as K
import tensorflow as tf
import pandas as pd

keras.backend.clear_session()

bd_test_path = 'bd_test.h5'
test_path = 'test.h5'
bd_net_path = 'bd_net.h5'
bd_weights_path = 'bd_weights.h5'
valid_path = 'valid.h5'

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data


cl_x_test, cl_y_test = data_loader(test_path)
bd_x_test, bd_y_test = data_loader(bd_test_path)
cl_x_val, cl_y_val = data_loader(valid_path)
print(bd_y_test.shape)
bd_model = keras.models.load_model(bd_net_path)
bd_model.load_weights(bd_weights_path)

cl_label_p = np.argmax(bd_model.predict(cl_x_test), axis=1)
clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
print('Clean Classification accuracy:', clean_accuracy)

bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
print('Attack Success Rate:', asr)


act = bd_model.get_layer("pool_3").output
model_act = Model(inputs=bd_model.input, outputs=act)
activations = model_act.predict(cl_x_val)

index = np.argsort(np.mean(activations,axis=(0,1,2)))
no_prune_acc=clean_accuracy


new_model = clone_model(bd_model)
new_model.load_weights(bd_weights_path)

layer = new_model.layers[5]
weights, biases = layer.get_weights()
mask_weight = np.ones_like(weights)
mask_bias = np.ones_like(biases)
acc = 0
prune2_acc = 0
prune4_acc = 0
prune10_acc = 0
data = np.zeros((60,3))

for c,i in enumerate(index):
    mask_weight[:, :, :, i] = 0
    mask_bias[i] = 0
    # Apply the mask to the weights
    new_weights = weights * mask_weight
    new_bias = biases * mask_bias
    # Set the new weights in the model
    layer.set_weights([new_weights, new_bias])
    cl_label_p = np.argmax(new_model.predict(cl_x_test), axis=1)
    acc = np.mean(np.equal(cl_label_p, cl_y_test))*100
    

    if no_prune_acc - acc >=2 and no_prune_acc - acc < 4 and acc > prune2_acc:
      prune2_acc = acc
      new_model_2 = clone_model(new_model)
      new_model_2.set_weights(new_model.get_weights())
      print('Channels pruned to save 2% model:',c+1)
    if no_prune_acc - acc >=4 and no_prune_acc - acc < 10 and acc > prune4_acc:
      prune4_acc = acc
      new_model_4 = clone_model(new_model)
      new_model_4.set_weights(new_model.get_weights())
      print('Channels pruned to save 4% model:',c+1)
    if no_prune_acc - acc >=10 and acc > prune10_acc:
      prune10_acc = acc
      new_model_10 = clone_model(new_model)
      new_model_10.set_weights(new_model.get_weights())
      print('Channels pruned to save 10% model:',c+1)


    bd_label_p = np.argmax(new_model.predict(bd_x_test), axis=1)
    asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
    print('Classification Accuracy with',c+1,'channels pruned:', acc)
    print('Attack Success Rate with',c+1,'channels pruned:', asr)
    data[c,0] = c+1; data[c,1]= asr; data[c,2]= acc

data_df = pd.DataFrame(data, columns=['Channels Pruned', 'Attack Success rate', 'Classification Accuracy'])
data_df.to_csv('data.csv')
    
class G_model (keras.Model):   
    def __init__ (self,model1,model2, **kwargs):  
        super ().__init__ (**kwargs)  
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        preds1 = np.argmax(self.model1.predict(X),axis=1)
        preds2 = np.argmax(self.model2.predict(X),axis=1)
        for i in range(len(preds1)):
            if preds1[i] != preds2[i]:
               preds1[i]=1283
        return preds1


cl_label_p = G_model(new_model_2,bd_model).predict(cl_x_test)
clean_accuracy_G2 = np.mean(np.equal(cl_label_p, cl_y_test))*100
print('Clean Classification accuracy of 2% Model:', clean_accuracy_G2)

bd_label_p = G_model(new_model_2,bd_model).predict(bd_x_test)
asr_G2 = np.mean(np.equal(bd_label_p, bd_y_test))*100
print('Attack Success Rate of 2% Model:', asr_G2)

cl_label_p = G_model(new_model_4,bd_model).predict(cl_x_test)
clean_accuracy_G4 = np.mean(np.equal(cl_label_p, cl_y_test))*100
print('Clean Classification accuracy of 4% Model:', clean_accuracy_G4)

bd_label_p = G_model(new_model_4,bd_model).predict(bd_x_test)
asr_G4 = np.mean(np.equal(bd_label_p, bd_y_test))*100
print('Attack Success Rate of 4% Model:', asr_G4)

cl_label_p = G_model(new_model_10,bd_model).predict(cl_x_test)
clean_accuracy_G10 = np.mean(np.equal(cl_label_p, cl_y_test))*100
print('Clean Classification accuracy of 10% Model:', clean_accuracy_G10)

bd_label_p = G_model(new_model_10,bd_model).predict(bd_x_test)
asr_G10 = np.mean(np.equal(bd_label_p, bd_y_test))*100
print('Attack Success Rate of 10% Model:', asr_G10)

d = {'Attack Success Rate':
 [asr_G2,asr_G4,asr_G10],
  'Recall Score':
   [clean_accuracy_G2,clean_accuracy_G4,clean_accuracy_G10]}

G_score_df = pd.DataFrame(data=d)
G_score_df.index = ['G_Model(2%)','G_Model(4%)','G_Model(10%)']
G_score_df.to_csv('G_score.csv')
G_model(new_model_2,bd_model).save('G_model2.h5')
G_model(new_model_4,bd_model).save('G_model4.h5')
G_model(new_model_10,bd_model).save('G_model10.h5')