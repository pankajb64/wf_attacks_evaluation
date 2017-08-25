
# coding: utf-8

# In[98]:

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D
from keras.utils.np_utils import to_categorical
#from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
#from keras.optimizers import SGD
import numpy as np
import os
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import re
import sys
import datetime

# In[14]:

def get_shuffled( max_feat_count, x_0_tr, lab_0_tr, x_1_tr, lab_1_tr, x_0_ts, lab_0_ts, x_1_ts, lab_1_ts):
    x_tr = np.vstack([x_0_tr, x_1_tr])
    print x_tr.shape
    x_ts = np.vstack([x_0_ts, x_1_ts])
    np.random.shuffle(x_tr)
    np.random.shuffle(x_ts)
    X_tr = x_tr[:, 0:max_feat_count]
    y_tr = x_tr[:, max_feat_count:]
    X_ts = x_ts[:, 0:max_feat_count]
    y_ts = x_ts[:, max_feat_count:]
    print X_tr.shape
    print y_tr.shape
    print X_ts.shape
    print y_ts.shape
    return X_tr, y_tr, X_ts, y_ts

def get_instances_multitab(num_test, feats_0_list, feats_1_list):
    feats_0_list.extend(feats_0_list[0:1000-len(feats_0_list)])
    feats_1_list.extend(feats_1_list[0:1000-len(feats_1_list)])
    f0 = np.array(feats_0_list)
    f1 = np.array(feats_1_list)
    print f0.shape, f1.shape
    np.random.shuffle(f0)
    np.random.shuffle(f1)
    f0_tr = f0[:-num_test, :]; f0_ts = f0[-num_test:, :]
    f1_tr = f1[:-num_test, :]; f1_ts = f1[:num_test:, :]
    
    return f0_tr, f1_tr, f0_ts, f1_ts

def get_shuffled_multitab(max_feat_count, f0_tr, f1_tr, f0_ts, f1_ts):
    return get_shuffled(max_feat_count, f0_tr, None, f1_tr, None, f0_ts, None, f1_ts, None)

def get_feats_multitab(feat_path, labels_path, max_feat_count):
    feats_0_list = []
    feats_1_list = []
    lab_dict = {}
    
    for fil in os.listdir(labels_path):
        labs_fil = []
        lines = []
        with open(os.path.join(labels_path, fil)) as f:
            lines = f.readlines()
        headers = lines[0].split(",")
        crawl_pattern=r"(.+)_labels\.csv"
	mch = re.match(crawl_pattern, fil)
	crawl = mch.group(1)
	dict_line = {}
        for i in xrange(1, len(lines)):
            fields = lines[i].split(",")
            batch = fields[0]
            site = fields[1]
            label = fields[3]
            keyname = "%s_%s_%s"%(crawl, batch, site)
            lab_dict[keyname] = label
	#print lab_dict["%s_0_google.com"%crawl]
	#print lab_dict["%s_1_google.com"%crawl]            
    #print lab_dict        
    for fil in os.listdir(feat_path):
        feats = []
        with open(os.path.join(feat_path, fil)) as f:
            feats = f.readlines()
            feats = [float(line) for line in feats]    
        feats = np.array(feats)
        feats1 = np.zeros(max_feat_count)
        length = min(len(feats), max_feat_count)
        feats1[:length] = feats[:length]
        feats1 = feats1.tolist()
        #print len(feats)
        labels = []
        pattern = r"(crawl\d+_\d+)_\d+-([^_]+)_(\d)_.*\.feat"
        #Get label based on crawl name, batch number and main site name
        mch = re.match(pattern, fil)
        crawl = mch.group(1)
        site  = mch.group(2)
        batch = mch.group(3)
        keyname = "%s_%s_%s"%(crawl, batch, site)
        label = lab_dict[keyname]
        feats1.append(label)
        feats1 = np.array(feats1, dtype='float32')
        
        if int(label) == 0:
            feats_0_list.append(feats1)
        else:
            feats_1_list.append(feats1)
            
    return feats_0_list, feats_1_list  


# In[62]:

def ConvNet(max_len, weights_path=None):
    model = Sequential()
    model.add(Conv1D(4, (2,), activation='elu', input_shape=(max_len,1)))
    model.add(Conv1D(8, (2, ), activation='elu'))
    model.add(Conv1D(16, (2, ), activation='elu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(32, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, activation='softmax'))    
    
    if weights_path:
        model.load_weights(weights_path)

    return model

# In[29]:

def get_model(max_feat_count, vgg = False):
    if not vgg:
        model = Sequential()        
        model.add(Dense(2, input_dim=max_feat_count, activation='elu'))
        model.add(Dense(11, activation='softmax' ))
    else:
        model = ConvNet(max_feat_count)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']  )
    return model


# In[47]:

def reshape(X_tr, X_ts, vgg = False):
    if vgg:
        X_tr = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
        X_ts = X_ts.reshape(X_ts.shape[0], X_ts.shape[1], 1)
    return X_tr, X_ts


# In[68]:

def run(feat_path, labels_path, max_feat_count, num_test, vgg, batch_size, epochs, per_epoch):
    X_tr, y_tr, X_ts, y_ts = get_shuffled_multitab( max_feat_count, *get_instances_multitab( num_test, *get_feats_multitab(feat_path, labels_path, max_feat_count) ) )
    y_tr_binary = to_categorical(y_tr, num_classes=11)
    y_ts_binary = to_categorical(y_ts, num_classes=11)
    X_tr, X_ts = reshape(X_tr, X_ts, vgg)
    model = get_model(max_feat_count, vgg)
    ep  = 1 if per_epoch else epochs
    if per_epoch:
        print('%d more epochs to go'%epochs)
    
    h_tr = model.fit(X_tr, y_tr_binary, batch_size=batch_size, epochs=ep)
    l_tr = h_tr.history['loss'][-1]
    a_tr = h_tr.history['acc'][-1]
    print l_tr
    print a_tr
    h_ts = model.evaluate(X_ts, y_ts_binary, batch_size=X_ts.shape[0])
    l_ts = h_ts[0]
    a_ts = h_ts[1]
    print l_ts
    print a_ts
    print model.predict(X_ts)
    if per_epoch:
        if epochs > 1:
            newl_tr, newa_tr, newl_ts, newa_ts = run(feat_path, labels_path, max_feat_count, num_test, vgg, batch_size, epochs-1, per_epoch)
            l_tr = [l_tr] + newl_tr; a_tr = [a_tr] + newa_tr
            l_ts = [l_ts] + newl_ts; a_ts = [a_ts] + newa_ts
        else:
            l_tr = [l_tr]; a_tr = [a_tr]; 
            l_ts = [l_ts]; a_ts = [a_ts]
    return l_tr, a_tr, l_ts, a_ts


# In[69]:

def batch(feat_path, labels_path, num_test, start_feat, end_feat, step_feat=100, vgg=False, batch_size=32, epochs=2, per_epoch=False):
    tr_loss = []; tr_acc = []; ts_loss = []; ts_acc = []
    for feat in xrange(start_feat, end_feat, step_feat):
        print "Starting run - max_feat_count = %d"%feat
        l_tr, a_tr, l_ts, a_ts = run(feat_path, labels_path, feat, num_test, vgg, batch_size, epochs, per_epoch)
        tr_loss.append(l_tr)
        tr_acc.append(a_tr)
        ts_loss.append(l_ts)
        ts_acc.append(a_ts)
    return tr_loss, tr_acc, ts_loss, ts_acc    


# In[101]:
'''
Edit these values to run different versions
'''
feat_path='data/mta3/feat/'
labels_path='data/mta3/labels/'
num_test=100
start_feat = 500
end_feat = 10500
step_feat = 500
vgg = (int(sys.argv[1]) == 1)
print "vgg %s"%vgg
batch_size=32
epochs=20
per_epoch=True
tr_loss, tr_acc, ts_loss, ts_acc = batch(feat_path, labels_path, num_test, start_feat, end_feat, step_feat, vgg, batch_size, epochs, per_epoch)


# In[104]:

save_dict = {'tr_loss': tr_loss, 'tr_acc': tr_acc, 'ts_loss' : ts_loss, 'ts_acc' : ts_acc }
timestamp=datetime.datetime.now().strftime("%Y%d%m%H%M%S")
print timestamp
pkl_file_name = 'results/mta_%s_%s_%s_%s_%s_%s_%s_%s.pkl'%(timestamp, start_feat, end_feat, step_feat, vgg, batch_size, epochs, per_epoch)
with open( pkl_file_name, 'wb') as f:
    pkl.dump(save_dict, f)
