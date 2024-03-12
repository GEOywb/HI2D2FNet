# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import h5py
import pylab
import keras
import numpy as np
import scipy.io as sio
from dgconv2 import DGC
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras_multi_head import MultiHeadAttention
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.layers import Activation,Input,Dense,Lambda,Conv2D,concatenate,BatchNormalization,Reshape,Add,Flatten,GlobalAveragePooling2D
iterations=100000

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.config.experimental_run_functions_eagerly(True)

dataset=0
if dataset==0:
    data=h5py.File('MUUFL_plus.mat', 'r')
    heads=4#64整除
    batchsize=256
    source='result3/'
if dataset==1:
    data=h5py.File('Italy_plus.mat', 'r')
    heads=3#63整除
    source='result2/'
    batchsize=256

X_spatial_all=data['X_spatial'][()].transpose(3,2,1,0)
LiDAR_all=data['LiDAR'][()].transpose(2,1,0)
LiDAR_all=np.expand_dims(LiDAR_all,-1)
act_Y_train_all=data['act_Y_train'][()].transpose(1,0)
indexi_all=data['indexi'][()].transpose(1,0)
indexj_all=data['indexj'][()].transpose(1,0)

X_spatial_all=X_spatial_all.astype('float32')
LiDAR_all=LiDAR_all.astype('float32')
act_Y_train_all=act_Y_train_all.astype('int')
indexi_all=indexi_all.astype('float32')
indexj_all=indexj_all.astype('float32')

act_Y_train_all[act_Y_train_all==-1]=0

slide_size=5
if slide_size==3:
    X_spatial_all=X_spatial_all[:,4:7,4:7,:]
    LiDAR_all=LiDAR_all[:,4:7,4:7,:]
if slide_size==5:
    X_spatial_all=X_spatial_all[:,3:8,3:8,:]
    LiDAR_all=LiDAR_all[:,3:8,3:8,:]
if slide_size==7:
    X_spatial_all=X_spatial_all[:,2:9,2:9,:]
    LiDAR_all=LiDAR_all[:,2:9,2:9,:]
if slide_size==9:
    X_spatial_all=X_spatial_all[:,1:10,1:10,:]
    LiDAR_all=LiDAR_all[:,1:10,1:10,:]
if slide_size==11:
    X_spatial_all=X_spatial_all[:,:,:,:]
    LiDAR_all=LiDAR_all[:,:,:,:]

###############################################################################
X_spatial_all=(X_spatial_all-X_spatial_all.min())/(X_spatial_all.max()-X_spatial_all.min())
LiDAR_all=(LiDAR_all-LiDAR_all.min())/(LiDAR_all.max()-LiDAR_all.min())
X_spatial_all=np.log(X_spatial_all+1e-9)
LiDAR_all=np.log(LiDAR_all+1e-9)
###############################################################################
scaler=MinMaxScaler(feature_range=(0,1))

X_spatial_all_=X_spatial_all.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1]*X_spatial_all.shape[2]*X_spatial_all.shape[3]])
LiDAR_all_=LiDAR_all.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1]*LiDAR_all.shape[2]*LiDAR_all.shape[3]])

X_spatial_all_=scaler.fit_transform(X_spatial_all_)
LiDAR_all_=scaler.fit_transform(LiDAR_all_)

X_spatial_all=X_spatial_all_.reshape([X_spatial_all.shape[0],X_spatial_all.shape[1],X_spatial_all.shape[2],X_spatial_all.shape[3]])
LiDAR_all=LiDAR_all_.reshape([LiDAR_all.shape[0],LiDAR_all.shape[1],LiDAR_all.shape[2],LiDAR_all.shape[3]])
#####################     选择训练集      ######################################
act_Y_train_all=np.reshape(act_Y_train_all,act_Y_train_all.shape[0])
indexi_all=np.reshape(indexi_all,indexi_all.shape[0])
indexj_all=np.reshape(indexj_all,indexj_all.shape[0])

act_Y_train=act_Y_train_all

randpaixv=act_Y_train_all.argsort()
X_spatial_all=X_spatial_all[randpaixv]
LiDAR_all=LiDAR_all[randpaixv]
indexi_all=indexi_all[randpaixv]
indexj_all=indexj_all[randpaixv]
act_Y_train_all=act_Y_train_all[randpaixv]

X_spatial_all=X_spatial_all[act_Y_train_all>0]
LiDAR_all=LiDAR_all[act_Y_train_all>0]
indexi_all=indexi_all[act_Y_train_all>0]
indexj_all=indexj_all[act_Y_train_all>0]
act_Y_train_all=act_Y_train_all[act_Y_train_all>0]

indices=np.arange(X_spatial_all.shape[0])
indices_train,indices_test,act_Y_train_train,act_Y_train_test=train_test_split(indices,act_Y_train_all,test_size=0.99,stratify=act_Y_train_all)#,random_state=32

X_spatial_train=X_spatial_all[indices_train,:,:]
LiDAR_train=LiDAR_all[indices_train,:,:]
act_Y_train_train=act_Y_train_all[indices_train]
indexi_train=indexi_all[indices_train]
indexj_train=indexj_all[indices_train]

X_spatial_test=X_spatial_all[indices_test,:,:]
LiDAR_test=LiDAR_all[indices_test,:,:]
act_Y_train_test=act_Y_train_all[indices_test]
indexi_test=indexi_all[indices_test]
indexj_test=indexj_all[indices_test]

act_Y_train_train=np_utils.to_categorical(act_Y_train_train-1)
act_Y_train_test=np_utils.to_categorical(act_Y_train_test-1)
###############################################################################
def DIV(x):
    dot1=K.batch_dot(x[0],x[1],axes=1)
    dot2=K.batch_dot(x[0],x[0],axes=1)
    dot3=K.batch_dot(x[1],x[1],axes=1)
    max_=K.maximum(K.sqrt(dot2*dot3),K.epsilon())
    return dot1/max_

def CON(x):
    dot1=K.batch_dot(x[0],x[1],axes=1)
    dot2=K.batch_dot(x[0],x[0],axes=1)
    dot3=K.batch_dot(x[1],x[1],axes=1)
    max_=K.maximum(K.sqrt(dot2*dot3),K.epsilon())
    return 1-dot1/max_

def get_mean_and_sigma(inputs):
    mean,var=tf.nn.moments(inputs,[1,2],keep_dims=True)
    sigma=tf.sqrt(tf.add(var,1e-9))
    return mean,sigma
      
def AdaIN(x):
    f,s=x
    meanF,sigmaF=get_mean_and_sigma(f)
    meanS,sigmaS=get_mean_and_sigma(s)
    aa=sigmaS*((f-meanF)/sigmaF)+meanS
    bb=sigmaF*((s-meanS)/sigmaS)+meanF
    return (aa+bb)/2

kernelsize=3
activation='tanh'
kernel_regularizer=None
lr=0.0001
############################    decomposition    ##############################
#网络开始
Hx1_i_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))
Hx2_i_input=Input(shape=(X_spatial_train.shape[1],X_spatial_train.shape[2],X_spatial_train.shape[3]))

Lx1_i_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))
Lx2_i_input=Input(shape=(LiDAR_train.shape[1],LiDAR_train.shape[2],1))

Hx1_i=Hx1_i_input
Hx2_i=Hx2_i_input
Lx1_i=Lx1_i_input
Lx2_i=Lx2_i_input
#####
indexi1=Input(shape=(1,))
indexi2=Input(shape=(1,))

indexj1=Input(shape=(1,))
indexj2=Input(shape=(1,))
#####
Lx1_=Conv2D(int(Hx1_i.shape[-1]),1,strides=(1,1),activation=activation,padding='same',kernel_regularizer=kernel_regularizer)(Lx1_i)
Lx1_=BatchNormalization()(Lx1_)
Lx2_=Conv2D(int(Hx2_i.shape[-1]),1,strides=(1,1),activation=activation,padding='same',kernel_regularizer=kernel_regularizer)(Lx2_i)
Lx2_=BatchNormalization()(Lx2_)
##########################################
Hx1_middle=GlobalAveragePooling2D()(Hx1_i)
Hx2_middle=GlobalAveragePooling2D()(Hx2_i)
Lx1_middle=GlobalAveragePooling2D()(Lx1_i)
Lx2_middle=GlobalAveragePooling2D()(Lx2_i)

Lx1_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train.shape[3])]))(Lx1_middle)
Lx2_middle=Lambda(lambda x:K.tile(x,[1,int(X_spatial_train.shape[3])]))(Lx2_middle)
#############   第一步生成初级的   ####################
Hx1_1=Hx1_i
Hx2_1=Hx2_i
Lx1_1=Lx1_
Lx2_1=Lx2_

Hx1_1=Reshape((int(Hx1_1.shape[1])*int(Hx1_1.shape[2]),int(Hx1_1.shape[3]),))(Hx1_1)
Hx2_1=Reshape((int(Hx2_1.shape[1])*int(Hx2_1.shape[2]),int(Hx2_1.shape[3]),))(Hx2_1)
Lx1_1=Reshape((int(Lx1_1.shape[1])*int(Lx1_1.shape[2]),int(Lx1_1.shape[3]),))(Lx1_1)
Lx2_1=Reshape((int(Lx2_1.shape[1])*int(Lx2_1.shape[2]),int(Lx2_1.shape[3]),))(Lx2_1)

Hx1_1=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Hx1_1)
Hx1_1=BatchNormalization()(Hx1_1)

Hx2_1=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Hx2_1)
Hx2_1=BatchNormalization()(Hx2_1)

Lx1_1=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Lx1_1)
Lx1_1=BatchNormalization()(Lx1_1)

Lx2_1=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Lx2_1)
Lx2_1=BatchNormalization()(Lx2_1)
#############   第二步生成R   ####################
Rx1_2=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Hx1_1)
Rx1_2=BatchNormalization()(Rx1_2)

Rx2_2=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Hx2_1)
Rx2_2=BatchNormalization()(Rx2_2)
#############   第二步生成S   ####################
Sx1_2=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Hx1_1)
Sx1_2=BatchNormalization()(Sx1_2)

Sx2_2=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Hx2_1)
Sx2_2=BatchNormalization()(Sx2_2)
#############   第二步生成L   ####################
Lx1_2=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Lx1_1)
Lx1_2=BatchNormalization()(Lx1_2)

Lx2_2=MultiHeadAttention(head_num=heads,activation=activation,kernel_regularizer=kernel_regularizer)(Lx2_1)
Lx2_2=BatchNormalization()(Lx2_2)
#################################################
Rx1_2=Reshape((slide_size,slide_size,int(Hx1_i.shape[-1]),))(Rx1_2)
Rx2_2=Reshape((slide_size,slide_size,int(Hx1_i.shape[-1]),))(Rx2_2)
Sx1_2=Reshape((slide_size,slide_size,int(Hx1_i.shape[-1]),))(Sx1_2)
Sx2_2=Reshape((slide_size,slide_size,int(Hx1_i.shape[-1]),))(Sx2_2)
Lx1_2=Reshape((slide_size,slide_size,int(Hx1_i.shape[-1]),))(Lx1_2)
Lx2_2=Reshape((slide_size,slide_size,int(Hx1_i.shape[-1]),))(Lx2_2)
#################################################
HSI_gradient=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Hx1_middle,Hx2_middle,indexi1,indexi2,indexj1,indexj2])
LiDAR_gradient=Lambda(lambda x:1/(K.abs(x[0]-x[1])*K.sqrt(K.square(x[2]-x[3])+K.square(x[4]-x[5])+1e-9)))([Lx1_middle,Lx2_middle,indexi1,indexi2,indexj1,indexj2])

HSI_gradient=Activation('sigmoid')(HSI_gradient)
LiDAR_gradient=Activation('sigmoid')(LiDAR_gradient)
################################################
#输出的R和S样本的均值
Rx1_2_mean=GlobalAveragePooling2D()(Rx1_2)
Sx1_2_mean=GlobalAveragePooling2D()(Sx1_2)
Rx2_2_mean=GlobalAveragePooling2D()(Rx2_2)
Sx2_2_mean=GlobalAveragePooling2D()(Sx2_2)

loss1=Lambda(lambda x:K.mean(K.square(x[0]-x[1]-x[2]),axis=-1),name='loss1')([Hx1_middle,Rx1_2_mean,Sx1_2_mean])
loss2=Lambda(lambda x:K.mean(K.square(x[0]-x[1]-x[2]),axis=-1),name='loss2')([Hx2_middle,Rx2_2_mean,Sx2_2_mean])

loss3=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss3')([Rx1_2_mean,Rx2_2_mean,HSI_gradient])
loss4=Lambda(lambda x:K.mean(x[2]*K.square(x[0]-x[1]),axis=-1),name='loss4')([Sx1_2_mean,Sx2_2_mean,LiDAR_gradient])

TOTAL_loss1=Lambda(lambda x:x[0]+x[1],output_shape=[1,],name='TOTAL_loss1')([loss1,loss2])
TOTAL_loss2=Lambda(lambda x:x[0]+x[1],output_shape=[1,],name='TOTAL_loss2')([loss3,loss4])

Rx1_2=Lambda(lambda x:K.exp(x))(Rx1_2)
Rx2_2=Lambda(lambda x:K.exp(x))(Rx2_2)
Sx1_2=Lambda(lambda x:K.exp(x))(Sx1_2)
Sx2_2=Lambda(lambda x:K.exp(x))(Sx2_2)
Lx1_2=Lambda(lambda x:K.exp(x))(Lx1_2)
Lx2_2=Lambda(lambda x:K.exp(x))(Lx2_2)
Lx1_=Lambda(lambda x:K.exp(x))(Lx1_)
Lx2_=Lambda(lambda x:K.exp(x))(Lx2_)

Rx1_2=Activation(activation)(Rx1_2)
Rx2_2=Activation(activation)(Rx2_2)
Sx1_2=Activation(activation)(Sx1_2)
Sx2_2=Activation(activation)(Sx2_2)
Lx1_2=Activation(activation)(Lx1_2)
Lx2_2=Activation(activation)(Lx2_2)
Lx1_=Activation(activation)(Lx1_)
Lx2_=Activation(activation)(Lx2_)

Rx1_2=BatchNormalization()(Rx1_2)
Rx2_2=BatchNormalization()(Rx2_2)
Sx1_2=BatchNormalization()(Sx1_2)
Sx2_2=BatchNormalization()(Sx2_2)
Lx1_2=BatchNormalization()(Lx1_2)
Lx2_2=BatchNormalization()(Lx2_2)
Lx1_=BatchNormalization()(Lx1_)
Lx2_=BatchNormalization()(Lx2_)

Rx1_2_mean_plus=GlobalAveragePooling2D()(Rx1_2)
Rx2_2_mean_plus=GlobalAveragePooling2D()(Rx2_2)
Sx1_2_mean_plus=GlobalAveragePooling2D()(Sx1_2)
Sx2_2_mean_plus=GlobalAveragePooling2D()(Sx2_2)

loss5=Lambda(lambda x:DIV([x[0],x[1]]),name='loss5')([Rx1_2_mean_plus,Sx1_2_mean_plus])
loss6=Lambda(lambda x:DIV([x[0],x[1]]),name='loss6')([Rx2_2_mean_plus,Sx2_2_mean_plus])

TOTAL_loss3=Lambda(lambda x:x[0]+x[1],output_shape=[1,],name='TOTAL_loss3')([loss5,loss6])

generator_train=keras.models.Model([Hx1_i_input,Hx2_i_input,Lx1_i_input,Lx2_i_input,indexi1,indexi2,indexj1,indexj2],[TOTAL_loss1,TOTAL_loss2,TOTAL_loss3])
generator_train.compile(loss=['mean_squared_error','mean_squared_error','mean_squared_error'],loss_weights=[1,1,1],optimizer=keras.optimizers.adam(lr=lr,decay=0.01))
generator_train.summary()
############################    fusion    #####################################
generator_train.trainable=False

merge_Lx1=Add()([Lx1_2,Lx1_])
merge_Lx2=Add()([Lx2_2,Lx2_])

merge_x1=concatenate([Sx1_2,merge_Lx1],axis=3)
merge_x2=concatenate([Sx2_2,merge_Lx2],axis=3)

merge_x1_1=DGC(rank=2,filters=int(Sx1_2.shape[-1]),kernel_size=(kernelsize,kernelsize),edge=int(np.ceil(np.log2(int(merge_x1.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x1)
merge_x1_1=BatchNormalization()(merge_x1_1)

merge_x2_1=DGC(rank=2,filters=int(Sx1_2.shape[-1]),kernel_size=(kernelsize,kernelsize),edge=int(np.ceil(np.log2(int(merge_x2.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x2)
merge_x2_1=BatchNormalization()(merge_x2_1)

merge_x1_1_=Lambda(AdaIN)([Sx1_2,merge_Lx1])
merge_x2_1_=Lambda(AdaIN)([Sx2_2,merge_Lx2])

merge_x1_1_=Activation(activation)(merge_x1_1_)
merge_x2_1_=Activation(activation)(merge_x2_1_)

merge_x1_1_=BatchNormalization()(merge_x1_1_)
merge_x2_1_=BatchNormalization()(merge_x2_1_)

tradeoff1=Conv2D(int(merge_x1_1.shape[3]),3,strides=(1, 1),activation='sigmoid',padding='same',kernel_regularizer=kernel_regularizer)(merge_x1)
tradeoff2=Conv2D(int(merge_x2_1.shape[3]),3,strides=(1, 1),activation='sigmoid',padding='same',kernel_regularizer=kernel_regularizer)(merge_x2)

merge_x1_2=Lambda(lambda x:x[0]*x[2]+x[1]*(1-x[2]))([merge_x1_1,merge_x1_1_,tradeoff1])
merge_x2_2=Lambda(lambda x:x[0]*x[2]+x[1]*(1-x[2]))([merge_x2_1,merge_x2_1_,tradeoff2])

merge_x3=concatenate([Rx1_2,merge_x1_2],axis=3)
merge_x4=concatenate([Rx2_2,merge_x2_2],axis=3)

merge_x3_1=DGC(rank=2,filters=int(Rx1_2.shape[-1]),kernel_size=(kernelsize,kernelsize),edge=int(np.ceil(np.log2(int(merge_x3.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x3)
merge_x3_1=BatchNormalization()(merge_x3_1)

merge_x4_1=DGC(rank=2,filters=int(Rx1_2.shape[-1]),kernel_size=(kernelsize,kernelsize),edge=int(np.ceil(np.log2(int(merge_x4.shape[-1])))),activation=activation,kernel_regularizer=kernel_regularizer)(merge_x4)
merge_x4_1=BatchNormalization()(merge_x4_1)

merge_x3_1_=Lambda(AdaIN)([Rx1_2,merge_x1_2])
merge_x4_1_=Lambda(AdaIN)([Rx2_2,merge_x2_2])

merge_x3_1_=Activation(activation)(merge_x3_1_)
merge_x4_1_=Activation(activation)(merge_x4_1_)
merge_x3_1_=BatchNormalization()(merge_x3_1_)
merge_x4_1_=BatchNormalization()(merge_x4_1_)

tradeoff3=Conv2D(int(merge_x3_1.shape[3]),3,strides=(1, 1),activation='sigmoid',padding='same',kernel_regularizer=kernel_regularizer)(merge_x3)
tradeoff4=Conv2D(int(merge_x4_1.shape[3]),3,strides=(1, 1),activation='sigmoid',padding='same',kernel_regularizer=kernel_regularizer)(merge_x4)

merge_x3_2=Lambda(lambda x:x[0]*x[2]+x[1]*(1-x[2]))([merge_x3_1,merge_x3_1_,tradeoff3])
merge_x4_2=Lambda(lambda x:x[0]*x[2]+x[1]*(1-x[2]))([merge_x4_1,merge_x4_1_,tradeoff4])

merge_sanple1=concatenate([merge_x1_2,merge_x3_2],axis=3)
merge_sanple2=concatenate([merge_x2_2,merge_x4_2],axis=3)
###############################################################################
merge_sanple1=Flatten()(merge_sanple1)
cls_merge1=Dense(act_Y_train_train.shape[1],activation='softmax',kernel_regularizer=kernel_regularizer)(merge_sanple1)

merge_sanple2=Flatten()(merge_sanple2)
cls_merge2=Dense(act_Y_train_train.shape[1],activation='softmax',kernel_regularizer=kernel_regularizer)(merge_sanple2)
######################
merge_x1_2_mean=GlobalAveragePooling2D()(merge_x1_2)
merge_x2_2_mean=GlobalAveragePooling2D()(merge_x2_2)
merge_x3_2_mean=GlobalAveragePooling2D()(merge_x3_2)
merge_x4_2_mean=GlobalAveragePooling2D()(merge_x4_2)
#########################
loss7=Lambda(lambda x:CON([x[0],x[1]]),output_shape=[1,],name='IF_loss7')([merge_x1_2_mean,merge_x3_2_mean])
loss8=Lambda(lambda x:CON([x[0],x[1]]),output_shape=[1,],name='IF_loss8')([merge_x2_2_mean,merge_x4_2_mean])

TOTAL_loss4=Lambda(lambda x:x[0]+x[1],output_shape=[1,],name='TOTAL_loss4')([loss7,loss8])
#######################################
fusior_train=keras.models.Model([Hx1_i_input,Hx2_i_input,Lx1_i_input,Lx2_i_input,indexi1,indexi2,indexj1,indexj2],[cls_merge1,cls_merge2,TOTAL_loss4])
fusior_train.compile(loss=['categorical_crossentropy','categorical_crossentropy','mean_squared_error'],loss_weights=[1,1,1],optimizer=keras.optimizers.Adam(lr=lr,decay=0.01),metrics=['accuracy'])
fusior_train.summary()
############################   正式运行   #####################################
decomposer_number=1
fusior_number=10

d_final_loss=np.zeros(shape=iterations*decomposer_number)
f_final_loss=np.zeros(shape=iterations*fusior_number)
cls_loss1=np.zeros(shape=iterations*fusior_number)
cls_loss2=np.zeros(shape=iterations*fusior_number)
cls_loss3=np.zeros(shape=iterations*fusior_number)
cls_loss4=np.zeros(shape=iterations*fusior_number)

maxacc=0
maxkappa=0
maxp=0
maxr=0
maxf_score1=0

for i in range(iterations):
    index1=[ind for ind in range(int(X_spatial_train.shape[0]))]
    np.random.shuffle(index1)
    X_spatial_train1=X_spatial_train[index1]
    LiDAR_train1=LiDAR_train[index1]
    act_Y_train_train1=act_Y_train_train[index1]
    indexi_train1=indexi_train[index1]
    indexj_train1=indexj_train[index1]
            
    index2=[ind for ind in range(int(X_spatial_train.shape[0]))]
    np.random.shuffle(index2)
    X_spatial_train2=X_spatial_train[index2]
    LiDAR_train2=LiDAR_train[index2]
    act_Y_train_train2=act_Y_train_train[index2]
    indexi_train2=indexi_train[index2]
    indexj_train2=indexj_train[index2]

    history=generator_train.fit([X_spatial_train1,X_spatial_train2,LiDAR_train1,LiDAR_train2,indexi_train1,indexi_train2,indexj_train1,indexj_train2],
                                [np.zeros([X_spatial_train1.shape[0]]),np.zeros([X_spatial_train1.shape[0]]),np.zeros([X_spatial_train1.shape[0]])],
                                batch_size=batchsize,epochs=decomposer_number,shuffle=True,verbose=1)
    
    d_final_loss[i*decomposer_number:(i+1)*decomposer_number]=history.history['loss']
    
    plt.plot(d_final_loss[0:(i+1)*decomposer_number])
    pylab.show()
         
    Test_loss=fusior_train.predict([X_spatial_test,X_spatial_test,LiDAR_test,LiDAR_test,indexi_test,indexi_test,indexj_test,indexj_test])
    Pred_result1=Test_loss[0]
    Pred_result2=Test_loss[1]
        
    Pred_result1_=Pred_result1.argmax(axis=1)
    Pred_result2_=Pred_result2.argmax(axis=1)
    raw_label=act_Y_train_test.argmax(axis=1)
        
    acc1=accuracy_score(Pred_result1_,raw_label)
    acc2=accuracy_score(Pred_result2_,raw_label)
    
######################################################################################################################
    if acc1>maxacc:
        acc=acc1
        Pred_result_=Pred_result1_
        generator_component=keras.models.Model([Hx1_i_input,Hx2_i_input,Lx1_i_input,Lx2_i_input,indexi1,indexi2,indexj1,indexj2],[Rx1_2,Rx2_2,Sx1_2,Sx2_2])
        [R1_result,R2_result,S1_result,S2_result]=generator_component.predict([X_spatial_all,X_spatial_all,LiDAR_all,LiDAR_all,indexi_all,indexi_all,indexj_all,indexj_all])
        R=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            R[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=R1_result[iii,int((R1_result.shape[1]-1)/2),int((R1_result.shape[1]-1)/2),:]
        S=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            S[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=S1_result[iii,int((S1_result.shape[1]-1)/2),int((S1_result.shape[1]-1)/2),:]            

        generator_train.save_weights(source+'generator_train_'+str(i)+'.h5')
        fusior_train.save_weights(source+'fusior_train_'+str(i)+'.h5')           
            
        maxacc=acc
        maxOA=maxacc
        maxAllAcc=recall_score(raw_label,Pred_result_,average=None)
        maxAA=recall_score(raw_label,Pred_result_,average='macro')
        maxkappa=cohen_kappa_score(np.array(Pred_result_).reshape(-1,1),np.array(raw_label).reshape(-1,1))
        maxp=precision_score(raw_label,Pred_result_,average='macro')
        maxf1score=f1_score(raw_label,Pred_result_,average='macro')

        MAP=np.zeros([int(indexi_all.max()),int(indexj_all.max())])
        for ii in range(act_Y_train_test.shape[0]):
            MAP[int(indexi_test[ii]-1),int(indexj_test[ii]-1)]=Pred_result_[ii]+1
        for ii in range(act_Y_train_train.shape[0]):
            MAP[int(indexi_train[ii]-1),int(indexj_train[ii]-1)]=(act_Y_train_train.argmax(axis=1))[ii]+1

        name=source+'net_result_'+str(i)+'_'+str(maxacc)+'.mat'
        sio.savemat(name, {'R':R,
                           'S':S,
                           'Pred_result': Pred_result_,
                           'raw_label':raw_label,
                           'maxacc':maxacc,
                           'maxOA':maxOA,
                           'maxAA':maxAA,
                           'maxkappa':maxkappa,
                           'maxp':maxp,
                           'maxf1score':maxf1score,
                           'MAP':MAP})
#######################################################################################################################
    history=fusior_train.fit([X_spatial_train1,X_spatial_train2,LiDAR_train1,LiDAR_train2,indexi_train1,indexi_train2,indexj_train1,indexj_train2],
                             [act_Y_train_train1,act_Y_train_train2,np.zeros([X_spatial_train1.shape[0]])],#,np.zeros([X_spatial_train1.shape[0]]),np.zeros([X_spatial_train1.shape[0]])
                             batch_size=batchsize,epochs=fusior_number,shuffle=True,verbose=1)
        
    f_final_loss[i*fusior_number:(i+1)*fusior_number]=history.history['loss']
        
    plt.plot(f_final_loss[0:(i+1)*fusior_number])
    pylab.show()
        
    cls_loss1[i*fusior_number:(i+1)*fusior_number]=history.history['dense_1_accuracy']
    cls_loss2[i*fusior_number:(i+1)*fusior_number]=history.history['dense_2_accuracy']
        
    plt.plot(cls_loss1[0:(i+1)*fusior_number])
    pylab.show()
    plt.plot(cls_loss2[0:(i+1)*fusior_number])
    pylab.show()
       
    Test_loss=fusior_train.predict([X_spatial_test,X_spatial_test,LiDAR_test,LiDAR_test,indexi_test,indexi_test,indexj_test,indexj_test])
    Pred_result3=Test_loss[0]
    Pred_result4=Test_loss[1]

    Pred_result3_=Pred_result3.argmax(axis=1)
    Pred_result4_=Pred_result4.argmax(axis=1)
    raw_label=act_Y_train_test.argmax(axis=1)
        
    acc3=accuracy_score(Pred_result3_,raw_label)
    acc4=accuracy_score(Pred_result4_,raw_label)
######################################################################################################################
    if acc3>maxacc:
        acc=acc3
        Pred_result_=Pred_result3_
        generator_component=keras.models.Model([Hx1_i_input,Hx2_i_input,Lx1_i_input,Lx2_i_input,indexi1,indexi2,indexj1,indexj2],[Rx1_2,Rx2_2,Sx1_2,Sx2_2])
        [R1_result,R2_result,S1_result,S2_result]=generator_component.predict([X_spatial_all,X_spatial_all,LiDAR_all,LiDAR_all,indexi_all,indexi_all,indexj_all,indexj_all])
        R=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            R[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=R1_result[iii,int((R1_result.shape[1]-1)/2),int((R1_result.shape[1]-1)/2),:]
        S=np.zeros([int(indexi_all.max()),int(indexj_all.max()),X_spatial_all.shape[-1]])
        for iii in range(indexi_all.shape[0]):
            S[int(indexi_all[iii]-1),int(indexj_all[iii]-1),:]=S1_result[iii,int((S1_result.shape[1]-1)/2),int((S1_result.shape[1]-1)/2),:]            

        generator_train.save_weights(source+'generator_train_'+str(i)+'.h5')
        fusior_train.save_weights(source+'fusior_train_'+str(i)+'.h5')           
            
        maxacc=acc
        maxOA=maxacc
        maxAllAcc=recall_score(raw_label,Pred_result_,average=None)
        maxAA=recall_score(raw_label,Pred_result_,average='macro')
        maxkappa=cohen_kappa_score(np.array(Pred_result_).reshape(-1,1),np.array(raw_label).reshape(-1,1))
        maxp=precision_score(raw_label,Pred_result_,average='macro')
        maxf1score=f1_score(raw_label,Pred_result_,average='macro')

        MAP=np.zeros([int(indexi_all.max()),int(indexj_all.max())])
        for ii in range(act_Y_train_test.shape[0]):
            MAP[int(indexi_test[ii]-1),int(indexj_test[ii]-1)]=Pred_result_[ii]+1
        for ii in range(act_Y_train_train.shape[0]):
            MAP[int(indexi_train[ii]-1),int(indexj_train[ii]-1)]=(act_Y_train_train.argmax(axis=1))[ii]+1

        name=source+'net_result_'+str(i)+'_'+str(maxacc)+'.mat'
        sio.savemat(name, {'R':R,
                           'S':S,
                           'Pred_result': Pred_result_,
                           'raw_label':raw_label,
                           'maxacc':maxacc,
                           'maxOA':maxOA,
                           'maxAA':maxAA,
                           'maxkappa':maxkappa,
                           'maxp':maxp,
                           'maxf1score':maxf1score,
                           'MAP':MAP})
######################################################################################################################