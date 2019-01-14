import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

get_ipython().magic('matplotlib inline')

neuron = 20;

MaxIter = 10

batch_size = 100

scaler = MinMaxScaler()

#df = pd.read_csv('data180cyc12All0.csv')

df = pd.read_csv('dataGrad180Train0.csv')

df = df.drop(['Cell Type'],axis=1)

df = df.drop(['Ux','Uy','Uz'],axis=1)

df = df.sample(frac=1).reset_index(drop=True)

def magGradUx(df):
    mGUx = np.sqrt(np.square(df['grad(Ux):0']) + np.square(df['grad(Ux):1']) + np.square(df['grad(Ux):2']))
    return mGUx

def magGradUy(df):
    mGUy = np.sqrt(np.square(df['grad(Uy):0']) + np.square(df['grad(Uy):1']) + np.square(df['grad(Uy):2']))
    return mGUy

def magGradUz(df):
    mGUz = np.sqrt(np.square(df['grad(Uz):0']) + np.square(df['grad(Uz):1']) + np.square(df['grad(Uz):2']))
    return mGUz


mgux = magGradUx(df)
mgux = mgux.values.reshape(len(mgux),1)
mguy = magGradUy(df)
mguy = mguy.values.reshape(len(mguy),1)
mguz = magGradUz(df)
mguz = mguz.values.reshape(len(mguz),1)

temp1 = np.concatenate((mgux, mguy), axis=1)
temp2 = np.concatenate((temp1, mguz), axis=1)
df2 = np.concatenate((temp2, df['nut'].values.reshape(len(df['nut']),1)  ), axis=1)

df2 = pd.DataFrame(df2)
#df2 = pd.DataFrame(    np.row_stack([df2.columns, df2.values]),     columns=['Ux','Uy','Uz','gradMagUx', 'gradMagUy', 'gradMagUz','nut'] )
df2 = pd.DataFrame(    np.row_stack([df2.columns, df2.values]),     columns=['gradMagUx', 'gradMagUy', 'gradMagUz','nut'] )
df2 = df2.drop([0], axis=0)

df = df2

dimAll = np.shape(df);

dim2 = dimAll[1];

# 60 %
df_train = df[:int(np.floor(0.6*len(df)))]
# 20%
rest = df[int(np.floor(0.6*len(df))):]
df_val = rest[:int(np.floor(0.5*len(rest)))]

X_train = scaler.fit_transform(df_train.drop(['nut'],axis=1).as_matrix())
y_train = scaler.fit_transform(df_train['nut'].as_matrix().reshape(-1, 1))

X_val = scaler.fit_transform(df_val.drop(['nut'],axis=1).as_matrix())
y_val = scaler.fit_transform(df_val['nut'].as_matrix().reshape(-1, 1))


# 20%
dfTest = rest[int(np.floor(0.5*len(rest))):]

X_test = scaler.fit_transform(dfTest.drop(['nut'],axis=1).as_matrix())
y_test = scaler.fit_transform(dfTest['nut'].as_matrix().reshape(-1, 1))

print(X_train.shape)
print(np.max(y_val),np.max(y_train),np.min(y_val),np.min(y_train))

def denormalize(df,norm_data):
    df = df['nut'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new

def neural_net_model(X_data,input_dim):
    W_1 = tf.Variable(tf.random_uniform([input_dim,neuron]))
    b_1 = tf.Variable(tf.zeros([neuron]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    W_2 = tf.Variable(tf.random_uniform([neuron,neuron]))
    b_2 = tf.Variable(tf.zeros([neuron]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)

    W_3 = tf.Variable(tf.random_uniform([neuron,neuron]))
    b_3 = tf.Variable(tf.zeros([neuron]))
    layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
    layer_3 = tf.nn.relu(layer_3)

#    W_4 = tf.Variable(tf.random_uniform([neuron,neuron]))
#    b_4 = tf.Variable(tf.zeros([neuron]))
#    layer_4 = tf.add(tf.matmul(layer_3,W_4), b_4)
#    layer_4 = tf.nn.relu(layer_4)

#    W_5 = tf.Variable(tf.random_uniform([neuron,neuron]))
#    b_5 = tf.Variable(tf.zeros([neuron]))
#    layer_5 = tf.add(tf.matmul(layer_4,W_5), b_5)
#    layer_5 = tf.nn.relu(layer_5)


#    W_6 = tf.Variable(tf.random_uniform([neuron,neuron]))
#    b_6 = tf.Variable(tf.zeros([neuron]))
#    layer_6 = tf.add(tf.matmul(layer_5,W_6), b_6)
#    layer_6 = tf.nn.relu(layer_6)

#    W_7 = tf.Variable(tf.random_uniform([neuron,neuron]))
#    b_7 = tf.Variable(tf.zeros([neuron]))
#    layer_7 = tf.add(tf.matmul(layer_6,W_7), b_7)
#    layer_7 = tf.nn.relu(layer_7)

#    W_8 = tf.Variable(tf.random_uniform([neuron,neuron]))
#    b_8 = tf.Variable(tf.zeros([neuron]))
#    layer_8 = tf.add(tf.matmul(layer_7,W_8), b_8)
#    layer_8 = tf.nn.relu(layer_8)

    W_O = tf.Variable(tf.random_uniform([neuron,1]))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(layer_3,W_O), b_O)


#    W_O = tf.Variable(tf.random_uniform([neuron,1]))
#    b_O = tf.Variable(tf.zeros([1]))
#    output = tf.add(tf.matmul(layer_2,W_O), b_O)

    return output,W_O

xs = tf.placeholder("float")
ys = tf.placeholder("float")

output,W_O = neural_net_model(xs,(dim2 - 1))

cost = tf.reduce_mean(tf.square(output-ys))
train = tf.train.AdamOptimizer(0.001).minimize(cost)
#train = tf.train.GradientDescentOptimizer(0.03).minimize(cost)

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

c_t = []
c_val = []


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    plt.ion()
    saver = tf.train.Saver()
    y_t = denormalize(df_train,y_train)

    for i in range(MaxIter):
        for j in range(0, X_train.shape[0], batch_size):
            batch_train_X = X_train[j:j+batch_size]
            batch_train_y = y_train[j:j+batch_size]
            sess.run([cost,train],feed_dict={xs:batch_train_X, ys:batch_train_y})
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        pred = sess.run(output, feed_dict={xs:X_train})
        pred = denormalize(df_train,pred)

        plt.xlabel('Cell count')
        plt.ylabel('nut')
        plt.title('nut prediction')
        plt.plot(range(len(y_train)), y_t,label='Original')

        plt.plot(range(len(y_train)), pred,'r-',label='Prediction')
        plt.legend(loc='best')
        plt.pause(0.001)
        plt.draw()

        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_val.append(sess.run(cost, feed_dict={xs:X_val,ys:y_val}))
        print('Epoch :',i,'Cost :',c_t[i])

    predVal = sess.run(output, feed_dict={xs:X_val})
    predTest = sess.run(output, feed_dict={xs:X_test})

    print('Cost :',sess.run(cost, feed_dict={xs:X_val,ys:y_val}))
    y_val = denormalize(df_val,y_val)
    predVal = denormalize(df_val,predVal)

    y_test = denormalize(dfTest,y_test)
    predTest = denormalize(dfTest,predTest)

    plt.plot(range(y_val.shape[0]),y_val,label="Original Data")
    plt.plot(range(y_val.shape[0]),predVal,label="Predicted Data")
    plt.legend(loc='best')
    """plt.ylabel('nut')
    plt.xlabel('Cell Count')
    plt.title('nut prediction')"""
    plt.show()
    if input('Save model ? [Y/N]') == 'Y':
        import os
        saver.save(sess, os.getcwd() + '/nut_dataset.ckpt')
        print('Model Saved')



plt.plot(np.sort(predTest, axis=0),'r') ;  plt.plot(np.sort(y_test, axis=0),'b') ;
