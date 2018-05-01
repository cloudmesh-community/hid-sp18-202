import requests
from flask import Flask
import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from os import listdir
from flask import Flask, request

app = Flask(__name__)

def download_data(url, filename):
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def data_partition(filename, ratio):
    file = open(filename,'r')
    training_file=filename+'_train'
    test_file=filename+'_test'
    data = file.readlines()
    count = 0
    size = len(data)
    ftrain =open(training_file,'w')
    ftest =open(test_file,'w')
    for line in data:
        if(count< int(size*ratio)):
            ftrain.write(line)
        else:
            ftest.write(line)
        count = count + 1        

def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]

@app.route('/')
def index():
    return "Demo Project!"

@app.route('/api/download/data')
def download():
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale'
    download_data(url=url, filename='dna.scale')
    return "Data Downloaded"

@app.route('/api/data/partition')
def partition():
    data_partition('dna.scale',0.95)
    return "Successfully Partitioned"

@app.route('/api/get/data/test')
def gettestdata():
    Xtest, ytest = get_data("dna.scale_test")
    
    return "Return Xtest and Ytest arrays"

@app.route('/api/get/data/train')
def gettraindata():
    Xtrain, ytrain = get_data("dna.scale_train")
    
    return "Return Xtrain and Ytrain arrays"

@app.route('/api/experiment/svm')
def svm():
    Xtrain, ytrain = get_data("dna.scale_train")
    Xtest, ytest = get_data("dna.scale_test")

    clf = SVC(gamma=0.015625, C=100, kernel='rbf')
    clf.fit(Xtrain, ytrain)

    test_size = Xtest.shape[0]
    accuarcy_holder = []
    for i in range(0, test_size):
        prediction = clf.predict(Xtest[i])
        print("Prediction from SVM: "+str(prediction)+", Expected Label : "+str(ytest[i]))
        accuarcy_holder.append(prediction==ytest[i])

    correct_predictions = sum(accuarcy_holder)
    print(correct_predictions)
    total_samples = test_size
    accuracy = float(float(correct_predictions)/float(total_samples))*100
    print("Prediction Accuracy: "+str(accuracy))
    return "Prediction Accuracy: "+str(accuracy)

    

# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
import numpy
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale"
names = ['A', 'C', 'G', 'T']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,4,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

if __name__ == '__main__':
    app.run(debug=True)
