from data_processing.dataprocess import generateData
from data_processing import read_data as rd
from weak_learners import svm, mlp, dt, NB
#import platform






if __name__=="__main__":
    #print(platform.python_version())

    #Comment this out once there's data
    #generateData()

    #Get training and testing data
    dataDict = rd.run()

    #svm.run(dataDict)
    #mlp.run(dataDict)
    dt.run(dataDict, ada=True)
    #NB.run(dataDict)
