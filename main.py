#from data_processing.dataprocess import generateData
from data_processing import read_data as rd
from weak_learners import svm, mlp, dt, NB
#import platform






if __name__=="__main__":
    #print(platform.python_version())

    #Comment this out once there's data
    #generateData()

    #Get training and testing data
    inData = "FARM"
    dataDicts = rd.run(inData)

    #svm.run(dataDicts)
    # Define value of inData:
    # <empty string> - MFCC + Power Spectrum
    # ARM - AR + MFCC
    # ARP - AR + Power Spectrum
    

    #svm.run(dataDict, inData)
    #mlp.run(dataDict)
    #dt.run(dataDicts, ada=False)
    #NB.run(dataDict)
