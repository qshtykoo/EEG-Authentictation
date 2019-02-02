#from data_processing.dataprocess import generateData
from data_processing import read_data as rd
from weak_learners import svm, mlp, dt, NB
import warnings

warnings.filterwarnings('ignore')


def runSVM(dataDict):

        svm.run(dataDict, inData)


if __name__=="__main__":
    import platform
    print(platform.architecture())
    #print(platform.python_version())

    #Comment this out once there's data
    #generateData()

    inData = "NWARM"

    #Get training and testing data
    datasets = ["ARM", "FARM", "WARM", "NWARM"]
    
    '''
    dataDict is a dictionary containing 3 datasets, which are respectively "beach", "finger" and "rest" which are already separated into training data(first 4 trials) and test data(last trial)
    croval_data is a list of data dictionaries which are composed of 4 folds of datasets for 4-fold cross validation, which is only applied for Decision Tree Classifier
    '''
    
    '''
    for inData in datasets:
        dataDict, croval_data = rd.run(inData)
        print(inData)
        runSVM(dataDict)
    '''
    # Define value of inData:
    # <empty string> - MFCC + Power Spectrum
    # ARM - AR + MFCC
    # ARP - AR + Power Spectrum

    # mlp.run(dataDicts)

    #NB.run(dataDicts)

    #dt.run(croval_data, ada=False, max_depth=13)
    #dt.run_grid_search(croval_data, ada=True)

    #dt.test_accuracy(dataDict, depth=13, key="rest")
    #T=[5, 10, 15, 20, 50, 100]
    #dt.test_accuracy(dataDict, depth=13, key="rest", ada=True, T=15)
    #data = dt.plot_best_features(dataDict, 'beach')
    

    #T = [5, 10, 15, 20, 50, 100]
    #for t in T:
    #    dt.test_accuracy(dataDict, depth=13, key="rest", ada=True, T=t)

