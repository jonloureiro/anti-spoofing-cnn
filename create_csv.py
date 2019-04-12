################################################################################
# git: jonloureiro/anti-spoofing-cnn
################################################################################

import csv

# GLOBAL VARIABLES #############################################################

PATH           = './Detectedface/'
CLIENT_TRAIN   = 'client_train_face.txt'
CLIENT_TEST    = 'client_test_face.txt'
IMPOSTER_TRAIN = 'imposter_train_face.txt'
IMPOSTER_TEST  = 'imposter_test_face.txt'

# FILES ########################################################################

def createLines(data, folder, contents, label):
    i = 0
    while i < len(contents):
        path = ''
        while contents[i] != ' ':
            if contents[i] == '\\':
                path += '/'
            else:
                path += contents[i]
            i += 1
        
        while contents[i] != '\n':
            i += 1

        data.append([folder+path, label])
        i += 1

def createFile(group, client, imposter):
    csvData = [['id', 'label']]
    createLines(csvData, 'ClientFace/', client, 'true')
    createLines(csvData, 'ImposterFace/', imposter, 'false')

    with open(group + '.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()

# MAIN #########################################################################

if __name__ == '__main__':
    fileClientTrain = open(PATH + CLIENT_TRAIN, 'r').read()
    fileImposterTrain = open(PATH + IMPOSTER_TRAIN, 'r').read()
    createFile('train', fileClientTrain, fileImposterTrain)

    fileClientTest = open(PATH + CLIENT_TEST, 'r').read()
    fileImposterTest = open(PATH + IMPOSTER_TEST, 'r').read()
    createFile('test', fileClientTest, fileImposterTest)
