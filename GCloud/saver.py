import keras.models as models
import pickle
from tensorflow.python.lib.io import file_io




def saveModel(model, file):
        models.save_model(model, file)
        print("Model saved.")
        
 
def saveModelToCloud(model, job_dir, name='model'):
        filename = name + '.h5' 
        model.save(filename)
        with file_io.FileIO(filename, mode='r') as inputFile:
                with file_io.FileIO(job_dir + '/' + filename, mode='w+') as outFile:
                        outFile.write(inputFile.read())

def saveListOfNumbers(numbers, outputPath):
        with open(outputPath, 'w') as file:
                file.writelines([str(number) + '\n' for number in numbers])
               
def saveObjectAsPickle(o, pathToOutput):
        with open(pathToOutput, 'wb') as output:
                pickle.dump(o, output, protocol=2)
        print("Pickle Saved.")        
                        
                        
                        