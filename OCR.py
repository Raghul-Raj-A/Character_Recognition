from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import subprocess

#letters = ['img028','img054'] #Character R
'''
letters = ['img001','img002','img003','img004','img005','img006','img007','img008','img009','img010','img011',
        'img012','img013','img014','img015','img016','img017','img018','img019','img020','img021','img022','img023','img024',
        'img025','img026','img027','img028','img029','img030','img031','img032','img033','img034','img035','img036','img037','img038','img039','img040','img041',
        'img042','img043','img044','img045','img046','img047','img048','img049','img050','img051','img052','img053','img054',
        'img055','img056','img057','img058','img059','img060','img061','img062'] #all characters

letters = ['img011','img012','img013','img014','img015','img016','img017','img018','img019',
        'img020','img021','img022','img023','img024','img025','img026','img027','img028',
        'img029','img030','img031','img032','img033','img034','img035','img036'] #Uppercase letters

letters = ['img037','img038','img039','img040','img041','img042','img043','img044','img045',
        'img046','img047','img048','img049','img050','img051','img052','img053','img054',
        'img055','img056','img057','img058','img059','img060','img061','img062'] #lowercase letters
'''
letters = ['img001','img002','img003','img004','img005','img006','img007','img008','img009','img010'] # digits


noc = len(letters)
wtih = [] #weight plotter           
wtho = []
NoI = []
Time = []
digit = ['img001','img010']
upper = ['img011','img036']
lower = ['img037','img062']
wtihw = []
wthow = []
inplot = False
#---------------------------------------------------------------------------------------------------------------------------------------------
#grayscale covertion

def convertToBW(imageIn):
    blackAndWhite = imageIn.convert('1')           # 0 -> black , 1 -> white
    blackAndWhite = np.array(blackAndWhite)*1      # convert to array
    return blackAndWhite                                                    # 0 -> black , 1 -> white

def toggleOnesAndZeros(blackAndWhite):                                      # 1 -> black , 0 -> white
    return (blackAndWhite ^1)
    
def modifyInputPixelsValues(blackAndWhite):      #redundant                 # Assign +3 -> black 
    in_blackAndWhite = blackAndWhite
    [numberRowPixels , numberColumnPixels] = blackAndWhite.shape            # find array dimensions
   
    for i in range (0,numberRowPixels):                                                                         
        for j in range (0,numberColumnPixels):                                  
            if in_blackAndWhite[i,j] >0 :                                   
                in_blackAndWhite[i,j] = 3

    toggled = in_blackAndWhite
    return toggled

#---------------------------------------------------------------------------------------------------------------------------------------------
#image resizing and cropping 
       
def crop(blackAndWhiteToggled):
    
    [numberOfRowPixels , numberOfColumnPixels] = blackAndWhiteToggled.shape              # find array dimensions

    #Finding the left and right side
    verticalSumOfBlackPixels                   = np.sum(blackAndWhiteToggled,axis=0)     # gives a list of number of black pixels in each column
    leftDetected = False
    for i in range(0,numberOfColumnPixels):
        if verticalSumOfBlackPixels[i] > 0 and leftDetected == False:             # there is a black pixel in this column
            leftDetected = True
            left = i                                                              # left
        elif verticalSumOfBlackPixels[i] > 0 and leftDetected == True:
            right = i                                                             # right
		
    #Finding the top and bottom side
    horizontalSumOfBlackPixels                   = np.sum(blackAndWhiteToggled,axis=1)   # gives a list of number of black pixels in each row
    topDetected = False
    for i in range(0,numberOfRowPixels):
        if horizontalSumOfBlackPixels[i] > 0 and topDetected == False:            # there is a black pixel in this column
            topDetected = True
            top = i                                                               # top
        elif horizontalSumOfBlackPixels[i] > 0 and topDetected == True:
            bottom = i                                                            # bottom
        
    v_CroppedBlackAndWhite_array  = blackAndWhiteToggled[:,(range(left,right+1))]
    finalCroppedBlackAndWhite     = v_CroppedBlackAndWhite_array[(range(top,bottom+1)),:]
    # Transform array back to image
    return finalCroppedBlackAndWhite

#---------------------------------------------------------------------------------------------------------------------------------------------
#normalization of image data(organising)
    
def normalize(character_in,width,height):
    
    character_in   = im.fromarray(character_in)
    normalized     = character_in.resize((width,height),im.Resampling.HAMMING)         # normalize image to desired dimensions
    NormalizedArray    = np.array(normalized) 
    return NormalizedArray                                                
 
#---------------------------------------------------------------------------------------------------------------------------------------------
#weight initialization
  
def initializeWeights(width,height,numberOfHiddenNeurons):
    Wi_h = np.random.random(size=(numberOfHiddenNeurons,height,width))-0.5  # input to hidden layer weights
    Wh_o = np.random.random(size=(noc,numberOfHiddenNeurons))-0.5            # hidden to output weights
    Bh   = np.random.random(numberOfHiddenNeurons) - 0.5                    # hidden layer biases
    Bo   = np.random.random(noc) - 0.5                                       # output layer biases    
    
    return Wi_h, Wh_o, Bh, Bo

#---------------------------------------------------------------------------------------------------------------------------------------------
#threshold activation function

def sigmoid(net_val):          
        # sigmoid function                                          
       out = 1 / (1 + np.exp(-net_val))
       return out

#---------------------------------------------------------------------------------------------------------------------------------------------
#Feed forward pass
       
def feedForward(normalized, Wi_h, Wh_o, Bh, Bo):                           
    
    n_h = 0
    [numberOfHiddenNeurons,height,width] = Wi_h.shape
    #From input layer to hidden
    outputOfHiddenNeurons = []
    netInputForHiddenNeurons = []
    
    for hiddenNeuron in range (0,numberOfHiddenNeurons):                 # forward pass from input to hidden
        
        for i in range (0,height):                                       # calculating net activation input
            for j in range (0,width):                           
                WxP = Wi_h[hiddenNeuron,i,j] * normalized[i,j]           # Weight X Input 
                n_h   = n_h + WxP                                        # The overall sum of W's X P's
               
        n_h = n_h + Bh[hiddenNeuron]                                     # total input = WP+Bias
        outputOfHiddenNeurons.append(sigmoid(n_h))                       # calculate and save hidden neurons output
        netInputForHiddenNeurons.append(n_h)                             
        n_h = 0                                        

    #From hidden to output
    outHiddenXweightsH_O      = outputOfHiddenNeurons * Wh_o             # out of hidden layer multiplied by weights from hidden to output layer
    netInputForOutNeurons     = np.sum(outHiddenXweightsH_O, axis= 1)    # sum of all (Wh_o weights X hidden neuron outputs), each row is the total input for each output neuron
    outputOfOutNeurons        = []
    
    for outputNeuron in range(0,noc):                                                        # Find and Calculate output of out neurons                                                                                    
        totalInputForNeuron   = netInputForOutNeurons[outputNeuron] + Bo[outputNeuron]      # the input to the kth output neuron
        outputOfOutNeurons.append(sigmoid(totalInputForNeuron))                             # get the output and save it

    return outputOfOutNeurons, outputOfHiddenNeurons                     

#---------------------------------------------------------------------------------------------------------------------------------------------
#Calculate Error At Output Neurons

def calculateErrorAtOutput(outputOfOutNeurons, targetOutput):
    outputError =[]
    for outputNeuron in range(0,noc):
        outputNeuronError = outputOfOutNeurons[outputNeuron] - targetOutput[outputNeuron]   # error = out - target
        outputError.append(outputNeuronError)                                               # save error for all outputs
    return outputError                                                                      

#---------------------------------------------------------------------------------------------------------------------------------------------
#Backpropagation through net and weight updation

def backPropagate(Wi_h, Wh_o, Bh, Bo, normalized, outputError, outputOfOutNeurons, outputOfHiddenNeurons, learningRate, momentum):

    oldWh_o    = np.array(Wh_o[:,:])                                   
    oldWi_h    = np.array(Wi_h[:,:]) 
   
    [numberOfHiddenNeurons,height,width] = Wi_h.shape

    #Back Propagating from output to hidden and adjusting weights

    for outputNeuron in range(0,noc):
        for hiddenNeuron in range(0,numberOfHiddenNeurons):
            # calculating the adjustment which is learning rate * error at current output neuron * sigmoid derivative *  output of current hidden neuron)
            adjustment                       = (learningRate * outputError[outputNeuron] * outputOfOutNeurons[outputNeuron] * (1 - outputOfOutNeurons[outputNeuron]) * outputOfHiddenNeurons[hiddenNeuron])
            Wh_o[outputNeuron, hiddenNeuron] = (momentum * Wh_o[outputNeuron, hiddenNeuron]) - adjustment 
            wtho.append(Wh_o)
    #Back Propagating from hidden to input and adjusting weights

    for hiddenNeuron in range(0,numberOfHiddenNeurons):
        deltaTotalError_hiddenNeuron = 0
        for outputNeuron in range(0,noc):
            # Calculate delta error at each output neuron with respect to current hidden neuron
            deltaErrorOutputNeuron_hiddenNeuron = outputError[outputNeuron] *  outputOfOutNeurons[outputNeuron] * (1-outputOfOutNeurons[outputNeuron]) * oldWh_o[outputNeuron,hiddenNeuron]
            deltaTotalError_hiddenNeuron = deltaTotalError_hiddenNeuron + deltaErrorOutputNeuron_hiddenNeuron # delta total error with respect to current hidden neuron

        # loop over all input weights connecting to current hidden neuron
        for i in range (0,height):
            for j in range (0,width):
                # delta Total Error with respect to weight to be adjusted. this weight is connecting input to current hidden layer
                deltaTotalError_inputTohiddenNeuronWeight = deltaTotalError_hiddenNeuron * outputOfHiddenNeurons[hiddenNeuron] *(1 - outputOfHiddenNeurons[hiddenNeuron]) * normalized[i,j]
                Wi_h[hiddenNeuron,i,j] = (momentum * Wi_h[hiddenNeuron,i,j]) - (learningRate * deltaTotalError_inputTohiddenNeuronWeight)
                wtih.append(Wi_h)
    return Wi_h, Wh_o
   
#---------------------------------------------------------------------------------------------------------------------------------------
#For Training 
   
def trainNet(Wi_h, Wh_o, Bh, Bo,height,width,numberOfTrainingSamples,learningRate,momentum,targetError):
    iteration  = 0
    totalError = 1
    errorList  = []     # to save all total error generated
    y_axis     = []     # for plotting the error minimization at the end
    start = time.time()   
    y_plot = []
    itr = 0
    a = b = c =0
    opt8 = input("Do to need to plot weight updation to iteration between input and hidden layer (y/n) :")
    if opt8 == 'y':
        opt9 = input("Enter hidden neuron :")
        opt10,opt11 = input("Enter input neuron in matrix form : ").split("*")
        a = int(opt9)
        b = int(opt10)
        c = int(opt11)
        inplot = True
    else:
        None
        inplot = False

    while totalError > targetError:                                            

        for letterToTrain in range(0,noc):                                      
            targetOutput = np.zeros(noc)                                        # target output is all zeros
            targetOutput[letterToTrain] = 1                                    # except the one to be trained
            
            for n in range (1,numberOfTrainingSamples):                        # number of training samples 

                #trainingSample      = 'DS_R/%s-%03d.png' %(letters[letterToTrain],n) #Only for character R
                trainingSample      = 'Train_data/%s-%03d.png' %(letters[letterToTrain],n) 
             
                character_in        = im.open(trainingSample)                 
                blackAndWhite       = convertToBW(character_in)           
                toggledBW           = toggleOnesAndZeros(blackAndWhite)
                croppedBW           = crop(toggledBW)                      # Croping Image to get character only
                normalized          = normalize(croppedBW,width,height)    
                
                #end of pre processing phase
                
                outputOfOutNeurons, outputOfHiddenNeurons = feedForward(normalized, Wi_h, Wh_o, Bh, Bo) 
                outputError         = calculateErrorAtOutput(outputOfOutNeurons, targetOutput)          
                Wi_h, Wh_o          = backPropagate(Wi_h, Wh_o, Bh, Bo, normalized, outputError, outputOfOutNeurons, outputOfHiddenNeurons, learningRate,momentum) 
                wtihw.append(Wi_h[a][b][c])
                itr = itr+1
                y_plot.append(itr)
        #Calculate the mean squared error
        
        totalError = 0
        for x in range(0,noc):
            squared     = 0.5 * outputError[x]**2
            totalError  = totalError + squared
            
        print('Error calculated = %f' %totalError)
        iteration = iteration + 1
        errorList.append(totalError)
        y_axis.append(iteration)
        NoI.append(iteration)
    end = time.time()-start
    Time.append(end)
    # Total Error vs Iteration graph
    print('Total Number of iterations %d' %iteration)
    opt1 = input("\nDo you want to visualize calculated error to iteration (y/n): ")
    if opt1 == 'y':
        plt.plot(y_axis, errorList)
        plt.ylabel('Total Error')
        plt.xlabel('Number Of Iterations')
        plt.show()
    else:
        plt.close()
    
    if inplot == True:
        plt.plot(y_plot, wtihw)
        plt.ylabel('updation wrt iteration')
        plt.xlabel('Number Of passes')
        plt.show()
    else:
        plt.close()

    return (Wi_h, Wh_o, Bh, Bo)

#---------------------------------------------------------------------------------------------------------------------------------------------
#Character Recogonization
    
def recognizeCharacter(inputNormalized,Wi_h,Wh_o,Bh,Bo):                
   
    outputOfOutNeurons, outputOfHiddenNeurons = feedForward(inputNormalized, Wi_h, Wh_o, Bh, Bo) 
    maxOut = np.argmax(outputOfOutNeurons) # here character recognized is neuron with highest output
    return letters[maxOut]

#---------------------------------------------------------------------------------------------------------------------------------------------
#For testing
def testnet():
    result = []
    test = []
    x = ""
    if len(letters) < 62:
        if(set(upper).issubset(set(letters))):
            test = letters
            #print("Testing only for Upper_case letters")
            x = "Test_data/Upper_case/"    
        if(set(lower).issubset(set(letters))):
            test = letters
            #print("Testing only for Lower_case letters")
            x = "Test_data/Lower_case/"  
        if(set(digit).issubset(set(letters))):
            test = letters
            #print("Testing only for Digits")
            x = "Test_data/Digits/"
        else:
            test = letters
            x = "Test_data/"
    else:
        test = letters
        x = "Test_data/"
            
    for letterToTest in range(0,len(test)):
        #test_data_location = 'C:/Users/raghu/Documents/7th_Semester/DoMS/Assignment_3/OCR_Python/Letter_R/Test_data/%s-%03d.png'%(test[letterToTest],letterToTest+1)
        test_data_location = x+'%s-%03d.png'%(test[letterToTest],letterToTest+1)
        imgin = im.open('%s' %test_data_location)          
        imginBW = convertToBW(imgin)                 
        imginBW = toggleOnesAndZeros(imginBW)
        #print('Image Loaded')     
        #imgin.show()
        inputCroppedBW          = crop(imginBW )                   
        inputNormalized         = normalize(inputCroppedBW,width,height)                   
        testout  = recognizeCharacter(inputNormalized,Wi_h, Wh_o, Bh, Bo)
        if testout == test[letterToTest]:
            result.append('match')
        
    acc = len(result)
    tot = len(test)
    accuracy = acc/tot
    #print("Accuracy : "+str(accuracy))
    return accuracy

#---------------------------------------------------------------------------------------------------------------------------------------------
def weights_I_H(nh):
    f = open('Weight_I_H.txt', 'r+')
    f.truncate(0)
    lis = []
    in_n = []
    in_neurons = 0
    for a in range (0,height):
        for b in range (0,width):    
            lis.append(Wi_h[nh-1][a][b])
    for no in lis:
        in_neurons = in_neurons+1
        in_n.append(in_neurons)        

    plt.plot(in_n, lis)
    plt.ylabel('Updated weights between Input and hidden layer')
    plt.xlabel('Number Of input neurons')
    plt.show()

    with open("Weight_I_H.txt", 'w') as output:
        for row in lis: #wt[saturated-1]:
            output.write(str(row) + '\n')   
    subprocess.call(['notepad.exe', 'Weight_I_H.txt'])  

#---------------------------------------------------------------------------------------------------------------------------------------------
def weights_H_O(ot):
    f = open('Weight_H_O.txt', 'r+')
    f.truncate(0)
    lis1 = []
    h_n = []
    h_neurons = 0
    for a in range (0,numberOfHiddenNeurons):
        lis1.append(Wh_o[ot-1][a])
    for non in lis1:
        h_neurons = h_neurons+1
        h_n.append(h_neurons)        

    plt.plot(h_n, lis1)
    plt.ylabel('Updated weights between Hidden and Output layer')
    plt.xlabel('Number Of Hidden neurons')
    plt.show()

    with open("Weight_H_O.txt", 'w') as output:
        for row in lis1: #wt[saturated-1]:
            output.write(str(row) + '\n')    
    subprocess.call(['notepad.exe', 'Weight_H_O.txt'])
#-------------------------------------------------------------------------------------------------------------------------------------------------

def stats():
    f = open('Stats.txt', 'r+')
    f.truncate(0)
    noiinih = len(wtih)
    noiinho = len(wtho)
    accur = testnet() 
    noi = len(NoI)
    ti = Time[len(Time)-1]
    intro = "\t>>>>DETAILED REPORT<<<<\n"
    txt = "Code compiled/edited by Raghul Raj A :)"
    note = "#Note : Target error and Number of training samples are predefined \n\t  values can be changed @ Parameters"

    with open('Stats.txt','w') as s:
        s.write(intro+'\n')
        s.write("Number of input neurons : "+str(width*height))
        s.write("\n\nNumber of hidden neurons : "+str(numberOfHiddenNeurons))
        s.write("\n\nNumber of output neurons : "+str(len(letters)))
        s.write("\n\nNumber of characters : "+str(len(letters)))
        s.write("\n\nNumber of training samples in each character : "+str(numberOfTrainingSamples))
        s.write("\n\nTotal number of training samples : "+str(len(letters)*numberOfTrainingSamples))
        s.write("\n\nPre-defined target error : "+str(targetError))
        s.write("\n\n"+note)
        s.write("\n\nNumber of times weight changed between layers: ")
        s.write("\n\nInput and Hidden :"+str(noiinih))
        s.write("\tHidden and output :"+str(noiinho))
        s.write("\n\nAccuracy : {:.5f} ".format(accur))
        s.write("\n\nNumber of iterations : "+str(noi))
        s.write("\n\nTime taken to train : {:.4f}s".format(ti))
        if(not(set(upper).issubset(set(letters)))):
            s.write("\n\n>>Warning : Model is not trained with UpperCase Letters")
        if(not(set(lower).issubset(set(letters)))):
            s.write("\n\n>>Warning : Model is not trained with LowerCase Letters")
        if(not(set(digit).issubset(set(letters)))):
            s.write("\n\n>>Warning : Model is not trained with Digits(0-9)")
        s.write("\n\n\t"+txt)
    subprocess.call(['notepad.exe', 'Stats.txt'])
    
#-------------------------------------------------------------------------------------------------------------------------------------------------

def character(output):
    switcher = {
        'img001':0,
        'img002':1,
        'img003':2,
        'img004':3,
        'img005':4,
        'img006':5,
        'img007':6,
        'img008':7,
        'img009':8,
        'img010':9,
        'img011':'A',
        'img012':'B',
        'img013':'C',
        'img014':'D',
        'img015':'E',
        'img016':'F',
        'img017':'G',
        'img018':'H',
        'img019':'I',
        'img020':'J',
        'img021':'K',
        'img022':'L',
        'img023':'M',
        'img024':'N',
        'img025':'O',
        'img026':'P',
        'img027':'Q',
        'img028':'R',
        'img029':'S',
        'img030':'T',
        'img031':'U',
        'img032':'V',
        'img033':'W',
        'img034':'X',
        'img035':'Y',
        'img036':'Z',
        'img037':'a',
        'img038':'b',
        'img039':'c',
        'img040':'d',
        'img041':'e',
        'img042':'f',
        'img043':'g',
        'img044':'h',
        'img045':'i',
        'img046':'j',
        'img047':'k',
        'img048':'l',
        'img049':'m',
        'img050':'n',
        'img051':'o',
        'img052':'p',
        'img053':'q',
        'img054':'r',
        'img055':'s',
        'img056':'t',
        'img057':'u',
        'img058':'v',
        'img059':'w',
        'img060':'x',
        'img061':'y',
        'img062':'z' 
    }
    return(switcher.get(output,'Incorrect character'))

#---------------------------------------------------------------------------------------------------------------------------------------------
#Parameters

learningRate = 0.5; momentum = 1; targetError = 0.0035 ;numberOfHiddenNeurons = 80
width=18; height = 16 ; numberOfTrainingSamples = 2

#---------------------------------------------------------------------------------------------------------------------------------------------
#Initialize And Train & test Network 

print('Training Initialized')
Wi_h, Wh_o, Bh , Bo = initializeWeights(width,height,numberOfHiddenNeurons)                     
Wi_h, Wh_o, Bh , Bo = trainNet(Wi_h, Wh_o, Bh , Bo,height,width,numberOfTrainingSamples,learningRate,momentum,targetError) 
print('\nNeural Net Trained')
print("\nTime taken to train the net : {} ".format(Time[len(Time)-1]))

opt2 = input("\nDo you want to test the network (y/n) :")
if opt2 == 'y':
    accu = testnet()
    checker = True
    print("\nAccuracy : {} ".format(testnet()))
    while accu < 0.60 and checker == True:
        print("\nWarning !! Accuracy is less than 60%")
        opt7 = input("Do you want to retrain the net (y/n) : ")
        if opt7 == 'y':
            Wi_h, Wh_o, Bh , Bo = trainNet(Wi_h, Wh_o, Bh , Bo,height,width,numberOfTrainingSamples,learningRate,momentum,targetError) 
            accu = testnet()
            checker = True
        else:
            checker = False     
else:
    None

#---------------------------------------------------------------------------------------------------------------------------------------------

opt3 = input("\nDo you want to plot updated weights b/w Input and Hidden layer (y/n) :")
if opt3 == 'y':
    print("\nNumber of hidden_neurons : "+str(numberOfHiddenNeurons))
    opt4 = int(input("In which hidden neurons you want to visualize :"))
    weights_I_H(opt4)
    print("\n>>>>>Updated weights values are stored in 'Weight_I_H.txt' file")
else:
    None

opt5 = input("\nDo you want to plot updated weights b/w Hidden and Output layer (y/n) :")
if opt5 == 'y':
    print("\nNumber of hidden_neurons : "+str(len(letters)))
    opt6 = int(input("In which output neurons you want to visualize :"))
    weights_H_O(opt6)
    print("\n>>>>>Updated weights values are stored in 'Weight_I_H.txt' file")
else:
    None

#---------------------------------------------------------------------------------------------------------------------------------------------
stats()

contd = True
while contd == True:
    # Image Preprocessing 
    opi = input("\nDo you want to recogonize a character: (y/n) :" )
    if opi == 'y':
        #documentLocation = "C:/Users/raghu/Documents/7th_Semester/DoMS/Assignment_3/ALPHANUM_DATASET/Img/img028-022.png"
        docuLoc = input("Enter file path:")
        docuLoc = docuLoc.replace('"','')
        documentLocation = docuLoc.replace("\\","/")
        print('Loading Image..')
        imageIn = im.open('%s' %documentLocation)          
        imageInBW = convertToBW(imageIn)                 
        imageInBW = toggleOnesAndZeros(imageInBW)
        print('Image Loaded')     
        imageIn.show()
        inputCroppedBW          = crop(imageInBW )                   
        inputNormalized         = normalize(inputCroppedBW,width,height)                   
        output  = recognizeCharacter(inputNormalized,Wi_h, Wh_o, Bh, Bo)
        print("\n>>The identified character is probably : {}" .format(character(output)))
        contd  = True
    else:
        contd = False
        print("\n\t\tThank you!!!\n")

#---------------------------------------------------------------------------------------------------------------------------------------------
#Copyright (c) 2022, Raghul Raj A !!!!