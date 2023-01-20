import tkinter as tk
from tkinter import ttk
import threading
import neurotile as nt
from PIL import ImageTk

# nt = None
# importFinished = False
# 
# def asyncNeurotileLoad():
#     global importFinished
#     global nt
#     nt = __import__("neurotile")
#     importFinished = True

curX = 0
curY = 0
curK = 256
scaleFactor = 1.0
imgW = 0
imgH = 0
stackIndex = 0
maxStackIndex = 0
inferenceOutput = None

def tensorToPhotoImage(tensor, mulFactor=255.0, targetSize=(512,512), saveScaleValues=False, stackIndex=0):
    global imgW
    global imgH
    global scaleFactor
    
    if len(tensor.shape) == 4:
        width = tensor.shape[2]
        height = tensor.shape[1]
    else:
        width = tensor.shape[1]
        height = tensor.shape[0]
        
    if saveScaleValues:
        imgW = width
        imgH = height
        scaleFactor = max(imgW,imgH) / 512.0
#     pilImage = nt.PIL.Image.new("RGB", (width, height))
#     print(type(stackIndex))
    pilImage = nt.PIL.Image.fromarray((tensor[:,:,stackIndex*3:(stackIndex+1)*3].numpy() * mulFactor).astype("uint8"))
    pilImage.thumbnail(targetSize)
    photoImage = ImageTk.PhotoImage(pilImage)
#     nt.plt.imshow(pilImage)
#     nt.plt.show()
#     nt.plt.imshow(baseImage[:,:,:3])
#     nt.plt.show()
    return photoImage
    
def loadProjectFromDirectory():
    global inputCanvasImage
    global inputCanvasPhoto
    global inputCanvasRegion
    global curX
    global curY
    global curK
    global stackIndex
    global maxStackIndex
    global texIndexScale
    projectDirectory = tk.filedialog.askdirectory(initialdir=nt.currentProjectPath, title="Load Neurotile Project")
    print(projectDirectory)
    projectName = projectDirectory[projectDirectory.rfind("/")+1:]
    projectDirectory = projectDirectory[:projectDirectory.rfind("/")+1]
    print(projectName)
    print(projectDirectory)
    nt.currentProjectFolder = projectDirectory
    nt.setProject(projectName)
    nt.loadModels()
    stackIndex = 0
    maxStackIndex = (nt.baseImage.shape[2]//3) - 1
    texIndexScale.configure(to = maxStackIndex)
    texIndexScale.set(0)
    drawInputCanvasImage()
    curK = min(256, min(imgW, imgH))
    curX = 0
    curY = 0
    inputCanvasRegion = None
    drawInputRegion()
    inference(curX,curY,curK)
    
    
def trainProjectFromDirectory():
    projectDirectory = tk.filedialog.askdirectory(initialdir=nt.currentProjectPath, title="Load Neurotile Project")
    projectName = projectDirectory[projectDirectory.rfind("/")+1:]
    projectDirectory = projectDirectory[:projectDirectory.rfind("/")+1]
    nt.currentProjectFolder = projectDirectory
    nt.setProject(projectName)
    
    images = filter(lambda x : ".png" in x or ".jpg" in x or ".jpeg" in x, nt.os.listdir(nt.currentProjectPath))
    for image in images:
        try:
            print(image, end=".")
            nt.os.remove(nt.currentProjectPath+image)
        except:
            print("could not remove %s" % (nt.currentProjectPath+image))
            pass
    nt.clearLossLog()
    
    nt.DISC_LEARNING_FACTOR = 0.5
    nt.FEATUREMAP_START_SIZE = 32
    nt.LAMBDA_L1 = 0.0
    nt.LAMBDA_LADV = 0.0
    nt.LAMBDA_LSTYLE = 10.0
#     nt.NUM_RESIDUAL_BLOCKS = 10
    nt.NUM_RESIDUAL_BLOCKS = 5
    nt.LSTYLE_LAYERS = 3
    nt.KERNEL_SIZE_RESIDUAL_BLOCKS = 7
    nt.IMAGE_USE_ADVANCED_PADDING = True
    
    nt.tf.keras.backend.clear_session()
    nt.createModels()
    
    nt.train(0,500,0.005,128,500,100)
#     nt.train(0,5000,0.005,128,500,100)
#     
#     nt.LAMBDA_LADV = 1.0
#     nt.LAMBDA_L1 = 1.0
#     
#     print(len(nt.genModel.residualBlocks))
#     nt.genModel.residualBlocks = nt.genModel.residualBlocks[:5]
#     print(len(nt.genModel.residualBlocks))
#     nt.NUM_RESIDUAL_BLOCKS = 5
#     nt.LSTYLE_LAYERS = 2
#     
#     nt.train(5000,10000,0.001,128,500,100)
#     nt.train(10000,16000,0.0002,128,500,100)
#     nt.train(16000,22000,0.00005,128,500,100)
#     nt.saveModels()
#     nt.train(22000,28000,0.00002,128,500,100)
#     nt.train(28000,30001,0.00001,128,500,100)
    nt.saveModels()
    
    nt.plotLosses(10,True)

def inference(x,y,k):
    global outputCanvasPhoto
    global outputCanvasImage
    global stackIndex
    global inferenceOutput
    inferenceInput = nt.getSubImage(x,y,k)
    inferenceInput = (1.0-noiseScale.get())*inferenceInput + nt.tf.random.uniform(shape=inferenceInput.shape, minval=0.0, maxval=noiseScale.get())
    inferenceOutput = nt.createTileableTexturesFromInput(inferenceInput, True, True)
    outputCanvasPhoto = tensorToPhotoImage(inferenceOutput, stackIndex=stackIndex)
    outputCanvasImage = outputCanvas.create_image(256,256, image=outputCanvasPhoto)
    
def inputCanvasCallback(event):
    global curX
    global curY
    curX = max(0,min(int(event.x * scaleFactor), imgW-curK-1))
    curY = max(0,min(int(event.y * scaleFactor), imgH-curK-1))
    
    drawInputRegion()
    if curX >= 0 and curX+curK <= imgW and curY >= 0 and curY+curK <= imgH:#actually not necessary after the added min(...) above
        inference(curX, curY, curK)
        
def noiseScaleCallback(event):
    inference(curX,curY,curK)

def kScaleCallback(event):
    global curK
    global curX
    global curY
    curK = min(int(kScale.get()), min(imgW,imgH))
    curX = min(curX, imgW-curK)
    curY = min(curY, imgH-curK)
    nt.tf.keras.backend.clear_session()
    drawInputRegion()
    inference(curX,curY,curK)
    
def texIndexCallback(event):
    global stackIndex
    global inputCanvasRegion
    stackIndex = int(texIndexScale.get())
    drawInputCanvasImage()
    inputCanvasRegion = None
    drawInputRegion()
    inference(curX,curY,curK)

def drawInputCanvasImage():
    global inputCanvasPhoto
    global inputCanvasImage
    global stackIndex
    inputCanvasPhoto = tensorToPhotoImage(nt.baseImage, mulFactor = 1.0, targetSize=(512,512), saveScaleValues=True, stackIndex=stackIndex)#save scale values in globals
    inputCanvasImage = inputCanvas.create_image(0, 0, image=inputCanvasPhoto, anchor="nw")

def drawInputRegion():
    global inputCanvasRegion
    if inputCanvasRegion is None:
         inputCanvasRegion = inputCanvas.create_rectangle(curX/scaleFactor, curY/scaleFactor, (curX+curK)/scaleFactor, (curY+curK)/scaleFactor)
    inputCanvas.coords(inputCanvasRegion,curX/scaleFactor, curY/scaleFactor, (curX+curK)/scaleFactor, (curY+curK)/scaleFactor)

def exportTextures():
    global inferenceOutput
    global maxStackIndex
    for i in range(maxStackIndex+1):
        nt.saveImage(inferenceOutput[:,:,i*3:(i+1)*3],"export"+str(i))

def exporter():
    global inputCanvas
    global inputCanvasImage
    global inputCanvasPhoto
    global inputCanvasRegion
    global outputCanvas
    global outputCanvasImage
    global outputCanvasPhoto
    global noiseScale
    global kScale
    global texIndexScale
    global root
    
    root = tk.Tk()
    root.title("Neurotile Exporter GUI")
    root.geometry("1050x750")
    root.resizable(False,False)

    tk.Button(root, text="Load Project", command=loadProjectFromDirectory).grid(row=0,column=0, sticky=tk.W)

    inputCanvas = tk.Canvas(root, bg="white", width=512, height=512)
    inputCanvas.grid(row=1, column=0)
    inputCanvas.bind("<B1-Motion>", inputCanvasCallback)
    inputCanvas.bind("<Button-1>", inputCanvasCallback)
    inputCanvasImage = None
    inputCanvasPhoto = None
    inputCanvasRegion = None
    # drawInputRegion()

    outputCanvas = tk.Canvas(root, bg="white", width=512, height=512)
    outputCanvas.grid(row=1,column=2)
    outputCanvasImage = None
    outputCanvasPhoto = None

    noiseLabel = tk.Label(root, text="input noise factor")
    noiseLabel.grid(row=2,column=0)
    noiseScale = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, length=300, orient=tk.HORIZONTAL)
    noiseScale.grid(row=3,column=0)
    noiseScale.bind("<ButtonRelease-1>", noiseScaleCallback)
    noiseScale.bind("<B1-Motion>", noiseScaleCallback)

    kLabel = tk.Label(root, text="input size")
    kLabel.grid(row=4, column=0)
    kScale = tk.Scale(root, from_=32, to=512, resolution=1, length=480, orient=tk.HORIZONTAL)
    kScale.set(256)
    kScale.grid(row=5, column=0)
    kScale.bind("<ButtonRelease-1>", kScaleCallback)
    #kScale.bind("<B1-Motion>", kScaleCallback)


    texIndexLabel = tk.Label(root, text="texture stack index")
    texIndexLabel.grid(row=6, column=0)
    texIndexScale = tk.Scale(root, from_=0, to=0, resolution=1, length=480, orient=tk.HORIZONTAL)
    texIndexScale.grid(row=7, column=0)
    texIndexScale.bind("<ButtonRelease-1>", texIndexCallback)

    tk.Button(root, text="Export Textures", command=exportTextures).grid(row=2,column=2, sticky=tk.W)

    # threading.Thread(target=asyncNeurotileLoad).start()
    root.mainloop()
    
def training():
    global root
    
    root = tk.Tk()
    root.title("Neurotile Training")
    root.geometry("1050x750")
    
    tk.Button(root, text="Train", command=trainProjectFromDirectory).grid(row=0,column=0, sticky=tk.W)
    
    root.mainloop()
    
    
def main_toExporter():
    global root
    root.destroy()
    exporter()
    mainWindow()

def main_toTraining():
    global root
    root.destroy()
    training()
    mainWindow()
    
def mainWindow():
    global root
    
    root = tk.Tk()
    root.title("Neurotile Main Menu")
    root.geometry("300x40")
    
    tk.Button(root, text="Train", command=main_toTraining).grid(row=0,column=0, sticky=tk.W)
    tk.Button(root, text="Export", command=main_toExporter).grid(row=0,column=1, sticky=tk.W)
    
    root.mainloop()

#for now we instantly move to the exporter since the rest of the gui project is TODO WIP.
#mainWindow()
exporter()