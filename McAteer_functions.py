from super_image import EdsrModel, ImageLoader
from ultralyticsplus import render_result, YOLO
from PIL import Image
from copy import deepcopy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import pickle


def createPan(imageL):
    # Estimate high resolution panchromatic image
    # This wouldn't be needed if an actual high resolution pan image
    # Was available.

    # My favourite pretrained single image super resolution GAN
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=3)
    # The model requires RGB images
    image3 = np.empty((imageL.shape[0], imageL.shape[1], 3))
    image3[:, :, 0] = imageL
    image3[:, :, 1] = imageL
    image3[:, :, 2] = imageL
    # Mash into PIL format
    image3 = Image.fromarray((image3 * 255).astype(np.uint8)) 
    inputs = ImageLoader.load_image(image3)
    # Mash into numpy array
    imageLSR = 255-model(inputs).detach().numpy()*255
    # Average over colour channels to get the luma channel
    imageLSR = np.squeeze(np.mean(imageLSR, axis=1))
    return imageLSR

def panSharpen(imageRGB):
    # Fake panchromatic sharpening
    # Real case would use the high resolution pan data
    # (if available) from a satellite/plane
    # Pan data can be higher resolution and lower noise since
    # None of the light is filtered out

    # Swap to HLS colour format
    imageHLS = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2HLS)
    # Create fake high resolution pan data
    # Using a GAN
    fakeHighResPan = createPan(imageHLS[:, :, 1])
    # Upsample the colour data using a simple kernel upsample
    # This is just so the colour data exists on the same grid
    # as the pan data
    imageHLSUpsampled = cv2.resize(imageHLS, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    # Replace the luma channel with the fake pan channel
    imageHLSUpsampled[:, :, 1] = fakeHighResPan
    # Ta dah
    # Now just swap back to RGB colours
    imagePanSharpRGB = cv2.cvtColor(imageHLSUpsampled, cv2.COLOR_HLS2BGR)
    return imagePanSharpRGB
    

def setupBuildingDetectionModel():
    # Use this nice YOLOv8 model from huggingface
    # It's not very accurate, but would just require retraining
    # It also only detects buildings, but yolo is great for plans, 
    # cars etc. I have previously trainined a YOLO(v2) network
    # for detecting aeroplanes/drones in restrictive airspaces
    # and this is easier since the planes all have a similar orientation
    
    # Thank you keremberke
    model = YOLO('keremberke/yolov8n-building-segmentation')

    model.overrides['conf'] = 0.9  # NMS confidence threshold
    model.overrides['iou'] = 0.02  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    # print(model.overrides)
    return model


def loadImages(imageDir, imgIndex=0, justCount=False):
    # read in the images in imageDir and applies the preprocessing

    # glob files from directory and sort so that the order is repeatable
    fileNames = sorted(glob.glob(imageDir + '/*'))
    
    numFiles = len(fileNames)

    # return the number of images in the directory
    if justCount:
        return numFiles

    # list of images
    images = []
    for name in fileNames:
        image = cv2.imread(name)
        # The pan sharpening isn't used, but is cool
        # So give it a go
        imagePP = preprocessImage(image, panSharpenAtLoadIn=0)
        images.append(imagePP)

    # if specific images are required then load those only
    if imgIndex:
        images = images[imgIndex]
        numFiles = len(imgIndex)

    return images, numFiles


def preprocessImage(imageBGR, panSharpenAtLoadIn=1):

    #     Preprocessing:
    # Standardise images. Images from different satellites have different 
    # properties: filter bands, resolution, bit depth, angle of view, 
    # sensor type, data format, artifacts (e.g., push-broom artifact).
    # Only some of these factors are relevant 

    # The images do not have meaningful Exif metadata. 
    # I therefore cannot meaningfully resample the images to the same resolution. 
    # This is an issue because I don't have a prior for the size of objects 
    # in the images other than that can be determined by eye

    # Change from BGR to RGB
    # Since OpenCV is annoying
    
    # To do this properly I'd also care about the actual colour space
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    if panSharpenAtLoadIn:
        # Do pan sharpening
        imageRGB = panSharpen(imageRGB)
    return imageRGB


def showDamageMap(image, render, damageMap, showPlots=1, savePlots=0, saveFilename='fig.png', dpi=600):
    # this just shows the damage map
    
    plt.figure()
    fig, [ax0, ax1, ax2] = plt.subplots(3, 1, constrained_layout=True)
    
    ax0.imshow(image)
    ax0.set_title('Image')
    
    mapVar = ax1.imshow(damageMap, vmin=0, vmax=1)
    ax1.set_title('Damage Map')
    c = fig.colorbar(mapVar, ax=ax1)
    c.ax.get_yaxis().set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    c.ax.set_ylabel('Probability of damage (uncalibrated)')
    
    ax2.imshow(render)
    ax2.set_title('Building Detection')
    if showPlots:
        plt.show()

    if savePlots:
        plt.savefig(saveFilename, dpi=dpi)



def padImage(image, tileSize=128):
    s = image.shape
    # pad image to be divisible by tileSize
    hR = s[0]%tileSize
    wR = s[1]%tileSize
    bottom = 0
    right = 0
    if hR != 0:
        bottom = tileSize - hR
    if wR != 0:
        right = tileSize - wR
    # Apply reflected boarder
    image = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_REFLECT)
    return image, bottom, right


def unPadImage(padImage, bottom, right):
    # Remove padding
    if bottom == 0:
        bottom = -padImage.shape[0]
    if right == 0:
        right = -padImage.shape[1]

    if len(padImage.shape) == 3:
        return padImage[:-bottom, :-right, :]
    else:
        return padImage[:-bottom, :-right]


def imageToTiles(image, tileSize=128):
    # Chop image into tiles for the classifier.
    # Image should be padded to an integer tile number.
    s = image.shape
    hN = int(s[0]/tileSize)
    wN = int(s[1]/tileSize)
    tileSet = np.empty((hN, wN, tileSize, tileSize, 3))

    for i in range(hN):
        for j in range(wN):
            tileSet[i, j, :, :, :] = image[i*tileSize:(i+1)*tileSize, j*tileSize:(j+1)*tileSize, :]
    return tileSet


def tilesToImage(tileSet):
    # Reconstitue tiles into whole image
    ts = tileSet.shape
    tileSize = ts[2]
    hN = ts[0]
    wN = ts[1]
    image = np.empty((tileSize*hN, tileSize*wN, 3))

    for i in range(hN):
        for j in range(wN):
            image[i*tileSize:(i+1)*tileSize, j*tileSize:(j+1)*tileSize, :] = tileSet[i, j, :, :, :]
    return image


def spinImage(image, ind, tileSize=128):
    # Smooth tiling of classifier
    rng = np.random.default_rng(137 + ind)
    spins = rng.integers(low=-tileSize, high=tileSize, size=2)
    imageSpan = np.roll(np.roll(image, spins[0], axis=0), spins[1], axis=1)
    return imageSpan, spins[0], spins[1]


def unspinImage(image, spinX, spinY):
    # undo spinImage to correct result
    return np.roll(np.roll(image, -spinX, axis=0), -spinY, axis=1)


def computeDamageArray(image, model, tileSize):
    # Compute damage score on image in tiled sense
    imgPad, bottomPad, rightPad = padImage(image, tileSize=tileSize)
    tiles = imageToTiles(imgPad, tileSize=tileSize)
    
    # Reshape tiles to be numberTiles by 128x128x3
    arrayTiles = np.reshape(tiles, (-1, tiles.shape[2], tiles.shape[3], tiles.shape[4]))
    # Run the model
    predictions = model.predict(arrayTiles)
    
    # flip to damage score
    predictions = 1-predictions
    tilesScore = np.reshape(predictions, tiles.shape[0:2])
    

    # Enlarge result of classifier to match the scale of the input image.
    padImageScore = cv2.resize(tilesScore, (0, 0), fx=tileSize, fy=tileSize, interpolation=cv2.INTER_NEAREST) 
    
    # Remove padding the padding applied to int input image so the resolution (x*y) are equal
    imageScore = unPadImage(padImageScore, bottomPad, rightPad)
    return imageScore


def computeDamageMap(image, model, tileSize=128, spin=0):
    # Estimate damage level in image. The model uses a fixed tileSize of 128x128 pixels
    # A score of 1 is high likelihood of damage and score of 0 is a low likelihood of damage.

    s = image.shape
    if spin == 0:
        imageScore = computeDamageArray(image, model, tileSize)
    else:
        if spin == 1:
            # default to 5 spins
            defSpins = 5
            imageScores = np.empty((defSpins, s[0], s[1]))
            for i in range(defSpins):
                spunImage, x, y = spinImage(image, i, tileSize=128)
                spunImageScore = computeDamageArray(spunImage, model, tileSize)
                imageScores[i, :, :] = unspinImage(spunImageScore, x, y)
            imageScore = np.mean(imageScores, axis=0)
        else:
            imageScores = np.empty((spin, s[0], s[1]))
            for i in range(spin):
                spunImage, x, y = spinImage(image, i, tileSize=128)
                spunImageScore = computeDamageArray(spunImage, model, tileSize)
                imageScores[i, :, :] = unspinImage(spunImageScore, x, y)
            imageScore = np.mean(imageScores, axis=0)
    return imageScore


def setupBuildingDamageModel():
    # Return model that I trained earlier
    
    image_size = (128, 128)
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    
    # issue with keras vs keras-core

    with (open("weightsFinal.pickle", "rb")) as openfile:
        while True:
            try:
                weights = pickle.load(openfile)
            except EOFError:
                break

    model.set_weights(weights)
    # For some reason load_model isn't working so I had to do this
    # nonsense. It should work like this
    # model = keras.load_model("save_at_25.keras")

    return model


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # I put together this CNN
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    
    size = 8
    
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    activation = "sigmoid"
    units = 1
    # help avoid overfitting
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

