import numpy as np
import matplotlib.pyplot as plt
from ultralyticsplus import render_result
from McAteer_functions import *




def _main():
    """Main entry point"""
    
    # Pretrained yolo building detector
    modelBuildingDetection = setupBuildingDetectionModel()
    # Bespoke damage model (trained on hurrican data)
    modelBuildingDamage = setupBuildingDamageModel()
    # Load images and do pre processing
    images, numImages = loadImages('data')

    # Just loop over each image. Image 5 is much higher resolution
    for ind in range(numImages):
        results = modelBuildingDetection.predict(images[ind], conf=0.02)
        print('Yolo model computed')

        # this damage mapping worked much better in the training data
        # (https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage/)
        # Not all of the damage is hurricane damage, this can just be retrained with more
        # relevant training data for an improved result
        damageMap = computeDamageMap(images[ind], modelBuildingDamage, tileSize=128, spin=1)
        print('Damage model computed')
        # These polygon detections could be written to a geojson file
        render = render_result(model=modelBuildingDetection, image=images[ind], result=results[0])
        # Burn PNGs of the analysis
        showDamageMap(images[ind], render, damageMap, showPlots=0, savePlots=1, saveFilename='analysis_' + str(ind) + '.png', dpi=900)
    


if __name__ == "__main__":
    # This only takes about 1 minute on my laptop (with an RTX 4060). I hope you have a GPU.
    _main()