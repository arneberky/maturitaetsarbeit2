import os
import PIL
import PIL.Image
from numpy import *

#only for black / grey / white pngs
def translate(folder):
    output = {}
    
    contents = os.listdir(folder)
    if ".DS_Store" in contents:
        contents.remove(".DS_Store")
    
    for category in contents:
        categoryoutput = []
        
        pngs = os.listdir(folder+"/"+category)
        if ".DS_Store" in pngs:
            pngs.remove(".DS_Store") #file automatically added from apple
        
        for png in pngs:
            img = PIL.Image.open(folder+"/"+category+"/"+png)
            img.convert("RGB")
            imgwidth, imgheight = img.size
            
            categoryoutput += [array([float(img.getpixel((x, y))) for x in range(0, imgwidth) for y in range(0, imgheight)])]
            
        output.update({category : categoryoutput})
    
    return output

def connect(inoutdict={}, folder="learn"):
    if inoutdict == {}:
        raise RuntimeError
    
    translation = translate(folder=folder)
    outputin, outputout = [], []
    
    for category in translation:
        outputin += [png for png in translation[category]]
        outputout += [inoutdict[category] for png in translation[category]]
    
    return {"input" : outputin, "output" : outputout}

def sketchy8x8print(output):
    for category in output:
        for item in output[category]:
            for i in range(8):
                print(item[8*i:8*(i+1)])
  
#connect(folder="test")