import os
import PIL
import PIL.Image
from numpy import *
import neuralnetwork
import functions

1.2

"""""
[[x1y1, x2y1, x3y1, x4y1],
 [x1y2, x2y2, x3y2, x4y2],
 [x1y3, x2y3, x3y3, x4y3]]
"""""

def getinput(folder="fulltest", category="HELLO", png="HELLO_0.png"):
    img = PIL.Image.open(folder+"/"+category+"/"+png)
    img.convert("L")
    imgwidth, imgheight = img.size
            
    return array([array([float(sum(img.getpixel((x, y)))/3) for x in range(0, imgwidth)]) for y in range(0, imgheight)])

def splitinput(input, a=10): #a offers leeway for very light pixels in a space
    sidelen, input, output, charstart = len(input), input.T, [], 0

    for i in range(len(input)):
        if ((255 * sidelen - input[i].sum()) < a):
            if ((i-charstart) < 2): charstart = i
                
            else: #if a distance between two spaces is detected:
                margin = (sidelen - i + charstart + 1) / 2
                charpart = array([array([255] * sidelen)] * max(int(floor(margin)), 0) + [input[i] for i in range(charstart+1, i)] + [array([255] * sidelen)] * max(int(ceil(margin)), 0)).T
                #print(charpart)
                formattedcharpart = functions.minmaxnormalization(charpart, minmax=[0, 255]).flatten("F")
                nnout = neuralnetwork.applyactivation(neuralnetwork.main().run(formattedcharpart, istraining=False)[-1], True)
                charinterpretation = neuralnetwork.interpretcharacter(nnout)
                print(charinterpretation)
                output += [charinterpretation]
                img = PIL.Image.fromarray(charpart)
                img = img.convert("L")
                img.save(f"fulltestcache/{charinterpretation}.png")
                charstart = i
                
    return print(output)
        

#splitinput(getinput(folder="fulltest", category="HELLO", png="HELLO_0.png"))
#splitinput(getinput(folder="fulltest", category="HELLO", png="HELLO_1.png"))
#splitinput(getinput(folder="fulltest", category="HELLO", png="smallernnwriting.png"))
splitinput(getinput(folder="fulltest", category="HELLO", png="sff.png"))
#splitinput(getinput(folder="fulltest", category="HELLO", png="snnw2.png"))