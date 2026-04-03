from sys import *
from functions import *
import pngconverter
from numpy import *

setrecursionlimit(500000)
random.seed(5)

leakyrelu = leakyreluactivation()

characters = [chr(i) for i in range(65, 91)]
inoutdict = {characters[i]: array(i * [0] + [1] + (len(characters) - i - 1) * [0]) for i in range(len(characters))}

def interpretcharacter(neuralnetworkoutput):
    return characters[argmax(neuralnetworkoutput)]

def randomvalue(target, fanin=1, fanout=1):
    randomranges = {"Weights": sqrt(2 / fanin), "Biases": 0.01}
    return random.uniform(-randomranges[target], randomranges[target])

def applyactivation(x, lastlayer: bool):
    return softmax(x) if lastlayer else leakyrelu.function(x)

def applyderivative(x, lastlayer: bool):
    return 1 if lastlayer else leakyrelu.derivative(x)

class NeuralNetwork():
    def __init__(self, architecture, dropoutrates=None):
        self.architecture = array([layer for layer in architecture])
        self.model = array([
            {
                "Weights": array([
                    array([randomvalue("Weights", architecture[layer-1]) for _ in range((architecture + [0])[layer-1])])
                    for _ in range(architecture[layer])
                ]),
                "Biases": array([randomvalue("Biases") for _ in range(architecture[layer])])
            }
            for layer in range(len(architecture))
        ])
        self.modellen = len(self.model)

        if dropoutrates is None:
            self.dropoutrates = [0] * self.modellen
        else:
            assert len(dropoutrates) == self.modellen
            self.dropoutrates = dropoutrates

    def printmodel(self):
        for layer in range(self.modellen):
            print(f"\nLayer {layer+1}:\n\nWeights:\n{self.model[layer]['Weights']}\n\nBiases:\n{self.model[layer]['Biases']}\n")

    def savemodel(self):
        with open("neuralnetwork.npy", "wb") as npyfile:
            save(npyfile, self.model)

    def getmodel(self):
        with open("neuralnetwork.npy", "rb") as npyfile:
            self.model = load(npyfile, allow_pickle=True)

    def savedata(self, lort):
        if lort == "learn":
            learn = pngconverter.connect(inoutdict=inoutdict, folder="learn")
            with open("learndata.npy", "wb") as npyfile:
                save(npyfile, array(minmaxnormalization(learn["input"], minmax=[0, 255])))
                save(npyfile, array(learn["output"]))
        elif lort == "test":
            test = pngconverter.connect(inoutdict=inoutdict, folder="test")
            with open("testdata.npy", "wb") as npyfile:
                save(npyfile, array(minmaxnormalization(test["input"], minmax=[0, 255])))
                save(npyfile, array(test["output"]))

    def getdata(self, lort):
        if lort == "learn":
            with open("learndata.npy", "rb") as npyfile:
                return load(npyfile, allow_pickle=True), load(npyfile, allow_pickle=True)
        elif lort == "test":
            with open("testdata.npy", "rb") as npyfile:
                return load(npyfile, allow_pickle=True), load(npyfile, allow_pickle=True)

    def applydropout(self, nodes, dropoutrate, istraining):
        if not istraining or dropoutrate == 0:
            return nodes
        mask = random.binomial(1, 1 - dropoutrate, size=nodes.shape)
        scale = 1.0 / (1.0 - dropoutrate)
        return (nodes * mask) * scale

    def train(self, input, expected, minibatchsize=128, epochs=1, learningrate=0.0001, l2lambda=0.0003):
        random.seed(6)
        random.shuffle((input := input * epochs))
        random.seed(6)
        random.shuffle((expected := expected * epochs))

        self.xpoints1, self.ypoints1 = array([i for i in range(int((len(input)//minibatchsize)/10))]), zeros(int((len(input)//minibatchsize)/10))
        self.ypoints1[0] = 100

        self.xpoints2, self.ypoints2 = array([i for i in range(int((len(input)//minibatchsize)/10))]), zeros(int((len(input)//minibatchsize)/10))
        self.ypoints2[0] = 10
        
        for minibatch in range(len(input)//minibatchsize):
            modelcache = self.model.copy()
            for sample in range(minibatchsize):
                inputid = minibatch * minibatchsize + sample
                output = self.run(input[inputid], istraining=True)
                activated_output = applyactivation(output[-1], True)
                errors = [array([0.0 for _ in self.model[layer]["Biases"]]) for layer in range(self.modellen - 1)] + [activated_output - expected[inputid]]

                for layer in range(self.modellen - 1, -1, -1):
                    for node in range(self.architecture[layer]):
                        nodederivative = applyderivative(output[layer][node], layer == self.modellen - 1) * errors[layer][node]
                        delta = nodederivative * learningrate
                        modelcache[layer]["Biases"][node] -= delta

                        if layer != 0:
                            for prevnode in range(self.architecture[layer - 1]):
                                l2penalty = l2lambda * self.model[layer]["Weights"][node][prevnode] / minibatchsize
                                errors[layer - 1][prevnode] += self.model[layer]["Weights"][node][prevnode] * nodederivative
                                activation = applyactivation(output[layer - 1][prevnode], False)
                                modelcache[layer]["Weights"][node][prevnode] -= (activation * delta + l2penalty)

            self.model = modelcache

        
            if (minibatch+1) % 10 == 0:
                statisticerror = sum(abs(errors[-1]))
                self.ypoints2[int(minibatch/10)] = statisticerror

                stdout.write("\rTraining... : [{:{}}] {:>3}% | Total Error of Minibatch : {:{}}".format(
                    '█' * int((minibatch + 1)/(len(input)//minibatchsize)*30),
                    30,
                    int(100.0*(minibatch + 1)/(len(input)//minibatchsize)),
                    round(statisticerror, 8),
                    12))
                stdout.flush()

                testdata = neuralnetwork.getdata("test")
                # Test durchführen
                self.test(testdata[0], testdata[1], minibatch)

                # Werte in Textdatei schreiben
                accuracy = self.ypoints1[int(minibatch/10)]
                with open("traininglog.txt", "a") as log_file:
                    log_file.write(f"Minibatch: {minibatch+1}, Accuracy: {accuracy}%, Error: {statisticerror}\n")
            
        print("\n")

    def test(self, input, expected, minibatch=None):
        correctoutput = 0
        for i in range(len(input)):
            output = self.run(input[i], istraining=False)[-1]
            activated_output = applyactivation(output, True)
            outputbool = str(bool(interpretcharacter(expected[i]) == interpretcharacter(activated_output)))
            correctoutput += int(outputbool == "True")
            expectedchar, actualchar = interpretcharacter(expected[i]), interpretcharacter(activated_output)

            if minibatch is not None:
                self.ypoints1[int(minibatch/10)] = round(correctoutput / len(input) * 100, 1)
                
            else:
                print("Test {:{}} -> Expected: {:<{}} | Output: {:<{}} | Correct?: {:<{}}".format(
                    str(i + 1), 3, expectedchar, 3, actualchar, 3, outputbool, 1))
        print(f"results are {round(correctoutput / len(input) * 100, 1)} percent correct")

    def run(self, input, istraining=False):
        nodes = [zeros(layersize) for layersize in self.architecture]
        nodes[0] = array(input) + self.model[0]["Biases"]

        for layer in range(1, self.modellen - 1):
            prevactivations = applyactivation(nodes[layer - 1], False)
            prevactivations = self.applydropout(prevactivations, self.dropoutrates[layer - 1], istraining)
            nodes[layer] = dot(prevactivations, self.model[layer]["Weights"].T) + self.model[layer]["Biases"]

        finalactivations = applyactivation(nodes[-2], False)
        finalactivations = self.applydropout(finalactivations, self.dropoutrates[-2], istraining)
        nodes[-1] = dot(finalactivations, self.model[-1]["Weights"].T) + self.model[-1]["Biases"]

        return nodes

def main():
    global neuralnetwork
    neuralnetwork = NeuralNetwork(
        architecture=[64, 128, 64, 26],
        dropoutrates=[0.0, 0.25, 0.1, 0.0]
    )
    neuralnetwork.getmodel()
    return neuralnetwork

if __name__ == "__main__":
    main()
    #neuralnetwork.savedata("learn")
    #neuralnetwork.savedata("test")
    #traindata = neuralnetwork.getdata("learn")
    #neuralnetwork.train(traindata[0], traindata[1])
    testdata = neuralnetwork.getdata("test")
    neuralnetwork.test(testdata[0], testdata[1])
    #neuralnetwork.savemodel()
