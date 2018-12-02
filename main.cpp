//Elliott Dobbs
//10 Feb. 2018
//Extension from Artificial Intelligence class project

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <bitset>
#include <math.h>
#include <cmath>

#include "NeuralNetwork.cpp"
using namespace std;

vector<double> sigmoid;
vector<double> hyperbolicTan;
double alpha;

//Link structure to represent weighted links between nodes
struct Link{
    int sendingLink;
    int recievingLink;
    double weight;
};

//Node structure to represent a node in the Nerual Network
struct Node{
    vector<Link> inputLinks;
    vector<Link> outputLinks;
    double outputActivation;
    double delta;
};

//Nerual network structure
struct NeuroNetwork{
    vector< vector<Node> > layers;
};

//Algorithm to caclulate the manhatten distance based on a 240 vector input state
int manhattenDistance(vector<double> input){
    
    
    int gameState[4][4] = {{16, 16, 16, 16}, {16, 16, 16, 16}, {16, 16, 16, 16}, {16, 16, 16, 16}};
    
    //Getting he current gamestate into the gamestate array
    for (int i = 0; i < input.size(); ++i){
        
        if (input[i] == 1){
            for (int gameIter = 1; gameIter < 16; ++gameIter){
                if (gameState[gameIter / 4][gameIter % 4] == 16){
                    gameState[gameIter / 4][gameIter % 4] = 15 - (i % 16);
                    break;
                }
            }
        }
    }
    
    int check[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    //Finding which tile belongs in [0][0]
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            if (i == 0 && j == 0)
                continue;
            else{
                check[gameState[i][j]] = 1;
            }
        }
    }
    
    //Setting tile 0 to the appropriate tile number
    for (int i = 0; i < 16; ++i){
        if (check[i] == 0){
            gameState[0][0] = i;
            break;
        }
    }
    
    int mDistance = 0, igoal, jgoal;
    int goalState[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 0}};
    
    //Calculating the manhatten distance
    for (int i = 0; i < 4; ++i){
        for (int j = 0; j < 4; ++j){
            
            //Getting where the tile is supposed to be
            for (int x = 0; x < 4; ++x){
                for(int y = 0; y < 4; ++y){
                    if (goalState[x][y] == gameState[i][j]){
                        igoal = x;
                        jgoal = y;
                        break;
                    }
                }
            }
            
            //Calculating the distance of the tile from its goal location
            int xdistance = abs(igoal-i);
            int ydistance = abs(jgoal-j);
            
            //If the tile is not 0, add the distance to the manhatten distance
            if (gameState[i][j] != 0)
                mDistance += xdistance + ydistance;
        }
    }
    
    return mDistance;
}

//Function to generate and fill the sigmpoid vector with values from input -5 to 5
void generateSigmoid(){
    double x, sig;
    
    for (int i = 0; i < 10000; ++i){
        x = -5.0 + ((double)i/1000.0);
        sig = 1.0/(1.0+exp(-x));
        sigmoid.push_back(sig);
    }
}

//Function to generate and fill the hyperbolic tangent vector with values from input -5 to 5
void generateHyperbolicTangent(){
    double x, hyp;
    
    for (int i = 0; i < 10000; ++i){
        x = -5.0 + ((double)i/1000.0);
        hyp = (exp(x) - exp(-x))/(exp(x) + exp(-x));
        hyperbolicTan.push_back(hyp);
    }
}

//Function to create the neural network structure.  The variables inside can be changed to desired input. variables like the hidden layers, their sizes, etc.
NeuroNetwork createNetwork(int inputNodes, int outputNodes){
    
    NeuroNetwork net;
    Node n;
    n.outputActivation = 0.0;
    n.delta = 0.0;
    int inputLayerNodes = inputNodes;
    int outputLayerNodes = outputNodes;
    vector<int> hiddenLayerNodes;
    hiddenLayerNodes.push_back(120);
    hiddenLayerNodes.push_back(60);
    
    vector<Node> tempLayer;
    
    //creating the input layer
    for (int i = 0; i < inputLayerNodes; ++i){
        tempLayer.push_back(n);
    }
    net.layers.push_back(tempLayer);
    tempLayer.clear();
    
    //creating the hidden layers
    for(int j = 0; j < hiddenLayerNodes.size(); ++j){
        for (int i = 0; i < hiddenLayerNodes[j]; ++i){
            tempLayer.push_back(n);
        }
        net.layers.push_back(tempLayer);
        tempLayer.clear();
    }
    
    //creating the output layer
    for (int i = 0; i < outputLayerNodes; ++i){
        tempLayer.push_back(n);
    }
    net.layers.push_back(tempLayer);
    tempLayer.clear();
    
    /////////Creating the links vectors for each node
    Link tempLink;
    tempLink.weight = 0.0;
    vector<Link> tempLinkVector;
    
    //making the outputLinks vector for the input layer
    for (int i = 0; i < net.layers[1].size(); ++i){
        tempLink.recievingLink = i;
        tempLinkVector.push_back(tempLink);
    }
    //setting outputLinks vector
    for (int i = 0; i < inputLayerNodes; ++i){
        for (int j = 0; j < tempLinkVector.size(); ++j){
            tempLinkVector[j].sendingLink = i;
        }
        net.layers[0][i].outputLinks = tempLinkVector;
    }
    
    //making the Links vectors for the hidden layers
    for (int layer = 1; layer < net.layers.size() - 1; ++layer){
        
        tempLinkVector.clear();
        
        //creating the inputLinks vector
        for (int i = 0; i < net.layers[layer-1].size(); ++i){
            tempLink.sendingLink = i;
            tempLinkVector.push_back(tempLink);
        }
        //setting the inputLinks
        for (int i = 0; i < net.layers[layer].size(); ++i){
            for (int j = 0; j < tempLinkVector.size(); ++j){
                tempLinkVector[j].recievingLink = i;
            }
            net.layers[layer][i].inputLinks = tempLinkVector;
        }
        tempLinkVector.clear();
        
        //creating the outputLinks vector
        for (int i = 0; i < net.layers[layer + 1].size(); ++i){
            tempLink.recievingLink = i;
            tempLinkVector.push_back(tempLink);
        }
        //setting outputLinks vector
        for (int i = 0; i < net.layers[layer].size(); ++i){
            for (int j = 0; j < tempLinkVector.size(); ++j){
                tempLinkVector[j].sendingLink = i;
            }
            net.layers[layer][i].outputLinks = tempLinkVector;
        }
    }
    
    //creating inputLinks for outputLayer
    tempLinkVector.clear();
    
    //creating the inputLinks vector
    for (int i = 0; i < net.layers[net.layers.size() - 2].size(); ++i){
        tempLink.sendingLink = i;
        tempLinkVector.push_back(tempLink);
    }
    //setting the inputLinks
    for (int i = 0; i < net.layers[net.layers.size() - 1].size(); ++i){
        for (int j = 0; j < tempLinkVector.size(); ++j){
            tempLinkVector[j].recievingLink = i;
        }
        net.layers[net.layers.size() - 1][i].inputLinks = tempLinkVector;
    }
    
    return net;
}

//function to create a 240 vector from a given uint64_t.  The uint64_t is obtained from the input file, then passed here to parse into something we can pass to the propigation algorithm.
vector<double> getInputVector(uint64_t input){
    
    //Converting the input to a bitset, then to a string which represents the input state in binary
    vector<double> result;
    bitset<64> x(input);
    string howdy = x.to_string();
    int missingNumber[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    //Iterates through the binary string, adding each number as a demultiplexed version into the results vector
    for (int i = 1; i < 16; ++i){
        
        string test = howdy.substr(i*4, 4);
        
        if(test == "0000"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            
            missingNumber[0] = 1;
        }
        else if(test == "0001"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            
            missingNumber[1] = 1;
        }
        else if(test == "0010"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[2] = 1;
        }
        else if(test == "0011"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[3] = 1;
        }
        else if(test == "0100"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[4] = 1;
        }
        else if(test == "0101"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[5] = 1;
        }
        else if(test == "0110"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[6] = 1;
        }
        else if(test == "0111"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[7] = 1;
        }
        else if(test == "1000"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[8] = 1;
        }
        else if(test == "1001"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[9] = 1;
        }
        else if(test == "1010"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[10] = 1;
        }
        else if(test == "1011"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[11] = 1;
        }
        else if(test == "1100"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[12] = 1;
        }
        else if(test == "1101"){
            result.push_back(0);
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[13] = 1;
        }
        else if(test == "1110"){
            result.push_back(0);
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[14] = 1;
        }
        else if(test == "1111"){
            result.push_back(1);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            result.push_back(0);
            
            missingNumber[15] = 1;
        }
        
    }
    
    return result;
}

//Function demultiplex an integer into a binary representation to be used as an expected output for the propigation algorithm
vector<double> getOutputVector(int input){
    
    vector<double> result;
    
    for (int i = 0; i < 29; ++i){
        if (input == i)
            result.push_back(1.0);
        else
            result.push_back(0.0);
    }
    
    return result;
}

//Function that iterates through each of the Links in a network, setting them to a random number between -1 and 1 with 0.001 percision
NeuroNetwork makeWeights(NeuroNetwork net){
    
    double random;
    
    for (int layer = 0; layer < net.layers.size(); ++layer){
        
        //set output weights
        if (layer != net.layers.size() - 1){
            for (int node = 0; node < net.layers[layer].size(); ++node){
                for (int nodeLink = 0;
                     nodeLink < net.layers[layer][node].outputLinks.size();
                     ++nodeLink){
                    
                    random = (double)(rand() % 2000 - 1000)/1000;
                    net.layers[layer][node].outputLinks[nodeLink].weight = random;
                }
            }
        }
        
        //set input weights based on the output link from the previous layer
        if (layer != 0){
            for (int node = 0; node < net.layers[layer].size(); ++node){
                for (int nodeLink = 0;
                     nodeLink < net.layers[layer][node].inputLinks.size();
                     ++nodeLink){
                    
                    net.layers[layer][node].inputLinks[nodeLink].weight = net.layers[layer - 1][net.layers[layer][node].inputLinks[nodeLink].sendingLink].outputLinks[node].weight;
                }
            }
        }
    }
    
    return net;
}

//Main function to teach the neural network.  Follows the pseudocode from toe book.  Takes in a set of example inputs and expected outputs, and a blank network.  Calls the makeWeights function to assign random weights to the links.  Then for each input state, propigates the input forward to the output layer, calculates the deltas, and propigates backwards calculating delta errors.  Finally updates the weights based on the errors and moves on to the next example.
NeuroNetwork backPropLearning(vector< vector<double> > inputActivations,
                              vector< vector<double> > outputActivations,
                              NeuroNetwork network,
                              int learningCycles){
    
    //Generate random link weights
    network = makeWeights(network);
    
    //Learning Cycles
    for (int cycle = 0; cycle < learningCycles; ++ cycle){
        
        //for each example...
        for (int example = 0; example < inputActivations.size(); ++example){
            if (example % 1000 == 999)
                cout << "Cycle: " << cycle << ".  Propigated " << example + 1 << " input examples..." << endl;
            
            //Set input activations
            for (int inputLayerNode = 0;
                 inputLayerNode < network.layers[0].size();
                 ++inputLayerNode){
                
                network.layers[0][inputLayerNode].outputActivation = inputActivations[example][inputLayerNode];
            }
            
            //Propigate input forward to output; layer by layer
            for (int layer = 1; layer < network.layers.size(); ++layer){
                for (int node = 0; node < network.layers[layer].size(); ++node){
                    
                    //Get weighted sum
                    double weightedSum = 0.0;
                    for (int inputLink = 0;
                         inputLink < network.layers[layer][node].inputLinks.size();
                         ++inputLink){
                        
                        weightedSum +=  network.layers[layer][node].inputLinks[inputLink].weight *
                        network.layers[layer - 1][network.layers[layer][node].inputLinks[inputLink].sendingLink].outputActivation;
                    }
                    
                    //Putting weighted sum into Sigmoid function
                    //This next line decodes the weighted sum into an integer to use as input to the sigmoid (or hyperbolic tangent) vector
                    int sum = (int)((weightedSum + 5) * 1000);
                    
                    //Use Sigmoid for last layer
                    if (layer == network.layers.size() - 1){
                        if (sum < 0)
                            network.layers[layer][node].outputActivation = 0.0;
                        else if (sum >= 10000)
                            network.layers[layer][node].outputActivation = sigmoid[9999];
                        else
                            network.layers[layer][node].outputActivation = sigmoid[sum];
                    }
                    
                    //Use hyperbolic tangent for all other layers
                    else{
                        if (sum < 0)
                            network.layers[layer][node].outputActivation = -1;
                        else if (sum >= 10000)
                            network.layers[layer][node].outputActivation = hyperbolicTan[9999];
                        else
                            network.layers[layer][node].outputActivation = hyperbolicTan[sum];
                    }
                    
                }
            }
            
            //Propigate deltas backwards
            for (int node = 0;
                 node < network.layers[network.layers.size() - 1].size();
                 ++node){
                
                //Calculating error
                double y_error = outputActivations[example][node] - network.layers[network.layers.size() - 1][node].outputActivation;
                if (y_error > 0)
                    y_error *= 2;
                
                //Calculating delta
                network.layers[network.layers.size() - 1][node].delta = (network.layers[network.layers.size() - 1][node].outputActivation) * (1 - network.layers[network.layers.size() - 1][node].outputActivation) * y_error;
            }
            
            //For the rest of the layers, caclulate deltas based on next layer deltas
            double summedWeightedJs;
            for (int layer = (network.layers.size() - 2); layer >= 0; --layer){
                for (int node = 0; node < network.layers[layer].size(); ++node){
                    
                    summedWeightedJs = 0.0;
                    for (int deltaJs = 0; deltaJs < network.layers[layer + 1].size(); ++ deltaJs){
                        summedWeightedJs += network.layers[layer][node].outputLinks[deltaJs].weight * network.layers[layer + 1][deltaJs].delta;
                    }
                    
                    network.layers[layer][node].delta = (1 - (network.layers[layer][node].outputActivation * network.layers[layer][node].outputActivation)) * summedWeightedJs;
                }
            }
            
            //Update weights
            for (int layer = 0; layer < network.layers.size(); ++layer){
                
                //Update output weights
                if (layer != network.layers.size() - 1){
                    for (int node = 0; node < network.layers[layer].size(); ++node){
                        for (int nodeLink = 0;
                             nodeLink < network.layers[layer][node].outputLinks.size();
                             ++nodeLink){
                            
                            network.layers[layer][node].outputLinks[nodeLink].weight += alpha * network.layers[layer][node].outputActivation * network.layers[layer + 1][network.layers[layer][node].outputLinks[nodeLink].recievingLink].delta;
                        }
                    }
                }
                
                //set input weights based on the output link from the previous layer
                if (layer != 0){
                    for (int node = 0; node < network.layers[layer].size(); ++node){
                        for (int nodeLink = 0;
                             nodeLink < network.layers[layer][node].inputLinks.size();
                             ++nodeLink){
                            
                            network.layers[layer][node].inputLinks[nodeLink].weight = network.layers[layer - 1][network.layers[layer][node].inputLinks[nodeLink].sendingLink].outputLinks[node].weight;
                        }
                    }
                }
            }
        }
    }
    
    return network;
}

//Function to test the network.  Propigates a given input example throught eh network and returns the network.
NeuroNetwork progigate(vector<double> inputActivations, NeuroNetwork network){
    
    //Set input activations
    for (int inputLayerNode = 0;
         inputLayerNode < network.layers[0].size();
         ++inputLayerNode){
        
        network.layers[0][inputLayerNode].outputActivation = inputActivations[inputLayerNode];
    }
    //Propigate input forward to output
    for (int layer = 1; layer < network.layers.size(); ++layer){
        for (int node = 0; node < network.layers[layer].size(); ++node){
            
            //Get weighted sum
            double weightedSum = 0.0;
            for (int inputLink = 0;
                 inputLink < network.layers[layer][node].inputLinks.size();
                 ++inputLink){
                
                weightedSum +=  network.layers[layer][node].inputLinks[inputLink].weight *
                network.layers[layer - 1][network.layers[layer][node].inputLinks[inputLink].sendingLink].outputActivation;
            }
            
            //Use sigmoid if its the last layer
            int sum = (int)((weightedSum + 5) * 1000);
            if (layer == network.layers.size() - 1){
                if (sum < 0)
                    network.layers[layer][node].outputActivation = 0.0;
                else if (sum >= 10000)
                    network.layers[layer][node].outputActivation = sigmoid[9999];
                else
                    network.layers[layer][node].outputActivation = sigmoid[sum];
            }
            //Use hyperbolic tangent if its any other layer
            else{
                if (sum < 0)
                    network.layers[layer][node].outputActivation = -1;
                else if (sum >= 10000)
                    network.layers[layer][node].outputActivation = hyperbolicTan[9999];
                else
                    network.layers[layer][node].outputActivation = hyperbolicTan[sum];
            }
            
        }
    }
    
    return network;
}

int main(){
    
    ifstream myfile[29];
    uint64_t howdy;
    vector< vector<double> > exampleInputActivations;
    vector< vector<double> > exampleOutputActivations;
    int randomChance, randomState;
    
    alpha = 0.1;
    
    int testNumber, maxTestStateNumber, learningCycles, inNodes, outNodes;
    cout << "Enter the amount of input nodes: ";
    cin >> inNodes;
    cout << "Enter the amount of output nodes: ";
    cin >> outNodes;
    cout << "Enter an amount of test files to get: ";
    cin >> testNumber;
    cout << "Enter the max state number to train on: ";
    cin >> maxTestStateNumber;
    cout << "Enter the number of learning cycles: ";
    cin >> learningCycles;
    
    //Opening the input files.  The commented line is for using the network on my local machine, with my local data
    for (int i = 0; i < 29; ++i){
        //string fileName = "/pub/faculty_share/daugher/datafiles/data/" + to_string(i) + "states.bin";
        string fileName = "Data/" + to_string(i) + "states.bin";
        myfile[i].open(fileName, ios::binary);
    }
    
    //Getting the input data.  Data is obtained by iterating through the files geting one random state from each of the files in the range from 0 to the given max file number.  This is done until the given amount of test states is achieved.
    cout << "Getting input data..." << endl;
    int inputStateIter = 0;
    for (int i = 1; i < testNumber; ++i){
        
        //This forces the loop to go through the current file until a state is found.  Need this since there is a random chance each state is chosen.
        bool found = false;
        while (found == false){
            
            //Iterates through each state of a file
            while (myfile[inputStateIter].read(reinterpret_cast<char *>(&howdy), sizeof(howdy))){
                
                randomChance = rand() % 100;
                
                //Ranadom chance that a file is selected OR if the file is the 0 state file, there is only one state anyways
                if (randomChance == 1 || inputStateIter == 0){
                    
                    //Parsing input
                    vector<double> inputActivation = getInputVector(howdy);
                    exampleInputActivations.push_back(inputActivation);
                    
                    //Getting output activations
                    vector<double> outputActivation = getOutputVector(inputStateIter);
                    exampleOutputActivations.push_back(outputActivation);
                    
                    found = true;
                    break;
                }
            }
            
            //Resetting the ifstream object
            myfile[inputStateIter].clear();
            myfile[inputStateIter].seekg(0, myfile[inputStateIter].beg);
        }
        
        //Iterates to the next state file
        ++inputStateIter;
        if (inputStateIter > maxTestStateNumber)
            inputStateIter = 0;
    }
    
    //Closes all the ifstream objects
    for (int i = 0; i < 29; ++i){
        myfile[i].close();
    }
    
    cout << "Input Obtained" << endl;
    
    //generate the sigmoid table
    generateSigmoid();
    cout << "Generated Sigmoid Values..." << endl;
    
    //generate the hyperbolic tangent table
    generateHyperbolicTangent();
    cout << "Generated Hyperbolic Tangent Values..." << endl;
    
    //Creating the base network
    NeuroNetwork network = createNetwork(240, 29);
    cout << "Created Neural Network..." << endl;
    
    //Performing the back-Propigation
    cout << "Training network..." << endl;
    NeuroNetwork resultingNetwork = backPropLearning(exampleInputActivations,
                                                     exampleOutputActivations,
                                                     network, learningCycles);
    cout << "Neural Network Trained..." << endl << endl;
    
    //User choosing input files to get a random state to propigate
    int continueInput = 1, stateChosen, manDist;
    vector<double> userInputActivation;
    
    while (continueInput == 1) {
        cout << "Input an integer (0-28) for which state pool you want\nto choose from: ";
        cin >> stateChosen;
        
        //gets random state from the given file
        while (userInputActivation.size() == 0){
            
            //string fileName = "/pub/faculty_share/daugher/datafiles/data/" + to_string(stateChosen) + "states.bin";
            string fileName = "Data/" + to_string(stateChosen) + "states.bin";
            myfile[0].open(fileName, ios::binary);
            
            while (myfile[0].read(reinterpret_cast<char *>(&howdy), sizeof(howdy))){
                
                randomChance = rand() % 1000^(stateChosen+1);
                
                if (randomChance == 1 || stateChosen == 0){
                    
                    //Parsing input
                    userInputActivation = getInputVector(howdy);
                    manDist = manhattenDistance(userInputActivation);
                    break;
                }
            }
            myfile[0].close();
        }
        
        //Propigates the random input state through the network
        resultingNetwork = progigate(userInputActivation, resultingNetwork);
        
        //Gets the results from the propigation.  Prints each output node and its activation.
        double max = -10.0;
        int maxIter;
        for (int i = 0; i < 29; ++i){
            if (resultingNetwork.layers[resultingNetwork.layers.size() - 1][i].outputActivation > max){
                max = resultingNetwork.layers[resultingNetwork.layers.size() - 1][i].outputActivation;
                maxIter = i;
            }
            cout << i << " : " << resultingNetwork.layers[resultingNetwork.layers.size() - 1][i].outputActivation << endl;
        }
        //Displays results
        cout << endl << "Max activation was : " << max << endl;
        cout << "At: " << maxIter << endl;
        cout << "Manhatten Distance: " << manDist << endl;
        cout << "result - Manhatten Distance: " << maxIter - manDist << endl;
        cout << endl << endl;
        cout << "continue? (1/0) : ";
        cin >> continueInput;
        userInputActivation.clear();
    }
    
    //User choosing whole input files to get data from
    continueInput = 1;
    int statesImproved;
    vector<int> differences, results;
    
    while (continueInput == 1) {
        cout << "Input an integer (0-28) for which state pool you want: ";
        cin >> stateChosen;
        
        
        //string fileName = "/pub/faculty_share/daugher/datafiles/data/" + to_string(stateChosen) + "states.bin";
        string fileName = "Data/" + to_string(stateChosen) + "states.bin";
        myfile[0].open(fileName, ios::binary);
        
        //Propigates each state in the given file through the network.
        while (myfile[0].read(reinterpret_cast<char *>(&howdy), sizeof(howdy))){
            
            //Propigates and get the manhatten distance of each state.
            userInputActivation = getInputVector(howdy);
            manDist = manhattenDistance(userInputActivation);
            resultingNetwork = progigate(userInputActivation, resultingNetwork);
            
            double max = -10.0;
            int maxIter;
            for (int i = 0; i < 29; ++i){
                if (resultingNetwork.layers[resultingNetwork.layers.size() - 1][i].outputActivation > max){
                    max = resultingNetwork.layers[resultingNetwork.layers.size() - 1][i].outputActivation;
                    maxIter = i;
                }
            }
            results.push_back(maxIter);
            differences.push_back(maxIter - manDist);
        }
        myfile[0].close();
        
        //Displaying results
        int average = 0, numberOfImprovableStates = 0;
        for (int i = 0; i < differences.size(); ++i){
            average += differences[i];
        }
        cout << "Average Result - Manhatten Distance : " << (double)average/differences.size() << endl;
        average = 0;
        for (int i = 0; i < results.size(); ++i){
            if (results[i] > stateChosen)
                ++average;
            if (results[i] <= stateChosen && differences[i] > 0)
                ++statesImproved;
            if (((differences[i] - results[i]) * -1) != stateChosen)
                ++numberOfImprovableStates;
        }
        cout << "# of States overestimated - # of states total: " << average << "-" << results.size() << endl;
        cout << "# of states improved: " << statesImproved << endl;
        cout << "# of improvable states: " << numberOfImprovableStates << endl;
        cout << "% of states improved: " << (double)statesImproved/results.size() << endl;
        cout << "% of states NOT improved: " << 1.0 - (double)statesImproved/results.size() << endl;
        if (numberOfImprovableStates != 0)
            cout << "% of improvable states that were improved: " << (double)statesImproved/numberOfImprovableStates << endl;
        cout << endl << endl;
        cout << "continue? (1/0) : ";
        cin >> continueInput;
        
        statesImproved = 0;
        userInputActivation.clear();
        results.clear();
        differences.clear();
    }
    
    return 1;
}













