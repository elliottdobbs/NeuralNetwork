//Elliott Dobbs
//Neural Network Implementation
//11-29-2018

#include <vector>
using namespace std;

//Link structure to represent weighted links between nodes
struct NeuralLink{
    int sendingNode;
    int recievingNode;
    double weight;
    double bias;
};

//Node structure to represent a node in the Nerual Network
struct NeuralNode{
    vector<NeuralLink> inputLinks;
    vector<NeuralLink> outputLinks;
    double activation;
    double delta;
};

//Layer Structure
struct NeuralLayer{
    bool input;
    bool output;
    int size;
    vector<NeuralNode> nodes;
};



//Nerual network structure
class NeuralNetwork{
    
    int numberOfLayers;
    vector<NeuralLayer> layers;
    vector<double> sigmoid;
    vector<double> hyperbolicTan;
    
    //Constructor
    NeuralNetwork(int layerCount, vector<int> sizes){
        
        numberOfLayers = layerCount;
        
        //Initializing the layer and nodes
        for (int i = 0; i < layerCount; ++i){
            NeuralLayer tempLayer;
            
            tempLayer.size = sizes[i];
            
            if (i == 0){
                tempLayer.input = true;
            }
            else {
                tempLayer.input = false;
            }
            
            if (i == layerCount - 1){
                tempLayer.output = true;
            }
            else {
                tempLayer.output = false;
            }
            
            //initializing the nodes
            for (int j = 0; j < sizes[i]; ++j){
                NeuralNode tempNode;
                
                tempNode.activation = 0.0;
                tempNode.delta = 0.0;
                
                tempLayer.nodes.push_back(tempNode);
            }
            
            layers.push_back(tempLayer);
        }
        
        //initializing the links with random weights now that we have all the nodes and layers
        for (int i = 0; i < layerCount; ++i){
            
            NeuralLink tempLink;
            tempLink.bias = 0.0;
            
            //Output Links
            if (i != layerCount - 1){
                
                //Iterates through the ith layer size
                for (int j = 0; j < sizes[i]; ++j){
                    tempLink.sendingNode = j;
                    
                    //iterates through the i+1th layer size
                    for (int k = 0; k < sizes[i+1]; ++k){
                        tempLink.recievingNode = k;
                        
                        double random = (double)(rand() % 2000 - 1000)/1000;
                        tempLink.weight = random;
                        
                        layers[i].nodes[j].outputLinks.push_back(tempLink);
                    }
                }
            }
            
            //Input Links
            if (i != 0){
                
                //Iterates through the i-1th layer size
                for (int j = 0; j < sizes[i-1]; ++j){
                    tempLink.sendingNode = j;
                    
                    //iterates through the ith layer size
                    for (int k = 0; k < sizes[i]; ++k){
                        tempLink.recievingNode = k;
                        
                        double random = (double)(rand() % 2000 - 1000)/1000;
                        tempLink.weight = random;
                        
                        layers[i].nodes[j].inputLinks.push_back(tempLink);
                    }
                }
            }
        }
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
    
    void generateHyperbolicTangent(){
        double x, hyp;
        
        for (int i = 0; i < 10000; ++i){
            x = -5.0 + ((double)i/1000.0);
            hyp = (exp(x) - exp(-x))/(exp(x) + exp(-x));
            hyperbolicTan.push_back(hyp);
        }
    }
    
    void setInputActivations(vector<double> acts){
        
        for (int i = 0; i < layers[0].size; ++i){
            layers[0].nodes[i].activation = acts[i];
        }
    }
    
    vector<double> getOutputActivations(){
        
        vector<double> result;
        
        for (int i = 0; i < layers[numberOfLayers - 1].size; ++i){
            result.push_back(layers[numberOfLayers - 1].nodes[i].activation);
        }
        
        return result;
    }
    
    //Function to test the network.  Propigates a given input example throught the network and returns the network.
    void progigate(){
        
        //Propigate input forward to output
        //for each layer after the input layer
        for (int layer = 1; layer < numberOfLayers; ++layer){
            //for each node in that layer
            for (int node = 0; node < layers[layer].size; ++node){
                
                //Get weighted sum
                double weightedSum = 0.0;
                //for each node in the layer before this one
                for (int inputLink = 0;
                     inputLink < layers[layer - 1].size;
                     ++inputLink){
                    
                    //WAIT... right now I dont think the links are correct.  I made 2 links for each node on another node.  One is the outputLink of one, while the other in the input of the other node
                    //Draw it out,,, might not matter
                    weightedSum +=  layers[layer][node].inputLinks[inputLink].weight *
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
    
/*    //Main function to teach the neural network.  Follows the pseudocode from toe book.  Takes in a set of example inputs and expected outputs, and a blank network.  Calls the makeWeights function to assign random weights to the links.  Then for each input state, propigates the input forward to the output layer, calculates the deltas, and propigates backwards calculating delta errors.  Finally updates the weights based on the errors and moves on to the next example.
    NeuroNetwork backPropLearning(vector< vector<double> > inputActivations,
                                  vector< vector<double> > outputActivations,
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
    */
    
    
};
