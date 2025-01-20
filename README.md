# Aaron Frost - Jan 2025
To learn about deep learning / neural networks, I'm creating a neural network that classifies a flower given its instance properties/features.
It is fully-connected feed forward, so all neurons in each layer influence all neurons in adjacent layers.

The first notebook is flower_classifier.py, but in the next I'll use NVIDIA transformer engine, with PyTorch, casting activations to FP16 for increased performance.
https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html

The dataset: https://archive.ics.uci.edu/dataset/53/iris
--> contains:
       - 3 flower classes
       - 50 instances of each class
       - 4 continuous features per instance

  The idea is that we can separate the 4D feature vectors into 3 groups.
     "One class is linearly separable from the other 2; the latter are not linearly separable from each other."
     --> this means that all the points in 4D space for *one* class are on one side of a line travelling through 4D space,
         and the other two class instances are all on the other side of that line.

   Multi-layer perceptrons (MLP) can classify data that is not linearly seperable, and take a flattened vector as input.
   Think of an MLP classifier not as splitting space into 3 groups based on geometric distances, but as sculpting a highly intricate decision landscape. If you're interested in computing distances to fixed centers, you'd want a centroid-based method like kNN.
   Important to note, however, that convolutional neural networks (CNN) outperform MLP and are industry standard for classification problems. However, for this example, since MLP is used in transformers (underlying GPT), we'll stick with MLP.

 Some breakdown of the fundamentals


# The Network
 First, let's look at a multilayer perceptron - some old tech.
   Neuron - container of a number called an "activation" 
     - The first layer of a network should contain all the input features.
         - If your input features are pixels of an image, you'd have Width*Height Neurons in the first layer, each containing a value like alpha or RGB.
     - The last (output layer) has one neuron for each class you're trying to split into.
         - The activations in the output layer neurons can be seen as the model's prediction for the class (the likelihood of being that class given the input features)

 It's resonable to expect a layered structure to behave intelligently because human recognition breaks down things into subcomponents.
   -->  Each subcomponent could be mapped to a neuron in a layer, ideally, with the activation representing the liklihood of that component being present.
   --> The expectation or hope, following "training" is to have each layer activate on a particular set of qualities from the original input vector.
       The leftmost layers, we hope, activate with more granular / microscopic components, and those neurons that activate should be fed forward to activate neurons representing a more macroscopic quality presense.
     ***Microscopic qualities to fed forward (left to right) through layers to Macroscopic activations
     ***Concrete to abstract layers***

 The hope might be for a single neuron in a particular layer to pick up on the presence of a quality.
   In order to step from one layer of the network to another, there has to be some sort of transformation.
   This is where "weights" come into play.
       On each "line" connecting the neruons of one layer to the neuron of the next (remember it's fully connected), you can imagine that there is a set of numbers that if you apply them as multiplicative weights, the sum will
       fit a certain criteria required for activation (be positive, for example) only when a given feature is present. The "weights", are like a mask, that determines whether a single quality is present or not given the qualities from the previous layer.

 Example I made up. "Should I get pizza classifier":
     Input features:
         Hunger: 1
         Pizza-love: 3
         Burrito-love: 4

Hidden layer 1 neuron:
  - maybe this neuron should fire if we like burritos more than pizza (arbitrary quality)
        neuron-activation = Hunger(1)*(weight: 0) + Pizza-love(weight:-1) + Burrito-love(weight: 1)
          = 0 + -3 + 4
          = 1 (it's positive so the neuron activated!)
      (notice I made the weight of pizza love -1, and burrito-love 1, this is so the neuron would fire if we like burritos more).
      These weights made this neuron fire (I guess we can call it the "burrito is better than pizza" neuron). This neuron firing should, ideally, feeding it forward, lead to activation of the output neuron that classifies the answer as no, let's get a burrito instead!

  This 3B1B video does a good job showing how weights are like a mask that determines presence of a feature: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1&ab_channel=3Blue1Brown

  When you compute the weighted sum, the output could range.. so you want to SQUISH the range into something more normalized, like 0 to 1.
     There is a method called the sigmoid func (aka "logistic curve") which puts very negative inputs close to 0 and positive inputs close to 1.
     The sigmoid curve measures "how positive or negative" the weighted sum is, AKA whether or not to "light up" the destination neuron
       input 0 -> output .5
       input -3 -> output close to 0
       input 1 -> output close to 1
     you can apply some bias (maybe a neuron should only activate if it is super positive? sure.), by adding (or subtracting) a number from the weighted sum before passing into sigmoid
     ex. activation = sigmoid(weighted_sum(prev_layer) + bias)
       By convention, every neuron has this bias value, added to the sum before squishing with the sigmoid.

Weights and biases combined are called "parameters".
"Learning" is finding the valid setting for all these parameters, to solve the problem at hand (have the correct output neurons fire for a given input neuron set)


   Okay, with all that out of the way, let's go over how we're going to translate this to vectors we can work with in Python.
     For each pair of layers:
     1. The neuron activations from the left-layer is a single vertical array (column vector)
     2. The weight values for all connections to the next layer are a 2D array
         - each row represents the weights to a neuron in the next layer

This means if you multiply a row from this 2d array of weights against the column vector (the entire left-layer activations), the number you recieve represents the weighted sum for the right-layer neuron associated with that row.
Matrix multiplication requires multiplying matching members and summing them together. If we multiply the weights 2D matrix against the column vector, we actually get the activations for the next row.
-->  since the height of the column vector is equal to the width of the matrix, you can multiply the 2d-Array by the column vector. To multiply two vectors component by component, and sum them all, you are computing the "dot product", is what this is called.
  The dot product is expressed this way algebraically.
The biases is just another column vector that gets added to the previous matrix-vector product.
Finally, apply the sigmoid to each compoent of the resulting column vector.
  OR you can apply a reLU  instead of sigmoid, which returns y=x when input is positive, and y=0 when input is negative.
    - reLU -> preferred for hidden layers
    - sigmoid -> preferred for output layers in binary classification
    
 # How the Network Learns 
 
 The entire network learns iteratively. 
 Start by initializing all the weights and biases to something random. With each learning "iteration", the weights and biases are getting closer to minimizing something we call a "cost" function, that measures how poorly the entire network is doing.
 Suppose there is a "cost" associated with a single training data point. This is the sqr magnitude of the expected vs actual activations in the output layer.
 If you average the cost across all training examples, you now have the cost function, which you can find the negative gradient of, and step the weights&biases in that direction --> minimizing the cost.

   Training data point:
     The desired output-layer neuron activations values (0-1), given a particular input activation vector.
   The COST of a single training example is outputActivations.sum(o => Math.Square(o.DesiredActivation - o.ActualActivation)) (use square to get the "distance from correct" -> higher distance => less correct => higher cost)
   The AVERAGE cost over ALL training examples is the COST FUNCTION.. a measure of how good the network is doing.. Although in principle you should compute the cost of every trainign example, at every step, for computational efficiency there is an optimization.
 The COST is a function of all the weights and biases, and returns the average over all training examples

  you can figure out what the best weights and biases are by travelling down the slope of the cost function. You should descend down the space at a step-rate relative to the slope, so you don't overstep the minimum cost.
     the "gradient" of a function gives you the direction of steepest incline, and multiply by -1 => steepest descent vector.
  So just compute the gradient direction, step the weights in biases in that direction (downhill), repeast that over and over while the cost return value decreases.
   

 How to compute gradient: 
   using "Backpropagation"

   For each training example, you can look at the output neuron activations given the current state of the data. 
     Depending on whether a given output neuron should fire or not, you can determine the changes to the weights and biases that need to occur in the prior layer.
     You have to add together the "desired" changes to the weights of the ouptut neurons.
     This gives you a list of "nugdges" that should happen to the second to last layer. And you recurse backwards through the network to gather these nudges over the whole network.
     You do this same routine for every trainingn example, and average together the desired changes for every weight and bias. This is loosly speaking, the negative gradient of the cost function.

   "Stochastic gradient descent" is the optimized backpropagation method, where you batch your training data, and compute a gradient descent step based on the "mini-batch", to converge to a local minimum of the cost function.
   Chain rule - to figure out the "nudge" that is needed to minimize cost for a particular neuron, you need to find the affect of change that a given weight has on the cost.  The chain rule expresses how the innermost variable of a composite function affects changes the outermost value.
       If you multiply the sensitivity that the inner function has to the innermost variable by the sensitivity the outer function has to the passed in function, you get the sensitivity the outer function has to the innermost variable. It is simply chaining together the "sensitivity" ratios.



# GPT - Generative Pretrained Transformers
 Let's talk about TRANSFORMERS.
 GPT - trained to take in string text, and produce prediciton for what comes next.
       Output is a probability distribution over different chunks of output that would follow.

   How data flows through a transformer:
     1. break down input into tokens.
     2. each token gets a vector (high dim vector embedding, corresponding to semantic meaning (no context with the word))
     3. all the tokens are passed through an "attention" block, which rebalances the vectors based on the other vectors in the phrase.
     4. in parallel, pass each vector through a multilayer perceptron (feed foward layer). 
         - Each neuron activiation is effectively asking a long list of questions about each token, and updating them based on the answers to those questions
     5. repeat the process
     6. the last vector (token) in the sequence should have the effective meaning of the passage "baked" into it. 
         --> multiply against "unembedding matrix" to map back to the token -> the values in this matrix are learned during training.
         --> only the last vector is used because it is much more efficient during training  
          --> compute probability into normalized distribution (adds up to 1) using "softmax" function.
          --> "temperature" can be added to give more weight to lower values (low T => hard maxing (choosing the most predictable word)). High temperature => less predictable.


 Embedding matrix:
     Each token maps to a vector which encodes the meaning of that token, and the position in the input.
   As these token vectors pass through the transformer model (attention blocks and multilayer perceptrons), they should "soak up" the meaning of surrounding context.


 How is the final probability distribution calculated:
   Given the very last vector in the "context", you map it to the vocabulary using an unembedding matrix, and then computing softmax.

   A temperature can be applied to the softmax "spice up" the output, making less probable outcomes more probable.


 Attention blocks:
   Calculates vector to add to the generic token embedding to get the meaning in context of the rest of the passage.
     The generic token embedding doesn't really encode any meaning, just the word with NO context.

A "head" of attention:
This is a pattern that represents the "updates needed" for a single transformation (ex. have adjectives adjust meaning of corresponding nouns --> the noun vectors should be updated to "soak in" the adjectives).

A "query" for a given token, in this example, would represent the question: "are there any surrounding adjectives?" --> query vector. Computing the query vector is a means of multiplying the embedding against a set of weights (the weights, when trained, will create this hypothetical question when multiplied against the input)
--> the true behavior of a query vector (what hypothetical "question" it asks about the token), is learned from data, as the query vector = input vector * weights vector

The "key matrix" is a second sequence of vector weights, called the keys, which you can imagine representing the answers to those questions, for each token.

If there is close alignment between a key and a query (using dot product), then the key answers the query. The embeddings of the key are then said to "attend to" the embedding of the query, where the dot product is high (similar)
Now, given the dot products (similarity scores), you can softmax each column of the attention head (imagine this being a matrix with query vectors for cols, key vectors for rows), with the row*col "dots" in each cell.
  The softmax of a given column represents the normalized distribution, representing which embedding tokens are most likely to affect other tokens, in terms of the query.

  With the key tokens determined, a third weight matrix is introduced for training (Value Matrix)
  The goal of the Value matrix is to determine the vector that needs to be added to the query token.
  You can imagine it moving the query token to encode the meaning (ex. attaining an ajectives quality in vector space, given the answer to the query)
  --> "If token X is relevant to adjusting the embedding of token Y, the value matrix weights are trained to apply a transformation to the X token, producing a value vector, that represents the quality that can be applied to token Y, when added to token Y's embedding"
      You add the corresponding value vectors to the query embedding, multiplied by the weights in each cell, computed in the attention head. The sum of the column after the value vector is applied represents the tokens being transformed by the full context, basically "soaking in" what should be paid attention to, and how (and how much) each term should affect each other term.

The result is a more refined, contextually rich embedding for each token.

In short:
  Query - weights trained to ask a question
  Key   - weights trained to answer that question
      Query * Key = positive if key answers query / the amount of relevance of the key to the query
  Value - "if key answers query, weights trained to apply key token's transformation to the "source".
  
--> this will refine teh input vector when the weighted value vectors are applied.


   GPT uses multiple attention heads in parallel before actually modifying the embeddign vector, so that an emebedding can be more richly contextualized, however the model decides to configure the weights. --> remember there are layers, so a richer layer is better than a layer that only encodes one query.

 Why multilayer perceptrons in between attention heads?
     The MLP enriches the input vectors with facts using a ReLU to clip off when a fact is not true.
     The MLP activations are Linear (no squishing) (to encode the answers to questions) -> ReLU (to evaluate the fact / clean up the facts) -> Linear (to map back down to the embedding space (this is a "down projection" matrix)) 
     The output is added to the input vector (imagine "basketball" encoding / context being added to the input "Michael Jordan")


 interestingly, high N-dimensional spaces can encode more distinct ideas and relationships than the number of dimensions (exponentially, in fact), thus, the superposition (multiple values for activations) encodes more complex relationships.
