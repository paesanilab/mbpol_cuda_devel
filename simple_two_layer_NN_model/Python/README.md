# Implementing a simple two_layer Neural Network

This example is an exercise from open course CS231n Convolutional Neural Networks for Visual Recognition from Stanford university: http://cs231n.stanford.edu/

The credit will be given to the authors of the course material.



In this work we will develop a neural network with fully-connected layers to perform classification.

The NN model consists of two layers, and a final classifier/loss caculation: 
1 A fully connected layer, with ReLN nonlinearity:

Fully connected layer: output_vector = input_vector * weight_matrix + bias_vector;

ReLN nonlinearity : output_vector = max(0, input_vector);
        
In mathmatical expression, the output vector $h_1$ of this layer is:
       $$ h1 = max( 0, (x * W1 + b1) )$$ 
where $x$ is the input vector (a sample), $W1$ and $b1$ are weight matrix and bias vector respectively.
    
2 A fully connected layer.
$$ h2 = h1 * W2 + b2$$


3 The final loss of the output classifier (the weight that it predicts model INCORRECTLY) uses softmax classifier. The softmax classifier means, the element at index $i$ in output vector ($h_i$) equals its exponential probability in the output vector: 
$$ h3_i = \frac{exp(h2_{i})} {\sum\limits_{j} exp(h2_j)} $$

The final loss equals to the negative value of logarithm of $h$ at correct classifier index:
For a sample $x$ whose correct classifier is $y$, its loss is:

 $$ L =  - log(h3_y) = -   log(    \frac{exp(h2_{y})} {\sum\limits_{j} exp(h2_j)}      )           $$  


As a example, if an input sample vector $x$ has its correct classifier $y=1$, and the output $h3$ classifier is a 5-element vector, then the loss on this sample is the negative value of logarithm of $h$ at index 1:
$$ L =  - log(h3_y) = -log(h3_1) $$


Further notice, if multiple samples are considered, the final loss is the average loss from each sample.
