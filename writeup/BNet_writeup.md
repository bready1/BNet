Here I propose and demonstrate a method for training neural networks using Bayesian updating. This method has a few advantages:
 - Learns from just a single pass of the data
 - Because it isn't using a gradient-based method we don't need to specific a loss function
 - Also don't need to specify a learning rate 

It has the main disadvantage of probably being strictly worse than normal back propagation for training a neural net. 

The idea behind this is that when using ReLU activation functions when the activation is either linear with the input (if it's activated) or zero (if it's not activated). So the activated neurons can be updated using Bayesian Linear Regression, and we leave the unactivated ones alone. Looking at a single layer with an activation function we have
$$\mathbf{a}^l=\sigma(\mathbf{a}^{l-1}\cdot\mathbf{w}^l)$$

where $\mathbf{a}^l$ is the activation of layer $l$ (the output), $\mathbf{a}^{l-1}$ is activation from the previous layer (the input into this layer), $\sigma$ is the activation function (in this case ReLU) ,and $\mathbf{w}^l$ is the weight matrix for this layer. (The bias can be incorporated by adding a $1$ to $\mathbf{a}^l$  and an extra row to $\mathbf{w}^l$).

This is a linear transformation with an activation around it, which is what neural nets are. And so we can look at each layer individually as
$$\mathbf{Y}=\sigma (\mathbf{x}\cdot\mathbf{w})$$

Because ReLU is linear when activated if we only consider the activated neurons (which correspond to $\mathbf{x}\cdot\mathbf{w}_j>0$) this is just a linear transformation:
$$\mathbf{Y}=\mathbf{x}\cdot\mathbf{w}$$

which we can update using Bayesian Linear Regression.

## Overview of the 'algorithm'

We have our data as $X_i,Y_i$ pairs and start with a standard forward pass through the network. Starting with $\mathbf{a}^0=X_i$ we calculate our activations using

$$\mathbf{a}^l=\sigma(\mathbf{a}^{l-1}\cdot \mathbf{w}^l)$$
![[BNet/NN1.png]]
The activation function can also be thought of as multiplying element wise by a vector $s^l$ which is $1$ when $\mathbf{a}^{l-1}\cdot\mathbf{w}^l_j>0$ and $0$ otherwise
$$\mathbf{a}^l=(\mathbf{a}^{l-1}\cdot \mathbf{w}^l)\circ s^l $$

When we do the forward pass we save each $a^l$ and $s^l$.
![[NN2.png]]
Now for the training part. When we feed in the data $X_i$ we would like the neural net to output something close to $Y_i$, this should be the output of the final layer ($l=L$). We want the inputs into the final layer to be transformed to give $Y_i$,
$$Y_i=(\mathbf{a}^{L-1}\cdot \mathbf{w}^L)\circ s^L$$. This can be done by only taking the columns of $\mathbf{w}^l$ for which $s_j^l=1$.

But if we only consider the elements of this for which $s^l=1$ we get the linear equation

$$Y_i'=\mathbf{a}^{L-1}\cdot (\mathbf{w}^L)'$$

This can be updated using Bayesian linear regression (more on this later), which will update the weight matrix $(\mathbf{w}^l)'$. We then plug the updated values of $(\mathbf{w}^l)'$ back into $\mathbf{w}^l$, leaving the other elements unperturbed. This has updated the final layer of the network.

Now for a dodgy step, to update the next layer (layer $L-1$) we have the input $\mathbf{a}^{L-2}$ but we need the desired output for the Bayesian update which we'll call $Y^l$, to do this we invert (or pseudoinvert, more on this later) the final layer 

$$Y^l=Y_i' \cdot \text{inv} (\mathbf{w}^l)'$$

We use only the activated elements, because these are the only ones that contributed to the linear transformation in the forward pass. 

For this layer we have the equation
$$(Y^{L-1})'=\mathbf{a}^{L-2}\cdot (\mathbf{w}^{L-1})'$$

Where the primes indicate we are taking only elements or columns for which $s^{L-1}=1$. We then update the weight matrix as before, and then perform the same inversion trick to find the next $Y^l$
$$ Y^l=Y^{l+1}\cdot \text{inv}(\mathbf{w}^l)$$
and begin again. 

This 'back propagation' is done all the way back through the network to layer $l=1$. The fact that this is propagating backwards does maybe hint that there are going to be more connections between this weird (bad) Bayesian updating scheme and normal backpropagation. 
![[NN3.png]]
So to recap:
1. We first perform a forward pass through the network, saving all the $\mathbf{a}^l$ and $s^l$.
2. For the final layer we perform a Bayesian updating step to make the network output closer to $Y_i$
3. 'Invert' the final layer to find the 'desired output' $Y^l$ for the next layer
4. Update the next layer using Bayes
5. Invert that layer to find the next 'desired output'
7. Repeat through all of the layers

We update the weights and then use the updated weights to find the 'target' of the next layer. 

## Bayesian updating step
The actual update step is the most important part of this algorithm, and so we will go into more depth. For this I was following the Wikipedia page for [Bayesian multivariate linear regression](https://en.wikipedia.org/wiki/Bayesian_multivariate_linear_regression)

We can write our linear transformation as a single matrix equation

$$\mathbf{Y}=\mathbf{X}\cdot\mathbf{w}$$

The matrix $\mathbf{w}$ has an uncertainty associated with it, represented by $\mathbf{\Lambda}$. The matrix $\mathbf{\Lambda}^{-1}$ is the covariance between the rows of $\mathbf{w}$. Roughly, a larger $\mathbf{\Lambda}$ means less uncertainty in the elements of $\mathbf{w}$. 

For each $(X_i,Y_i)$ pair, we update both $\mathbf{w}$ and $\mathbf{\Lambda}$:

$$\mathbf{w} \to (X_i^TX_i+\mathbf{\Lambda})^{-1}(X_i^TY_i + \mathbf{\Lambda}\mathbf{w})$$
$$\mathbf{\Lambda}\to X_i^TX_i+\mathbf{\Lambda}$$

Bayesian linear regression has the nice property that each update uses all the information from the data $(X_i,Y_i)$, so you don't need to run through the data multiple times. Updating multiple times from the same data would in fact lead us to update too far. Additionally, it is independent of the order that we update in; starting from the same prior ($\mathbf{w}_0$ and $\Lambda_0$) we can update from the data in any order and will get the same result (assuming we update from each datum exactly once). 

Because this algorithm makes a few pretty dubious steps, the order-independence property will certainly not hold (neither will the 'updating all the way' property I assume). Therefore it might be good to only update some of the from each data point, and then do multiple passes through all of the data. If we want to do this the updating method can be easily modified, such that each update only updates a fraction $c$ amount of the total
$$\mathbf{w} \to (cX_i^TX_i+\mathbf{\Lambda})^{-1}(cX_i^TY_i + \mathbf{\Lambda}\mathbf{w})$$
$$\mathbf{\Lambda}\to cX_i^TX_i+\mathbf{\Lambda}$$

In a normal linear regression case we could use these equations to update, and just run through the data $1/c$ times. 

It seems like $c$ plays an analogous role to the learning rate $\alpha$ for normal Backpropagation/SGC, controlling how large steps to take. $c$ acts as a hyperparameter, and I am unsure whether setting it to anything except 1 is a good idea. 

## Calculating $Y^l$
When we update the weights of each layer $\mathbf{w}^l$ we need the 'target' $Y^l$ to update towards. This is done by performing some kind of inversion on the just updated weights ahead of it in the network $\mathbf{w}^{l+1}$. We want to invert the equation:
$$Y^{l+1}=Y^l\cdot \mathbf{w}^{l+1}$$
Using the newly updated weights $\mathbf{w}^{l+1}$, and only using elements of $Y^{l+1}$ and $\mathbf{w}^{l+1}$ which correspond to activations ($s_j^{l+1}=1$). There is no guarantee that $\mathbf{w}^{l+1}$ will be a square matrix, and so we cannot just invert it. Furthermore if $Y^l$ is of greater length than $Y^{l+1}$, then inverting the equation will be under determined and there will be infinitely many possible $Y^l$. But there is a desired property of $Y^l$ which we can use to select a single one; we want the $Y^l$ which is closest to $\mathbf{a}^l$. This means when we update the weight matrix $\mathbf{w}^l$  we update it the minimum amount to be consistent with the new information, rather than have it fluctuate wildly with each update. Using this criterion $Y^l$ is given by
$$Y^l=\mathbf{a}^l + \mathbf{w}^{l+1}((\mathbf{w}^{l+1})^T\cdot\mathbf{w}^{l+1})^{-1}(Y^{l+1}-\mathbf{a}^l\mathbf{w}^{l+1})^T$$

If the superscripts are making this hard to read, it looks like this when we remove some of them
$$Y^l=\mathbf{a} + \mathbf{w}(\mathbf{w}^T\mathbf{w})^{-1}(Y^{l+1}-\mathbf{a}\mathbf{w})^T$$

A derivation can be found below[^bignote]. 

[^bignote]: From [here](https://faculty.math.illinois.edu/~mlavrov/docs/484-spring-2019/ch4lec4.pdf) we can solve for the smallest (minimum norm) vector $\mathbf{X}$ satisfying 
	$$\mathbf{X}\mathbf{B}=\mathbf{Y}$$
	using
	$$\mathbf{X}=\mathbf{B}(\mathbf{B}^T\mathbf{B})^{-1}\mathbf{Y}^T$$
	We want to solve $Y^{l+1}=Y^l\cdot \mathbf{w}^{l+1}$ for $Y^l$ such that $Y^l$ is as close to $\mathbf{a}^l$ has possible. We can define $Y^l=\mathbf{a}^l+\delta$, we want to find the smallest $\delta$ which satisfies this equation:
	$$Y^{l+1}=(\mathbf{a}^l+\delta)\cdot \mathbf{w}^{l+1}$$
	This can be rearranged to give
	$$\delta \cdot \mathbf{w}^{l+1}=Y^{l+1}-\mathbf{a}^l\cdot \mathbf{w}^{l+1}$$
	The right hand side of this equation will play the role of $\mathbf{Y}$, and so we can solve for $\delta$:
	$$\delta=\mathbf{w}^{l+1}((\mathbf{w}^{l+1})^T\cdot\mathbf{w}^{l+1})^{-1}(Y^{l+1}-\mathbf{a}^l\mathbf{w}^{l+1})^T$$
	and so the $Y^l$ closest to $\mathbf{a}^l$ is
	$$Y^l=\mathbf{a}^l + \mathbf{w}^{l+1}((\mathbf{w}^{l+1})^T\cdot\mathbf{w}^{l+1})^{-1}(Y^{l+1}-\mathbf{a}^l\mathbf{w}^{l+1})^T$$
	
