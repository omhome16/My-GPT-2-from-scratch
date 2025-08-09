# A Deep Dive into a GPT-2 Implementation in PyTorch

This document provides a detailed explanation of a from-scratch implementation of a GPT-2 style transformer model. The focus is on the underlying architecture and the mathematical operations that enable the model to process and generate human-like text.

## Model Architecture: The Decoder-Only Transformer

This model is a **decoder-only transformer**, which is the standard architecture for GPT (Generative Pre-trained Transformer) models. It is designed specifically for language generation tasks. The model processes a sequence of input tokens and predicts the next token in the sequence. This is achieved by stacking multiple identical **transformer blocks**.

The overall data flow is as follows:

1.  Input token indices are converted into dense vector representations called **token embeddings**.
2.  A **positional embedding** is added to each token embedding to give the model information about the order of tokens in the sequence.
3.  This combined embedding vector is passed through a series of transformer blocks.
4.  The output of the final block is passed through a final layer normalization step and then a linear layer (the "language model head") to produce a probability distribution over the entire vocabulary for the next token.

Each transformer block is composed of two primary sub-layers:

1.  **Causal Self-Attention:** Allows each token to look at and gather information from previous tokens in the sequence.
2.  **Multi-Layer Perceptron (MLP):** A feed-forward neural network that processes the output of the attention layer, adding representational power.

Both of these sub-layers use **residual connections**, where the input to the layer is added to its output. This helps prevent the vanishing gradient problem in deep networks. Each sub-layer is also preceded by **Layer Normalization**.

## Core Components Explained

### Layer Normalization

Layer Normalization (LayerNorm) is a crucial technique for stabilizing the training of deep neural networks. Unlike Batch Normalization, it normalizes the inputs across the features for each data sample independently. It ensures that the inputs to the next layer have a consistent mean and variance.

**Mathematical Formulation:**

For a given input vector $x$ from a layer, LayerNorm calculates the mean ($\mu$) and variance ($\sigma^2$) across all its components. The normalized output $y$ is then computed as:

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Where:

* $\epsilon$ is a small constant added for numerical stability (e.g., $1e-5$).
* $\gamma$ (gamma) is a learnable scaling parameter (initialized to 1).
* $\beta$ (beta) is a learnable shifting parameter (initialized to 0).

These learnable parameters allow the network to scale and shift the normalized output, preserving its representational power.

### Causal Self-Attention

The heart of the transformer is the self-attention mechanism. In this model, we use **causal** (or masked) self-attention, which ensures that when predicting the token at position *t*, the model can only attend to tokens at positions $0, 1, ..., t$. It cannot "see" into the future.

The mechanism works by projecting the input vector $x$ for each token into three separate vectors: a **Query (Q)**, a **Key (K)**, and a **Value (V)**.

1.  **Scores:** The model calculates attention scores by taking the dot product of the Query vector of a given token with the Key vectors of all previous tokens. This score determines how much "attention" or focus to place on other tokens.
2.  **Scaling:** These scores are scaled down by dividing by the square root of the dimension of the key vectors ($d_k$). This prevents the dot products from becoming too large, which would result in extremely small gradients after the softmax function.
3.  **Softmax:** A softmax function is applied to the scaled scores to get a set of weights that sum to 1. These weights represent the contribution of each token to the current token's representation.
4.  **Output:** The final output for the token is a weighted sum of the Value vectors of all tokens it attended to.

**Mathematical Formulation (Scaled Dot-Product Attention):**

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This model uses **multi-head attention**, which means it performs this calculation multiple times in parallel with different, learned linear projections for Q, K, and V. This allows the model to jointly attend to information from different representation subspaces at different positions. The outputs of the heads are then concatenated and projected back to the original embedding dimension.

### Multi-Layer Perceptron (MLP)

Following the attention sub-layer in each transformer block is a simple yet powerful position-wise feed-forward network, also known as a Multi-Layer Perceptron (MLP). It is applied independently to each token's representation. This component allows the model to perform more complex transformations on the information gathered by the attention mechanism.

The MLP in this implementation consists of:

1.  A linear layer that expands the embedding dimension by a factor of 4 (e.g., from 768 to 3072).
2.  A GELU (Gaussian Error Linear Unit) activation function.
3.  A final linear layer that projects the dimension back down to the original embedding size.

**Mathematical Formulation:**

$$
MLP(x) = GELU(xW_1 + b_1)W_2 + b_2
$$

Where:

* $x$ is the output from the attention sub-layer.
* $W_1$, $b_1$, $W_2$, and $b_2$ are the learnable weights and biases of the two linear layers.

The expansion and contraction within the MLP is a common pattern that allows the model to learn more complex feature representations before the information is passed to the next transformer block.

## Acknowledgements

This implementation is inspired by the minimalist and educational approaches to building GPT models, which aim to distill the core concepts of the transformer architecture into clean and understandable code.
