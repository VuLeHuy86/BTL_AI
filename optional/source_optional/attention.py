from torch import tril, ones, matmul, movedim
from torch.nn import Module, Linear
from torch.nn.functional import softmax

class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        keys = self.k_layer(input)
        queries = self.q_layer(input)
        values = self.v_layer(input)

        scores = matmul(queries, movedim(keys, 1, 2)) / (C ** 0.5)
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0, float('-inf')
        )[0]
        weights = softmax(scores, dim=-1)
        return matmul(weights, values)
