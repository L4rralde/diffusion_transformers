import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        patch_size: int,
        n_channels: int,
        dim: int,
        num_heads: int,
        mlp_ratio: int
    ) -> None:
        super().__init__()
        #Extra token?
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.projection = LinearEmbedding(n_channels*patch_size**2, dim)
        self.encoder = Encoder(n_blocks, dim, num_heads, mlp_ratio)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        #Image of shape B x h x w x c
        assert len(image.shape) == 4, "Expected image with shape B x h x w x c"
        batch_size, c, h, w = image.shape
        assert h%self.patch_size == 0 and w%self.patch_size == 0,\
                "Image need to be resized to patchify"
        assert c == self.n_channels, "Number of channels mismatch"

        patches = patchify(image, self.patch_size) #B x n_patches x (c * patch_size^2)
        patches = self.projection(patches) #B x n_patches x dim

        #Add class token?
        sincos = sin_cos_embedding(patches)
        patches += sincos
        #B x n_patches x dim or B x (n_patches +1) x dim if extra token added
        tokens = self.encoder(patches)

        return tokens


#Non-learnable constant function
def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    #Expect to receive an image of shape B x h x w x c
    batch_size, c, h, w = image.shape

    number_patches = h*w//(patch_size**2) #
    patches = image.view(batch_size, number_patches, -1) #B x n x (p^2 c) = B x (h x w /(p^2)) x (p^2 c) = B x h x w c
    return patches


#Learnable embbeding    
class LinearEmbedding(nn.Module):
    def __init__(self, patch_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(patch_dim, hidden_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        #Expects a B x (p^2 c) tensor
        projection = self.project(patch)
        return projection


#Non-learnable position embedding. SinCos?
def position_embedding(patches: torch.Tensor) -> torch.Tensor:
    #By the moment don't add positional embedding. TODO
    raise NotImplementedError("")


def sin_cos_embedding(patches: torch.Tensor) -> torch.Tensor:
    assert len(patches.shape) == 3, "Expected an input tensor of shape B x n x dim"
    batch_size, n_tokens, dim = patches.shape
    thetas = torch.Tensor([
        [
            i / (10_000**(2*j/dim))
            for j in torch.arange(0, dim) #js
        ]
        for i in  torch.arange(0, n_tokens) #is
    ])
    sincos = torch.zeros(n_tokens, dim, dtype=patches.dtype).to(patches.device)

    sincos[:, 0::2] = torch.sin(thetas[:, 0::2])
    sincos[:, 1::2] = torch.cos(thetas[:, 1::2])
    sincos = sincos.unsqueeze(0)
    sincos = sincos.expand(batch_size, n_tokens, dim)
    
    return sincos


class Encoder(nn.Module):
    def __init__(self, n_blocks: int, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.blocks = [
            Block(dim, num_heads, mlp_ratio)
            for _ in range(n_blocks)
        ]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            tokens = b(tokens)
        
        return tokens

#Extra learnable token?
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int #Used to compute mlp_hidden_dim
    ) -> None:
        super().__init__()

        self.norm_1 = nn.LayerNorm(dim) #Learnable fisrt norm layer. Normalizes accros the dim dimension
        self.attention = Attention(dim, num_heads)
        self.norm_2 = nn.LayerNorm(dim)
        self.mlp = BlockMlp(dim, mlp_ratio*dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        #Tokens of shape B x n_tokens x dim
        tokens += self.attention(self.norm_1(tokens))
        tokens += self.mlp(self.norm_2(tokens))

        return tokens


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim%num_heads==0, "dim must be divisible by number of heads"
        self.dim = dim
        self.num_heads = num_heads

        self.head_dim = dim//num_heads #Or query size
        self.qkv = nn.Linear(dim, 3*dim) #query, key, value embedding

        #dim to dim projection
        self.project = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        #Tokens are of shape B x n_tokens x dim
        batch_size, n_tokens, dim = tokens.shape
        assert dim == self.dim, "Dimension mismatch"

        #Embed qkv tokens
        qkv = self.qkv(tokens) #B x n_tokens x (3*dim)
        qkv = qkv.view(batch_size, n_tokens, 3, self.dim) # B x n x 3 x dim
        #Now num_heads-reshaping awareness
        qkv = qkv.view(batch_size, n_tokens, 3, self.num_heads, self.head_dim) #B x n x 3 x heads x head_d
        qkv = qkv.transpose(1, 3) #B x heads x 3 x n x heads_d
        q, k, v = torch.unbind(qkv, 2) #3 tensors of shape B x heads x n x heads_d
        #softmax((Q * K)/scale) * value. Attention
        heads_tokens = nn.functional.scaled_dot_product_attention(q, k, v) #B x heads x n x heads_d

        #Now reshaping back to token dimension
        tokens = heads_tokens.transpose(1, 2) #B x n x heads x heads_dim
        tokens = tokens.view(batch_size, n_tokens, self.dim) #B x n x (heads*heads_dim) = B x n x dim

        #Final projection
        tokens = self.project(tokens)

        return tokens


class BlockMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
