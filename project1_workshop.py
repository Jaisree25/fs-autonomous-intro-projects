import torch

# Part 1 - What is a tensor? Vectors and matrices
def part1():
        
    print("Part 1 - What is a tensor? Vectors and matrices\n")

    scalar = torch.tensor(5)
    print("Scalar:", scalar)
    print("Shape:", scalar.shape)

    print("\n")

    vector = torch.tensor([1.0, 2.0, 3.0])
    print("Vector:", vector)
    print("Shape:", vector.shape)

    print("\n")

    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print("Matrix:", matrix)
    print("Shape:", matrix.shape)

    print("\n")

    tensor3d = torch.rand(2, 3, 4) 
    print("3D Tensor:", tensor3d)
    print("Shape:", tensor3d.shape)

    print("\n")

# Part 2 - Addition, subtraction, multiplication
def part2():
    
    print("Part 2 - Addition, subtraction, multiplicatios\n")

    A = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    B = torch.tensor([[6, 5, 4],
                      [3, 2, 1]])
    
    print("A =\n", A)
    print("B =\n", B)
    print("A + B =\n", A + B)
    print("A - B =\n", A - B)
    print("A * B (elementwise) =\n", A * B)

    # Matrix multiplication

    C = torch.tensor([[1, 2],
                      [3, 4],
                      [5, 6]])
    D = torch.tensor([[7, 8],
                      [9, 10]])
    print("C =\n", C)
    print("D =\n", D)
    print("C @ D =\n", C @ D)

# Part 3 - Squeezing, unsqueezing, and transforming
def part3():

    print("Part 3 - Squeezing, unsqueezing, and transforming")

    x = torch.rand(2, 1, 3, 3)
    print("Original shape:", x.shape)

    x_squeezed = torch.squeeze(x)
    print("After squeeze:", x_squeezed.shape)

    x_unsqueezed = torch.unsqueeze(x_squeezed, dim=1)
    print("After unsqueeze:", x_unsqueezed.shape)

    x_reshaped = x_unsqueezed.view(2, -1)
    print("After transform (reshaped):", x_reshaped.shape)

if __name__ == "__main__":
    print(torch.__version__)
    part1()
    part2()
    part3()