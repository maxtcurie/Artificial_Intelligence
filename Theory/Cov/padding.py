import numpy as np

def demonstrate_padding_types():
    input_shape = (5, 5)
    data = np.ones(input_shape)
    filter_size = 3

    # Valid padding
    valid_pad = np.pad(data, ((0, 0), (0, 0)), mode='constant')
    print("Valid Padding:")
    print(valid_pad)
    print()

    # Same padding
    same_pad = np.pad(data, filter_size // 2, mode='constant')
    print("Same Padding:")
    print(same_pad)
    print()

    # Causal padding
    causal_pad = np.pad(data, ((0, 0), (filter_size - 1, 0)), mode='constant')
    print("Causal Padding:")
    print(causal_pad)
    print()

demonstrate_padding_types()