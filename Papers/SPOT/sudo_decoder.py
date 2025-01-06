def autoregressive_decoder(slots, patch_features, num_patches, d_model):
    """
    slots: Fixed object-centric slots from the encoder, shape (k, d_model)
    patch_features: Initial input patch features (e.g., [BOS] token for the first step)
    num_patches: Total number of patches to predict
    d_model: Dimensionality of slots and patch embeddings
    """

    # Initialize the input sequence with the [BOS] token
    input_sequence = [BOS]  # Shape: (1, d_model)
    output_sequence = []    # To store predicted patch features

    for t in range(num_patches):
        # Step 1: Self-Attention on the input patch sequence
        # Compute Queries (Q), Keys (K), Values (V) from the current input sequence
        Q_self = Linear(input_sequence)  # Shape: (t+1, d_model)
        K_self = Linear(input_sequence)  # Shape: (t+1, d_model)
        V_self = Linear(input_sequence)  # Shape: (t+1, d_model)

        # Compute causal self-attention scores
        attention_scores_self = Softmax(Q_self @ K_self.T / sqrt(d_model))  # Shape: (t+1, t+1)
        # Apply triangular mask to enforce autoregressive behavior
        attention_scores_self = mask_upper_triangle(attention_scores_self)

        # Weighted aggregation for self-attention
        self_attention_output = attention_scores_self @ V_self  # Shape: (t+1, d_model)

        # Step 2: Cross-Attention with Slots
        # Compute Queries (Q), Keys (K), and Values (V) for cross-attention
        Q_cross = Linear(self_attention_output)  # Shape: (t+1, d_model)
        K_slots = Linear(slots)                  # Shape: (k, d_model)
        V_slots = Linear(slots)                  # Shape: (k, d_model)

        # Compute cross-attention scores
        attention_scores_cross = Softmax(Q_cross @ K_slots.T / sqrt(d_model))  # Shape: (t+1, k)

        # Weighted aggregation for cross-attention
        cross_attention_output = attention_scores_cross @ V_slots  # Shape: (t+1, d_model)

        # Step 3: Feedforward network to predict the next patch
        # Process the cross-attention output to generate the next patch feature
        FFN_output = FeedForward(cross_attention_output[-1])  # Shape: (1, d_model)

        # Step 4: Append the predicted patch to the output sequence
        output_sequence.append(FFN_output)

        # Update the input sequence for the next step
        input_sequence = Concatenate(input_sequence, FFN_output)  # Shape: (t+2, d_model)

    # Return the full sequence of predicted patches
    return output_sequence
