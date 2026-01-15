# Speaker Attractor Generator Design

## Overview

The **Generator** is the second component of the speaker diarization pipeline, following the WavLeJEPA encoder. Its purpose is to hallucinate a set of **attractors** that represent speaker profiles from the encoded audio embeddings.

### Pipeline Position

```
Raw Audio → [WavLeJEPA Encoder] → Frame Embeddings → [Generator] → Speaker Attractors
                                       ↓
                               (N frames, 768d)          → (K attractors, D_attractor)
```

---

## Architecture

### High-Level Flow

```
WavLeJEPA (frozen) → Linear Attention Stack → Contextualized Frames [N, D]
                          (4-6 layers, gated)           ↓ KV
                                                  Cross-Attention ←── Q: GRU hidden
                                                        ↓
                                                  GRU step → attractor + confidence
                                                        ↑ input: [prev_attractor; cross_attn_output]
                                                  (loop until confidence < θ)
```

1. **Input**: Frame embeddings from frozen WavLeJEPA encoder `[N, 768]`
2. **Contextualization**: Linear attention stack processes frames into contextualized representations
3. **Iterative Generation**: GRU generates attractors one at a time, cross-attending to frames at each step
4. **Confidence-based Stopping**: Generate until confidence drops below threshold

### Components

#### 1. Linear Attention Stack

Processes WavLeJEPA frame embeddings through 4-6 layers of linear attention (Mamba-2 or RWKV-7 style) with standard gating. Output is a sequence of contextualized frame representations `[N, D]` that serve as keys/values for cross-attention.

**Properties:**
- O(N) complexity via parallel scan
- Gated activations for controlled information flow
- Maintains sequence length (no pooling)

#### 2. Cross-Attention Module

At each GRU step, the current GRU hidden state queries the contextualized frames:

```
Q = W_q @ h_t                    # [1, D] from GRU hidden
K = W_k @ contextualized_frames  # [N, D]
V = W_v @ contextualized_frames  # [N, D]
context_t = softmax(Q @ K.T / √D) @ V  # [1, D]
```

This provides fresh, attractor-specific context at each generation step.

#### 3. Recurrent Generator (GRU)

Generates attractors one at a time. At each step:

```
# Cross-attend to get fresh context
context_t = CrossAttention(query=h_{t-1}, kv=contextualized_frames)

# GRU input is concatenation of previous attractor and cross-attention output
x_t = [a_{t-1}; context_t]  # Step 0: [start_token; context_0]

# GRU update
h_t = GRU(x_t, h_{t-1})

# Output heads
a_t = MLP_attractor(h_t)  # Project to attractor space
c_t = σ(MLP_confidence(h_t))  # Scalar confidence ∈ (0, 1)
```

**Input composition:**
- Step 0: Learned start token concatenated with initial cross-attention context
- Step t > 0: Previous attractor `a_{t-1}` concatenated with fresh cross-attention context

#### 4. Confidence Head

A small MLP that outputs a scalar confidence ∈ (0, 1) for each generated attractor. Generation stops when confidence < threshold.

**Training concern**: How do we train the confidence head? Options:
- Supervise with speaker count (if available)
- Learn implicitly via the energy objective
- Use an auxiliary loss based on attractor utility

---

## Energy-Based Training

### Core Idea

We want attractors that explain the audio well. An **energy function** should be low when:
- Each frame embedding is close to at least one attractor
- Attractors are distinct (not collapsed)

### Proposed Energy Function

```
E(A, X) = E_assignment(A, X) + λ_sep * E_separation(A) + λ_cov * E_coverage(X, A)
```

Where:
- `A = {a_1, ..., a_K}` are the generated attractors
- `X = {x_1, ..., x_N}` are the frame embeddings

#### E_assignment: Frame-to-Attractor Assignment

Each frame should be well-explained by its closest attractor. Using **soft assignment with L2 distance**:

```
d_ik = ||x_i - a_k||²                 # L2 squared distance
w_ik = softmax(-d_ik / τ)            # soft assignment weights (softmin)
E_assignment = (1/N) * Σ_i Σ_k w_ik * d_ik
```

Soft assignment provides smooth gradients everywhere, enabling stable optimization. Temperature `τ` controls assignment sharpness (lower τ → harder assignments).

#### E_separation: Attractor Diversity

Attractors should not collapse. Using **hinge loss (margin-based)** to shut off the repulsive force once attractors are sufficiently distinct:

```
E_separation = Σ_{k≠j} max(0, margin - ||a_k - a_j||)
```

The margin-based formulation improves stability—once attractors are `margin` apart, no further gradient pushes them away. This prevents the separation term from dominating and lets the assignment term take over.

#### E_coverage: No Orphan Frames

Optional: Penalize attractors that don't explain any frames (encourages parsimony):

```
usage_k = Σ_i w_ik  # how many frames assigned to attractor k
E_coverage = Σ_k max(0, min_usage - usage_k)
```

---

## Test-Time Optimization

At inference, we refine the generated attractors by **minimizing the energy function** with respect to `A`:

```python
def refine_attractors(A_init, X, num_steps=50, lr=0.01):
    A = A_init
    for _ in range(num_steps):
        grad_A = grad(E, argnums=0)(A, X)
        A = A - lr * grad_A
    return A
```

This allows the model to:
- Fix minor errors in the initial generation
- Adapt to edge cases not seen during training
- Provide a principled optimization objective

### Considerations

- **Projection**: After gradient steps, may need to re-normalize attractors
- **Stopping criteria**: Use energy convergence, not just fixed steps
- **Initialization matters**: Good generator → fewer refinement steps needed

---

## Training Procedure

### Objective

```
L = E(A, X) + λ_conf * L_confidence
```

**Confidence loss (auxiliary usage-based):** Train the confidence head to predict whether each attractor is "useful" based on its actual assignment mass:

```
usage_k = Σ_i w_ik                           # total soft-assignment mass for attractor k
target_k = 1 if usage_k > threshold else 0   # e.g., threshold = frames equivalent to 0.5s audio
L_confidence = (1/K) * Σ_k BCE(c_k, target_k)
```

This ensures confidence is self-consistent with the energy function: "If I explain data, I am confident."

### Training Flow

1. Encode audio with **frozen** WavLeJEPA encoder
2. Pass frame embeddings through linear attention stack
3. Generate attractors with GRU (cross-attending at each step)
4. Compute energy loss
5. Backprop through generator (linear attention stack + cross-attention + GRU)

**Trainable components:** Linear attention stack, cross-attention, GRU, all output heads
**Frozen components:** WavLeJEPA encoder

### Questions to Resolve

- [ ] Do we need ground truth speaker labels for any part of training?
- [ ] How many attractors to generate during training (fixed max, or truly variable)?

---

## Implementation Sketch

```python
@dataclass
class GeneratorConfig:
    # Dimensions
    input_dim: int = 768          # WavLeJEPA output dim
    hidden_dim: int = 768         # GRU hidden size
    attractor_dim: int = 768      # Output attractor dimension
    
    # Linear attention stack
    num_layers: int = 4           # Number of linear attention layers (4-6)
    
    # Cross-attention
    num_heads: int = 8            # Multi-head attention heads
    
    # Generation
    max_attractors: int = 10      # Maximum speakers to generate
    confidence_threshold: float = 0.5
    
    # Energy weights
    lambda_separation: float = 1.0   # Weight for separation term (hinge loss)
    lambda_coverage: float = 0.1     # Weight for coverage term
    separation_margin: float = 1.0   # Margin for hinge loss (attractors must be this far apart)
    
    # Temperature annealing (deterministic annealing)
    tau_start: float = 1.0           # Initial temperature (soft assignment)
    tau_end: float = 0.1             # Final temperature (hard assignment)
    
    # Confidence training
    usage_threshold: float = 0.5     # Seconds of audio an attractor must explain to be "useful"


class LinearAttentionLayer(eqx.Module):
    """Single layer of gated linear attention (Mamba-2/RWKV-7 style)."""
    
    # ... implementation details (parallel scan, gating, etc.)
    
    def __call__(self, x: Array) -> Array:
        """Process sequence with O(N) linear attention."""
        ...


class LinearAttentionStack(eqx.Module):
    """Stack of gated linear attention layers."""
    
    layers: list[LinearAttentionLayer]
    
    def __call__(self, x: Array) -> Array:
        """Process frame embeddings through all layers.
        
        Args:
            x: [N, D] frame embeddings
            
        Returns:
            [N, D] contextualized frame representations
        """
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttention(eqx.Module):
    """Multi-head cross-attention for querying contextualized frames."""
    
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    
    w_q: eqx.nn.Linear  # query projection
    w_k: eqx.nn.Linear  # key projection
    w_v: eqx.nn.Linear  # value projection
    w_o: eqx.nn.Linear  # output projection
    
    def __call__(self, query: Array, kv: Array) -> Array:
        """
        Args:
            query: [D] GRU hidden state
            kv: [N, D] contextualized frames
            
        Returns:
            [D] attended context vector
        """
        Q = self.w_q(query)                    # [num_heads * head_dim]
        K = jax.vmap(self.w_k)(kv)             # [N, num_heads * head_dim]
        V = jax.vmap(self.w_v)(kv)             # [N, num_heads * head_dim]
        
        # Reshape for multi-head attention
        Q = Q.reshape(self.num_heads, self.head_dim)      # [H, head_dim]
        K = K.reshape(-1, self.num_heads, self.head_dim)  # [N, H, head_dim]
        V = V.reshape(-1, self.num_heads, self.head_dim)  # [N, H, head_dim]
        
        # Scaled dot-product attention per head
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(jnp.einsum('hd,nhd->nh', Q, K) * scale, axis=0)  # [N, H]
        out = jnp.einsum('nh,nhd->hd', attn, V)  # [H, head_dim]
        
        # Concatenate heads and project
        return self.w_o(out.reshape(-1))  # [D]


class AttractorGenerator(eqx.Module):
    """Generate speaker attractors from frame embeddings.
    
    Architecture:
    1. Linear attention stack contextualizes frame embeddings
    2. GRU generates attractors iteratively
    3. Cross-attention queries frames at each GRU step
    4. Confidence head determines when to stop
    """
    
    config: GeneratorConfig = eqx.field(static=True)
    
    # Frame processing
    linear_attn_stack: LinearAttentionStack
    
    # Cross-attention (queries frames at each GRU step)
    cross_attn: CrossAttention
    
    # GRU generator (input: prev_attractor concat cross_attn_output)
    gru_cell: eqx.nn.GRUCell  # input_size = attractor_dim + hidden_dim
    start_token: Array        # [attractor_dim] learned start token
    
    # Output heads
    attractor_head: eqx.nn.MLP   # hidden_dim -> attractor_dim
    confidence_head: eqx.nn.MLP  # hidden_dim -> 1
    
    def __call__(self, frame_embeddings: Array, *, key) -> tuple[Array, Array, Array]:
        """
        Args:
            frame_embeddings: [N, input_dim] from frozen WavLeJEPA encoder
            
        Returns:
            attractors: [max_attractors, attractor_dim] padded array
            confidences: [max_attractors] padded with 0 for invalid entries
            valid_count: scalar, number of valid attractors (for downstream masking)
        """
        cfg = self.config
        max_K = cfg.max_attractors
        
        # Contextualize frames through linear attention stack (done once)
        contextualized = self.linear_attn_stack(frame_embeddings)  # [N, D]
        
        # Initialize GRU hidden state (e.g., mean pool or learned)
        h_init = jnp.mean(contextualized, axis=0)  # [D]
        
        # While loop state
        init_state = (
            self.start_token,                       # prev_attractor: [attractor_dim]
            h_init,                                 # h: GRU hidden [hidden_dim]
            jnp.zeros((max_K, cfg.attractor_dim)),  # attractors buffer
            jnp.zeros((max_K,)),                    # confidences buffer
            jnp.array(0),                           # step counter
            jnp.array(True),                        # continue flag
        )
        
        def cond(state):
            _, _, _, _, step, cont = state
            return cont & (step < max_K)
        
        def body(state):
            prev_attractor, h, attractors, confs, step, _ = state
            
            # Cross-attend: GRU hidden queries contextualized frames
            context = self.cross_attn(query=h, kv=contextualized)  # [D]
            
            # GRU input: concatenate previous attractor with cross-attention context
            x = jnp.concatenate([prev_attractor, context])  # [attractor_dim + D]
            
            # GRU step
            h_new = self.gru_cell(x, h)
            
            # Generate attractor and confidence
            a = self.attractor_head(h_new)
            c = jax.nn.sigmoid(self.confidence_head(h_new).squeeze())
            
            # Update buffers
            attractors = attractors.at[step].set(a)
            confs = confs.at[step].set(c)
            
            return (a, h_new, attractors, confs, step + 1, c > cfg.confidence_threshold)
        
        final = jax.lax.while_loop(cond, body, init_state)
        _, _, attractors, confidences, valid_count, _ = final
        
        return attractors, confidences, valid_count
```

---

## Design Decisions (Resolved)

| Decision | Choice |
|----------|--------|
| Attractor dimension | 768d (same as frame embeddings) |
| Variable-length generation | `jax.lax.while_loop` with early stopping; output padded to `max_attractors` with `valid_count` for downstream masking |
| Assignment type | Soft assignment (softmin) |
| Distance metric | L2 squared distance |
| GRU hidden initialization | Mean-pooled contextualized frames |
| Energy term balancing | Fixed hyperparameters: `λ_sep=1.0` (with margin-based hinge loss), `λ_cov=0.1` |
| Softmin temperature `τ` | Annealed (deterministic annealing): start high (soft), linearly decay to low (hard) over training |
| Confidence head training | Auxiliary usage loss: BCE against binary target derived from soft-assignment mass (threshold ≈ 0.5s audio) |

## Open Questions

None currently—architecture and training strategy are fully specified. Ready for implementation.

---

## Next Steps

1. Finalize architecture decisions
2. Implement `AttractorGenerator` module
3. Implement energy functions
4. Implement test-time optimization loop
5. Set up training with synthetic multi-speaker data
