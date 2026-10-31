# burn-mamba

The Mamba-1, Mamba-2 and Mamba-3 blocks for burn-stack, written only with
portable Burn tensor ops.

The terms of the containers (family, sequence, token, row, cache, segment, …)
are in the [burn-stack glossary](../burn-stack/GLOSSARY.md). The families of
this crate are Mamba-1, Mamba-2 and Mamba-3.

## Language

### The plant

**Plant**:
The state recurrence of a block: its transition, its write and its readout.
_Avoid_: SSM core, system (alone)

**State**:
The `h` of the plant: what the plant carries from one micro-step to the next.
_Avoid_: hidden state, latent state, SSM state, recurrent state

**Gate**:
The SiLU output gate `z` of a block.
_Avoid_: gate (for the gain, or for a tap that closes at token boundaries)

### SSD evaluation

**SSD path**:
The method that computes the chunkwise SSD scan: `Minimal`, `Serial` or
`SerialRecalculated`. All SSD paths give the same values and gradients.
_Avoid_: algorithm, variant, mode

**Pathway**:
How Mamba-3 maps its trapezoidal recurrence onto SSD calls. Double-SSD makes
one SSD call for each term of the trapezoid. Single-SSD makes one SSD call for
all terms. Each pathway works with each SSD path.
_Avoid_: algorithm, variant, mode, path

**Accumulator**:
The running sum that single-SSD keeps in place of the state. It is equal to the
state only at call boundaries.
_Avoid_: state (for the accumulator)

### Mamba-3 options

**MambaProduct**:
The Mamba-3 option that runs `u` micro-steps for each token.
_Avoid_: DeltaProduct (a different mechanism)

**MIMO**:
The Mamba-3 option in which the write and the readout of each head have rank
`m > 1` and share one state.

**MIMO rank**:
The rank `m` of the write and of the readout of each head.

**SISO**:
MIMO with rank 1. It is not a separate form.

### Time axes

**Micro-step**:
One recurrence step of the plant. Under MambaProduct, a token has `u`
micro-steps. Every other token has one.
_Avoid_: sub-step, inner step, step (alone)

**Write axis**:
The time axis with one entry for each micro-step.
_Avoid_: folded axis, sequence axis

**Read axis**:
The time axis with one entry for each token.

### The trapezoid

**Trapezoid**:
The discretisation rule of Mamba-3. It writes the current sample and the
samples at its taps into the state.

**Tap**:
An earlier sample that the `β` term of the write reads.

**Lag**:
The number of write positions between a tap and the current sample.

**Lag-1 tap**:
The tap one write position back: the previous micro-step.
_Avoid_: horizontal tap, interior tap, near tap

**Lag-`u` tap**:
The tap `u` write positions back: the same micro-step of the previous token. At
`u = 1`, it is the lag-1 tap.
_Avoid_: vertical tap, far tap

**Tap pattern**:
The choice of which taps exist, and whether the lag-1 tap closes at token
boundaries.
_Avoid_: trapezoid (for the choice)

**Mass**:
The share of the step size `Δ` that the write of one micro-step gives to one
sample. The masses of a micro-step add up to `Δ`.
_Avoid_: weight, coefficient

### Rotation

**Rotation**:
The orthogonal factor of the Mamba-3 state transition. It turns the state by a
data-dependent amount at each step. It is not a positional encoding.
_Avoid_: RoPE, positional encoding, complex transition

**RoPE trick**:
The technique that moves the accumulated rotation out of the state and into B
and C. The SSD scan then sees only a scalar decay.
_Avoid_: RoPE (as a name for the rotation)

**Rotary position embedding**:
The positional encoding of transformers: a fixed rotation whose angle counts
positions. The rotation of Mamba-3 is different, because its angle depends on
the data.
_Avoid_: vanilla RoPE, standard RoPE, RoPE

**Rotation kind**:
The group of the rotated part of the state: `Real1D`, `Complex2D`,
`Quaternion4D` or `Rotor4D`. `Real1D` is the trivial group, so "no rotation"
means `Real1D`.
_Avoid_: RoPE kind

**Rotation fraction**:
The share of the state that the rotation turns. The rest of the state does not
turn.
_Avoid_: RoPE fraction

**Rotation range**:
The bound on the angle of one micro-step, in units of `π·Δ`.

### Positive systems

**Positive system**:
A scalar recurrence for each head, next to the plant. It reads only the
in-projection and sets coefficients of the plant.
_Avoid_: side system, gate

**Gain**:
The positive system that sets the decay of the plant.
_Avoid_: gate, Kalman gate

**Tropical register**:
The positive system that adds a soft maximum `max(c + a, b)` to the readout.
_Avoid_: max-plus gate

### Packing

**Origin**:
The cache that each segment of a packed call starts from. With no origin, a
segment starts fresh. It is not the learnable initial state.
_Avoid_: initial cache, initial state
