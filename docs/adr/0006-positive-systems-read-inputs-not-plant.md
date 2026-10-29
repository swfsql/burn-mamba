# A positive system reads the inputs, never the plant

The positive systems (`mamba3/positive/`) set the coefficients of the plant,
but they read only the in-projection, never the plant state. So they are a
cascade in front of the LPV plant, and the chunkwise pass still works. A gain
that read the state would break the chunkwise pass.
