# This is a file where you should put your own functions

# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

# TODO: Datasets go here.

# -----------------------------------------------------------------------------
# Network architectures
# -----------------------------------------------------------------------------

# TODO: Define network architectures here

def create_network(arch, **kwargs):
    # TODO: Change this function for the architectures you want to support
    if arch == 'arch1':
        return create_network_arch1(**kwargs)
    elif arch == 'arch2':
        return create_network_arch2(**kwargs)
    else:
        raise Exception(f"Unknown architecture: {arch}")

# -----------------------------------------------------------------------------
# Training and testing loops
# -----------------------------------------------------------------------------

# TODO: Define training, testing and model loading here

# -----------------------------------------------------------------------------
# Pruning
# -----------------------------------------------------------------------------

# TODO: Put functions related to pruning here


