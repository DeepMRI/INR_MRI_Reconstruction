import numpy as np


def grappa(kspaces_u, scale, kernel_shape=(4, 5)):

    num_sen, h, w = kspaces_u.shape

    """ Find True ACS """
    # Find the length of the true ACS domain, including lines due to downsampling
    finder = np.any(kspaces_u[0], axis=-1)
    # center region index
    idx_lower = np.max(np.where(finder[:int(h/2)] == 0)) + 1                        # the last all-zero row + 1 of the upper half of the graph
    idx_upper = np.min(np.where(finder[int(h/2):] == 0)) - 1 + int(h/2)             # the first all-zero row - 1 of the bottom half of the graph
    center_line_num = idx_upper - idx_lower

    matrix_h = scale * (kernel_shape[0] - 1) + 1
    assert matrix_h <= center_line_num

    """ Building the weight matrix """
    # Has equation of X * w = Y in mind
    # X = inMatrix, Y = outMatrix, w = weights
    # Each kernel adds one row to the X and Y matrices
    acsPhases = center_line_num - matrix_h + 1
    numKernels = acsPhases * w
    kernelSize = kernel_shape[0] * kernel_shape[1] * num_sen
    outSize = num_sen * (scale - 1)

    inMatrix = np.zeros([numKernels, kernelSize], dtype=complex)
    outMatrix = np.zeros([numKernels, outSize], dtype=complex)
    hkw = kernel_shape[1] // 2          # Half kernel width
    hkh = kernel_shape[0] // 2          # Half kernel height
    kidx = 0                            # "Kernel index" for counting the number of kernels.

    for i in np.arange(idx_lower, idx_upper)[:acsPhases]:
        phases = np.arange(i, i + matrix_h, scale)          # Phases of the kernel
        for j in range(w):
            freqs = np.arange(j - hkw, j + hkw + 1)         # Frequencies of the kernel
            freqs = freqs % w                               # For circular indexing

            selected = np.arange(phases[hkh-1] + 1, phases[hkh])
            selected = selected % h                         # Selected Y phases

            tempX = kspaces_u[:, phases[:, None], freqs[None, :]]
            tempY = kspaces_u[:, selected, j]

            # Filling in the matrices row by row
            inMatrix[kidx] = tempX.reshape([kernelSize, ])
            outMatrix[kidx] = tempY.reshape([outSize, ])

            kidx += 1
    # Calculate the weight matrix
    weights = np.linalg.pinv(inMatrix) @ outMatrix

    """ GRAPPA Reconstruction """
    # Performing a naive reconstruction according to first principles causes an overlap problem
    # The lines immediately before and after the ACS lines are not necessarily spaced with the sampling rate as the spacing
    # This causes alteration of the original data if GRAPPA reconstruction is performed naively
    # The solution is to perform reconstruction on a blank, and then overwrite all vlaues with the original data
    # This alleviates the problem of having to do special operations for the values at the edges
    # Also, the lines of k-space at the start or end of k-space may be neglected (depending on the implementation)
    # This requires shifting the finder by the sampling rate to look at the phase from one step above
    # If the downsampling does not match due to incorrect k-space dimensions, errors will be overwritten by the final correction process

    # Find the indices to fill, including at the beginning of k-space
    fill_finder = np.where(finder == 1)[0] + 1
    fill_finder = np.array(list(filter(lambda x: x < h, fill_finder)))
    # Shift from first fill line to beginning of kernel data
    upShift = (hkh - 1) * scale + 1
    # Shift from first fill line to end of kernel data
    downShift = hkh * scale - 1

    grappa_k = np.zeros_like(kspaces_u, dtype=complex)

    for i in fill_finder:
        phases = np.arange(i - upShift, i + downShift + 1, scale)
        phases = phases % h                                 # Circularly indexed phases
        for j in range(w):
            freqs = np.arange(j - hkw, j + hkw + 1)
            freqs = freqs % w                               # Circularly indexed frequencies

            kernel = kspaces_u[:, phases[:, None], freqs[None, :]].copy()
            # One line of the input matrix
            tempX = kernel.reshape([kernelSize, ])
            # One line of the output matrix
            tempY = tempX @ weights
            tempY = tempY.reshape([num_sen, scale - 1])

            # Selected lines of the output matrix to be filled in
            if phases[hkh-1] > phases[hkh]:
                selected = np.arange(phases[hkh-1] + 1, phases[hkh] + h)
            else:
                selected = np.arange(phases[hkh - 1] + 1, phases[hkh])
            selected = selected % h

            grappa_k[:, selected, j] = tempY.copy()

    # Filling in all the original data.
    # Doing it this way solves the edge overlap problem
    grappa_k[:, finder, :] = kspaces_u[:, finder, :].copy()

    return grappa_k

