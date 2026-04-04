#include "moe_topk.cuh"

// MoE router: for each token take Top-K experts by probability.
//
// logits:        [batchSize, numExperts], row-major, stride in elements (inputStride)
// outIdxs:  [batchSize, topK], int32 expert ids, stride idxsStride
// topkWeights:  [batchSize, topK], __half, stride outStride
//
// Tie-breaking when probabilities are equal: smaller expert index is preferred.

void MoeTopK(size_t batchSize, size_t numExperts, size_t topK, const __half* logits,
             size_t inputStride, int32_t* outIdxs, size_t idxsStride, __half* topkWeights,
             size_t outStride) {
    // YOUR CODE HERE
    // NB: no need to do any allocations here
    // NB: no explicit cudaDeviceSynchronize is required here
}
