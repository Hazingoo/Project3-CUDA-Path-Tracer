#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"

struct BVHNodeGPU {
    float3 bboxMin;
    float3 bboxMax;
    int left;   // index of left child or -1
    int right;  // index of right child or -1
    int start;  // start index into triangle index array
    int count;  // number of triangles if leaf; 0 for interior
};

struct BuiltBVH {
    std::vector<BVHNodeGPU> nodes;
    std::vector<int> triIndices; // reordered triangle indices (into Scene::triangles)
};

// Build a per-mesh BVH over triangles in [triOffset, triOffset+triCount)
// Triangles and vertices are in the mesh's object space (as stored in Scene).
BuiltBVH buildMeshBVH(const std::vector<Triangle>& triangles,
                      const std::vector<Vertex>& vertices,
                      int triOffset,
                      int triCount,
                      int leafSize = 8,
                      int numBuckets = 16);


