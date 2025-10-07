#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sceneStructs.h"

struct BVHNodeGPU {
    float3 bboxMin;
    float3 bboxMax;
    int left;   
    int right; 
    int start; 
    int count;  
};

struct BuiltBVH {
    std::vector<BVHNodeGPU> nodes;
    std::vector<int> triIndices; 
};

BuiltBVH buildMeshBVH(const std::vector<Triangle>& triangles,
                      const std::vector<Vertex>& vertices,
                      int triOffset,
                      int triCount,
                      int leafSize = 8,
                      int numBuckets = 16);


