#include "bvh.hpp"

#include <algorithm>
#include <limits>

namespace {

struct BBox {
    glm::vec3 bmin{ std::numeric_limits<float>::infinity() };
    glm::vec3 bmax{ -std::numeric_limits<float>::infinity() };
    
    void expand(const glm::vec3& p) {
        bmin = glm::min(bmin, p);
        bmax = glm::max(bmax, p);
    }
    
    void expand(const BBox& b) {
        expand(b.bmin);
        expand(b.bmax);
    }
    
    glm::vec3 center() const {
        return 0.5f * (bmin + bmax);
    }
    
    glm::vec3 extent() const {
        return bmax - bmin;
    }
    
    float surfaceArea() const {
        glm::vec3 e = glm::max(extent(), glm::vec3(0.0f));
        return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
};

struct BuildTri {
    int triIndex; // index into Scene::triangles
    BBox bbox;
    glm::vec3 centroid;
    
    BuildTri() : triIndex(0), bbox(), centroid(0.0f) {}
};

struct Task {
    int start;
    int count;
    int nodeIndex;
};

}

static BBox triBBoxObject(const Triangle& t, const std::vector<Vertex>& V) {
    const glm::vec3& a = V[t.idx_v0].pos;
    const glm::vec3& b = V[t.idx_v1].pos;
    const glm::vec3& c = V[t.idx_v2].pos;
    
    BBox bxp;
    bxp.expand(a);
    bxp.expand(b);
    bxp.expand(c);
    
    return bxp;
}

BuiltBVH buildMeshBVH(const std::vector<Triangle>& triangles,
                      const std::vector<Vertex>& vertices,
                      int triOffset,
                      int triCount,
                      int leafSize,
                      int numBuckets)
{
    BuiltBVH out;
    if (triCount <= 0) return out;

    std::vector<BuildTri> tris;
    tris.reserve(triCount);
    
    for (int i = 0; i < triCount; ++i) {
        const Triangle& T = triangles[triOffset + i];
        BuildTri bt;
        bt.triIndex = triOffset + i;
        bt.bbox = triBBoxObject(T, vertices);
        bt.centroid = bt.bbox.center();
        tris.push_back(bt);
    }

    out.nodes.reserve(std::max(1, triCount*2));
    out.triIndices.resize(triCount);

    // Iterative build
    out.nodes.push_back({make_float3(0, 0, 0), make_float3(0, 0, 0), -1, -1, 0, 0});
    
    std::vector<Task> stack;
    stack.push_back({0, triCount, 0});

    while (!stack.empty()) {
        Task task = stack.back();
        stack.pop_back();
        const int start = task.start;
        const int count = task.count;
        const int nodeIdx = task.nodeIndex;

        // Compute bounding box for this node
        BBox nodeBox;
        for (int i = 0; i < count; ++i) {
            nodeBox.expand(tris[start + i].bbox);
        }
        
        out.nodes[nodeIdx].bboxMin = make_float3(nodeBox.bmin.x, nodeBox.bmin.y, nodeBox.bmin.z);
        out.nodes[nodeIdx].bboxMax = make_float3(nodeBox.bmax.x, nodeBox.bmax.y, nodeBox.bmax.z);

        if (count <= leafSize) {
            // Create leaf node
            out.nodes[nodeIdx].start = start;
            out.nodes[nodeIdx].count = count;
            out.nodes[nodeIdx].left = -1;
            out.nodes[nodeIdx].right = -1;
            continue;
        }

        // Choose split by binned SAH
        glm::vec3 ext = nodeBox.extent();
        int axis = 0;
        if (ext.y > ext.x && ext.y >= ext.z) {
            axis = 1;
        } else if (ext.z > ext.x && ext.z >= ext.y) {
            axis = 2;
        }
        
        const float minA = nodeBox.bmin[axis];
        const float maxA = nodeBox.bmax[axis];
        
        if (maxA <= minA) {
            //  fallback to median split
            std::nth_element(tris.begin() + start, tris.begin() + start + count/2, tris.begin() + start + count,
                [axis](const BuildTri& a, const BuildTri& b) {
                    return a.centroid[axis] < b.centroid[axis];
                });
            
            int mid = start + count / 2;
            int leftIdx = (int)out.nodes.size();
            int rightIdx = leftIdx + 1;
            
            out.nodes.push_back({});
            out.nodes.push_back({});
            out.nodes[nodeIdx].left = leftIdx;
            out.nodes[nodeIdx].right = rightIdx;
            out.nodes[nodeIdx].count = 0;
            
            stack.push_back({mid, start + count - mid, rightIdx});
            stack.push_back({start, mid - start, leftIdx});
            continue;
        }

        const int B = std::max(8, std::min(numBuckets, 32));
        
        struct Bucket {
            int n = 0;
            BBox b;
        };
        
        std::vector<Bucket> buckets(B);
        const float scale = (float)B / (maxA - minA + 1e-8f);
        
        for (int i = 0; i < count; ++i) {
            int bi = std::min(B - 1, std::max(0, (int)((tris[start + i].centroid[axis] - minA) * scale)));
            buckets[bi].n++;
            buckets[bi].b.expand(tris[start + i].bbox);
        }

        // Prefix/suffix to evaluate splits
        std::vector<int> nL(B);
        std::vector<int> nR(B);
        std::vector<BBox> bL(B);
        std::vector<BBox> bR(B);
        
        BBox acc;
        int accN = 0;
        for (int i = 0; i < B; ++i) {
            acc.expand(buckets[i].b);
            accN += buckets[i].n;
            bL[i] = acc;
            nL[i] = accN;
        }
        
        acc = BBox{};
        accN = 0;
        for (int i = B - 1; i >= 0; --i) {
            acc.expand(buckets[i].b);
            accN += buckets[i].n;
            bR[i] = acc;
            nR[i] = accN;
        }

        float bestCost = std::numeric_limits<float>::infinity();
        int bestSplit = -1;
        
        for (int s = 0; s < B - 1; ++s) {
            if (nL[s] == 0 || nR[s + 1] == 0) continue;
            
            float cost = nL[s] * bL[s].surfaceArea() + nR[s + 1] * bR[s + 1].surfaceArea();
            if (cost < bestCost) {
                bestCost = cost;
                bestSplit = s;
            }
        }

        if (bestSplit < 0) {
            // Fallback to median split
            std::nth_element(tris.begin() + start, tris.begin() + start + count/2, tris.begin() + start + count,
                [axis](const BuildTri& a, const BuildTri& b) {
                    return a.centroid[axis] < b.centroid[axis];
                });
            
            int mid = start + count / 2;
            int leftIdx = (int)out.nodes.size();
            int rightIdx = leftIdx + 1;
            
            out.nodes.push_back({});
            out.nodes.push_back({});
            out.nodes[nodeIdx].left = leftIdx;
            out.nodes[nodeIdx].right = rightIdx;
            out.nodes[nodeIdx].count = 0;
            
            stack.push_back({mid, start + count - mid, rightIdx});
            stack.push_back({start, mid - start, leftIdx});
            continue;
        }

        // Partition by bucket threshold
        float splitCoord = minA + (bestSplit + 1) * (maxA - minA) / (float)B;
        auto midIt = std::partition(tris.begin() + start, tris.begin() + start + count,
            [axis, splitCoord](const BuildTri& t) {
                return t.centroid[axis] < splitCoord;
            });
        
        int mid = (int)(midIt - tris.begin());
        if (mid == start || mid == start + count) {
            // force median
            mid = start + count / 2;
        }

        int leftIdx = (int)out.nodes.size();
        int rightIdx = leftIdx + 1;
        
        out.nodes.push_back({});
        out.nodes.push_back({});
        out.nodes[nodeIdx].left = leftIdx;
        out.nodes[nodeIdx].right = rightIdx;
        out.nodes[nodeIdx].count = 0;
        
        stack.push_back({mid, start + count - mid, rightIdx});
        stack.push_back({start, mid - start, leftIdx});
    }

    // Fill triangle index array with the order in tris[]
    for (int i = 0; i < triCount; ++i) {
        out.triIndices[i] = tris[i].triIndex;
    }
    
    return out;
}


