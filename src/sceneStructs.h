#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec2 uv;
    
    __host__ __device__ Vertex() : uv(0.0f, 0.0f) {}
    __host__ __device__ Vertex(glm::vec3 p, glm::vec3 n) : pos(p), nor(n), uv(0.0f, 0.0f) {}
    __host__ __device__ Vertex(glm::vec3 p, glm::vec3 n, glm::vec2 t) : pos(p), nor(n), uv(t) {}
};

struct Triangle
{
    int idx_v0, idx_v1, idx_v2;  // Vertex indices
    int materialId;
    
    __host__ __device__ Triangle() : idx_v0(0), idx_v1(0), idx_v2(0), materialId(0) {}
    __host__ __device__ Triangle(int v0, int v1, int v2, int matId) 
        : idx_v0(v0), idx_v1(v1), idx_v2(v2), materialId(matId) {}
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    
    int numTriangles;
    int triangleOffset;
    int numVertices;
    int vertexOffset;   
    glm::vec3 bboxMin;  
    glm::vec3 bboxMax;  

    int bvhNodeOffset = 0;
    int bvhNodeCount = 0;
    int bvhTriIndexOffset = 0;
    int bvhTriIndexCount = 0;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
    
    int textureID;           
    int hasTexture;          
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    
    float lensRadius;
    float focalDistance;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;  
};
