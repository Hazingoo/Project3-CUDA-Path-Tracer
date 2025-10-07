#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct TextureData {
    cudaTextureObject_t texObj = 0;
    cudaArray_t array = nullptr;
    int width = 0;
    int height = 0;
    std::string filepath;
};

struct ImageHost {
    int width = 0;
    int height = 0;
    std::vector<float4> pixels;
};

ImageHost loadImage(const char* filepath);

void uploadTexture(const ImageHost& img, TextureData& tex);

void freeTexture(TextureData& tex);

#ifdef __CUDACC__

__device__ inline glm::vec3 sampleTexture(cudaTextureObject_t texObj, glm::vec2 uv) {
    float4 color = tex2D<float4>(texObj, uv.x, uv.y);
    return glm::vec3(color.x, color.y, color.z);
}

__device__ inline glm::vec3 sampleTextureBilinear(cudaTextureObject_t texObj, 
                                                   glm::vec2 uv, 
                                                   int width, 
                                                   int height) {
    uv.x = uv.x - floorf(uv.x);
    uv.y = uv.y - floorf(uv.y);
    
    float x = uv.x * width;
    float y = uv.y * height;
    
    x -= 0.5f;
    y -= 0.5f;
    
    int ix = int(floorf(x));
    int iy = int(floorf(y));
    
    float fx = x - ix;
    float fy = y - iy;
    
    float u0 = (ix + 0.5f) / width;
    float u1 = (ix + 1.5f) / width;
    float v0 = (iy + 0.5f) / height;
    float v1 = (iy + 1.5f) / height;
    
    float4 s00 = tex2D<float4>(texObj, u0, v0);
    float4 s10 = tex2D<float4>(texObj, u1, v0);
    float4 s01 = tex2D<float4>(texObj, u0, v1);
    float4 s11 = tex2D<float4>(texObj, u1, v1);
    
    glm::vec3 c00(s00.x, s00.y, s00.z);
    glm::vec3 c10(s10.x, s10.y, s10.z);
    glm::vec3 c01(s01.x, s01.y, s01.z);
    glm::vec3 c11(s11.x, s11.y, s11.z);
    
    glm::vec3 c0 = glm::mix(c00, c10, fx);
    glm::vec3 c1 = glm::mix(c01, c11, fx);
    return glm::mix(c0, c1, fy);
}

#endif

