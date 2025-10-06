#pragma once

#include <cuda_runtime.h>
#include <vector>

struct EnvironmentMap {
    cudaTextureObject_t texture = 0;
    cudaArray_t array = nullptr; 
    int width = 0;
    int height = 0;
    float* P = nullptr;     
    float* q = nullptr;     
    int*   alias = nullptr; 
};

struct EnvAliasHost {
    std::vector<float> P;
    std::vector<float> q;
    std::vector<int> alias;
};

struct EnvImageHost {
    int width = 0;
    int height = 0;
    std::vector<float4> texels; 
};

EnvImageHost loadHDR(const char* filepath);
EnvAliasHost buildEnvAlias(const float4* texels, int W, int H);
void uploadEnv(const EnvImageHost& img, const EnvAliasHost& alias, EnvironmentMap& env);
void freeEnv(EnvironmentMap& env);

#ifdef __CUDACC__
#include <glm/glm.hpp>
#include <math_constants.h>

extern __device__ EnvironmentMap d_env;

__device__ glm::vec3 sampleEnvAlias(const EnvironmentMap& env, float u1, float u2,
                                    glm::vec3& wi, float& pdf);
__device__ float envPdfAlias(const EnvironmentMap& env, const glm::vec3& dir);

__device__ inline void  dirToUV(const glm::vec3& d, float& u, float& v);
__device__ inline glm::vec3 uvToDir(float u, float v);
__device__ inline glm::vec3 envTex(cudaTextureObject_t tex, float u, float v);

__device__ __forceinline__ glm::vec3 sampleEnvAlias(const EnvironmentMap& env, float u1, float u2,
                                                    glm::vec3& wi, float& pdf)
{
    const int W = env.width;
    const int H = env.height;
    const int N = W * H;
    
    int i = min(int(u1 * N), N - 1);
    float t = u2;
    int pick = (t < env.q[i]) ? i : env.alias[i];

    int x = pick % W;
    int y = pick / W;
    float u = (x + 0.5f) / float(W);
    float v = (y + 0.5f) / float(H);

    wi = uvToDir(u, v);

    float theta = v * CUDART_PI_F;
    float sinT = fmaxf(sinf(theta), 1e-6f);
    float Ppick = env.P[pick];
    pdf = (Ppick * float(N)) / (2.0f * CUDART_PI_F * CUDART_PI_F * sinT);

    return envTex(env.texture, u, v);
}

__device__ __forceinline__ float envPdfAlias(const EnvironmentMap& env, const glm::vec3& dir) {
    float u, v;
    dirToUV(dir, u, v);
    
    int x = min(int(u * env.width), env.width - 1);
    int y = min(int(v * env.height), env.height - 1);
    int pick = y * env.width + x;
    
    float theta = v * CUDART_PI_F;
    float sinT = fmaxf(sinf(theta), 1e-6f);
    
    return (env.P[pick] * float(env.width * env.height)) /
           (2.0f * CUDART_PI_F * CUDART_PI_F * sinT);
}

__device__ inline void dirToUV(const glm::vec3& d, float& u, float& v) {
    glm::vec3 n = glm::normalize(d);
    float phi = atan2f(n.z, n.x);
    if (phi < 0) {
        phi += 2.0f * CUDART_PI_F;
    }
    float theta = acosf(fminf(fmaxf(n.y, -1.0f), 1.0f));
    u = phi / (2.0f * CUDART_PI_F);
    v = theta / CUDART_PI_F;
}

__device__ inline glm::vec3 uvToDir(float u, float v) {
    float phi = u * 2.0f * CUDART_PI_F;
    float theta = v * CUDART_PI_F;
    float st = sinf(theta);
    float ct = cosf(theta);
    float sp = sinf(phi);
    float cp = cosf(phi);
    return glm::vec3(st * cp, ct, st * sp);
}

__device__ inline glm::vec3 envTex(cudaTextureObject_t tex, float u, float v) {
    float4 t = tex2D<float4>(tex, u, v);
    return glm::vec3(t.x, t.y, t.z);
}

#endif


