#include "env.h"

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <stb_image.h>

static inline float Luma(const float4& c){ return 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z; }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

EnvImageHost loadHDR(const char* filepath){
    EnvImageHost img;
    int w=0,h=0,nc=0;
    stbi_set_flip_vertically_on_load(false);
    float* data = stbi_loadf(filepath, &w, &h, &nc, 3);
    if (!data || w<=0 || h<=0){
        if (data) stbi_image_free(data);
        throw std::runtime_error("Failed to load HDR image");
    }
    img.width = w; img.height = h;
    img.texels.resize(w*h);
    for (int i=0;i<w*h;i++){
        float r=data[3*i+0], g=data[3*i+1], b=data[3*i+2];
        img.texels[i] = make_float4(r,g,b,1.0f);
    }
    stbi_image_free(data);
    return img;
}

EnvAliasHost buildEnvAlias(const float4* texels, int W, int H){
    EnvAliasHost out;
    const int N = W*H;
    out.P.resize(N); out.q.resize(N); out.alias.resize(N);

    double total = 0.0;
    for (int y=0; y<H; ++y){
        float v = (y + 0.5f) / float(H);
        float theta = v * float(M_PI);
        float s = std::max(std::sin(theta), 0.0f);
        for (int x=0; x<W; ++x){
            int i = y*W + x;
            float w = Luma(texels[i]) * s;
            out.P[i] = w;
            total += w;
        }
    }
    if (total <= 0.0) {
        for (int i=0;i<N;++i) out.P[i] = 1.0f/float(N);
    } else {
        for (int i=0;i<N;++i) out.P[i] = float(out.P[i]/total);
    }

    std::vector<int> small; small.reserve(N);
    std::vector<int> large; large.reserve(N);
    std::vector<float> Pn(N);
    for (int i=0;i<N;++i){
        Pn[i] = out.P[i]*N;
        (Pn[i] < 1.0f ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()){
        int a = small.back(); small.pop_back();
        int b = large.back(); large.pop_back();
        out.q[a] = Pn[a];
        out.alias[a] = b;
        Pn[b] = (Pn[b] + Pn[a]) - 1.0f;
        (Pn[b] < 1.0f ? small : large).push_back(b);
    }
    while (!large.empty()) { int b = large.back(); large.pop_back(); out.q[b]=1.0f; out.alias[b]=b; }
    while (!small.empty()) { int a = small.back(); small.pop_back(); out.q[a]=1.0f; out.alias[a]=a; }
    return out;
}

void uploadEnv(const EnvImageHost& img, const EnvAliasHost& alias, EnvironmentMap& env){
    // Copy alias arrays
    const int N = img.width * img.height;
    env.width = img.width; env.height = img.height;

    cudaMalloc(&env.P, N * sizeof(float));
    cudaMalloc(&env.q, N * sizeof(float));
    cudaMalloc(&env.alias, N * sizeof(int));
    cudaMemcpy(env.P, alias.P.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(env.q, alias.q.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(env.alias, alias.alias.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA array for texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    env.array = nullptr;
    cudaMallocArray(&env.array, &channelDesc, img.width, img.height, 0);
    cudaMemcpy2DToArray(env.array, 0, 0, img.texels.data(), img.width*sizeof(float4), img.width*sizeof(float4), img.height, cudaMemcpyHostToDevice);

    // Texture resource/view descriptions
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = env.array;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;   // U wrap
    texDesc.addressMode[1] = cudaAddressModeClamp;  // V clamp
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&env.texture, &resDesc, &texDesc, nullptr);
}

void freeEnv(EnvironmentMap& env){
    if (env.P) cudaFree(env.P), env.P=nullptr;
    if (env.q) cudaFree(env.q), env.q=nullptr;
    if (env.alias) cudaFree(env.alias), env.alias=nullptr;
    if (env.texture){
        cudaDestroyTextureObject(env.texture);
        env.texture = 0;
    }
    if (env.array){
        cudaFreeArray(env.array);
        env.array = nullptr;
    }
}


