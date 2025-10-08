#include "texture.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#include <stb_image.h>

ImageHost loadImage(const char* filepath) {
    ImageHost img;
    int w = 0;
    int h = 0;
    int nc = 0;
    
    bool isHDR = stbi_is_hdr(filepath);
    
    if (isHDR) {
        // Load HDR image
        stbi_set_flip_vertically_on_load(false);
        float* data = stbi_loadf(filepath, &w, &h, &nc, 3);
        
        if (!data || w <= 0 || h <= 0) {
            if (data) stbi_image_free(data);
            throw std::runtime_error(std::string("Failed to load HDR texture: ") + filepath);
        }
        
        img.width = w;
        img.height = h;
        img.pixels.resize(w * h);
        
        for (int i = 0; i < w * h; i++) {
            float r = data[3 * i + 0];
            float g = data[3 * i + 1];
            float b = data[3 * i + 2];
            img.pixels[i] = make_float4(r, g, b, 1.0f);
        }
        
        stbi_image_free(data);
    } else {
        stbi_set_flip_vertically_on_load(false);
        unsigned char* data = stbi_load(filepath, &w, &h, &nc, 3);
        
        if (!data || w <= 0 || h <= 0) {
            if (data) stbi_image_free(data);
            throw std::runtime_error(std::string("Failed to load texture: ") + filepath);
        }
        
        img.width = w;
        img.height = h;
        img.pixels.resize(w * h);
        
        for (int i = 0; i < w * h; i++) {
            float r = data[3 * i + 0] / 255.0f;
            float g = data[3 * i + 1] / 255.0f;
            float b = data[3 * i + 2] / 255.0f;
            
            auto srgbToLinear = [](float c) {
                return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
            };
            
            r = srgbToLinear(r);
            g = srgbToLinear(g);
            b = srgbToLinear(b);
            
            img.pixels[i] = make_float4(r, g, b, 1.0f);
        }
        
        stbi_image_free(data);
    }
    
    std::cout << "Loaded texture: " << filepath 
              << " (" << img.width << "x" << img.height << ")" << std::endl;
    
    return img;
}

void uploadTexture(const ImageHost& img, TextureData& tex) {
    tex.width = img.width;
    tex.height = img.height;
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&tex.array, &channelDesc, img.width, img.height, 0);
    cudaMemcpy2DToArray(tex.array, 0, 0, img.pixels.data(),
                        img.width * sizeof(float4), 
                        img.width * sizeof(float4),
                        img.height, 
                        cudaMemcpyHostToDevice);
    
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.array;
    
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;   
    texDesc.addressMode[1] = cudaAddressModeWrap;   
    texDesc.filterMode = cudaFilterModeLinear;      
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;                    
    
    cudaCreateTextureObject(&tex.texObj, &resDesc, &texDesc, nullptr);
    
    std::cout << "Uploaded texture to GPU: " 
              << img.width << "x" << img.height << std::endl;
}

void freeTexture(TextureData& tex) {
    if (tex.texObj) {
        cudaDestroyTextureObject(tex.texObj);
        tex.texObj = 0;
    }
    if (tex.array) {
        cudaFreeArray(tex.array);
        tex.array = nullptr;
    }
}

