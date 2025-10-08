#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#pragma region ThrustFunctors
struct IsPathAlive {
    __host__ __device__ bool operator()(const PathSegment& p) const {
        return p.remainingBounces > 0;
    }
};
#pragma endregion

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "env.h"
#include "bvh.hpp"
#include "texture.h"

extern __device__ BVHNodeGPU* d_bvh_nodes;
extern __device__ int* d_bvh_tri_indices;

#define ERRORCHECK 1

#ifndef PI
#define PI 3.14159265359f
#endif

// Russian Roulette parameters 
#define RR_MIN_DEPTH 3

// BVH toggle 
#define USE_BVH true


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];
        
        glm::vec3 avgColor = pix / (float)iter;
        float exposure = 4.0f;  
        avgColor *= exposure;
        
        glm::vec3 toneMapped = avgColor / (glm::vec3(1.0f) + avgColor);
        
        auto gammaCorrect = [](float c) {
            return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
        };
        toneMapped = glm::vec3(gammaCorrect(toneMapped.x), gammaCorrect(toneMapped.y), gammaCorrect(toneMapped.z));
        
        glm::ivec3 color;
        color.x = glm::clamp((int)(toneMapped.x * 255.0f), 0, 255);
        color.y = glm::clamp((int)(toneMapped.y * 255.0f), 0, 255);
        color.z = glm::clamp((int)(toneMapped.z * 255.0f), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static int* dev_material_sort_indices = NULL;
static Vertex* dev_vertices = NULL;
static Triangle* dev_triangles = NULL;
static EnvironmentMap dev_env = {};
static BVHNodeGPU* dev_bvh_nodes = NULL;
static int* dev_bvh_tri_indices = NULL;
static cudaTextureObject_t* dev_textures = NULL;  // Array of texture objects for materials

__device__ inline float luminance(const glm::vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ inline glm::vec3 clampLuminance(const glm::vec3& c, float maxL) {
    float L = luminance(c);
    if (L > maxL) {
        float s = maxL / fmaxf(L, 1e-6f);
        return c * s;
    }
    return c;
}

// Device global definitions
__device__ BVHNodeGPU* d_bvh_nodes;
__device__ int* d_bvh_tri_indices;

// Host copy mirrors device-global d_env; not read inside device code
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_material_sort_indices, pixelcount * sizeof(int));

    // Initialize vertex and triangle data for meshes
    if (!scene->vertices.empty()) {
        cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
        cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);
    }
    if (!scene->triangles.empty()) {
        cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    // Upload BVH buffers if present 
    if (!scene->bvhNodes.empty()) {
        cudaMalloc(&dev_bvh_nodes, scene->bvhNodes.size() * sizeof(BVHNodeGPU));
        cudaMemcpy(dev_bvh_nodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice);
        
        // Calculate BVH statistics for debugging
        int totalNodes = scene->bvhNodes.size();
        int leafNodes = 0;
        for (const auto& node : scene->bvhNodes) {
            if (node.count > 0) { // leaf node
                leafNodes++;
            }
        }
        printf("=== BVH Statistics ===\n");
        printf("Total BVH nodes: %d\n", totalNodes);
        printf("Leaf nodes: %d\n", leafNodes);
        printf("Internal nodes: %d\n", totalNodes - leafNodes);
        printf("Total triangles: %zu\n", scene->triangles.size());
        printf("BVH triangle indices: %zu\n", scene->bvhTriIndices.size());
    }
    if (!scene->bvhTriIndices.empty()) {
        cudaMalloc(&dev_bvh_tri_indices, scene->bvhTriIndices.size() * sizeof(int));
        cudaMemcpy(dev_bvh_tri_indices, scene->bvhTriIndices.data(), scene->bvhTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    
#if USE_BVH
    printf("Using BVH acceleration \n");
    if (scene->bvhNodes.empty()) {
        printf("WARNING: BVH enabled but no BVH data found, will fall back to brute force\n");
    }
#else
    printf("Using brute force intersection (BVH disabled)\n");
#endif
    printf("========================\n");

    // Copy host pointers to device globals
    cudaMemcpyToSymbol(d_bvh_nodes, &dev_bvh_nodes, sizeof(BVHNodeGPU*));
    cudaMemcpyToSymbol(d_bvh_tri_indices, &dev_bvh_tri_indices, sizeof(int*));

    if (!scene->environmentHDR.empty()) {
        try {
            EnvImageHost img = loadHDR(scene->environmentHDR.c_str());
            EnvAliasHost alias = buildEnvAlias(img.texels.data(), img.width, img.height);
            uploadEnv(img, alias, dev_env);
            cudaMemcpyToSymbol(d_env, &dev_env, sizeof(EnvironmentMap));
        } catch (const std::exception&){
        }
    }
    if (!scene->textures.empty()) {
        std::vector<cudaTextureObject_t> texObjs;
        texObjs.reserve(scene->textures.size());
        
        for (auto& tex : scene->textures) {
            if (!tex.filepath.empty() && tex.texObj == 0) {
                try {
                    ImageHost img = loadImage(tex.filepath.c_str());
                    uploadTexture(img, tex);  
                } catch (const std::exception& e) {
                    std::cerr << "Failed to load/upload texture: " << tex.filepath << " - " << e.what() << std::endl;
                    tex.texObj = 0;
                }
            }
            
            if (tex.texObj != 0) {
                texObjs.push_back(tex.texObj);
            }
        }
        
        // Upload texture object array to GPU
        if (!texObjs.empty()) {
            cudaMalloc(&dev_textures, texObjs.size() * sizeof(cudaTextureObject_t));
            cudaMemcpy(dev_textures, texObjs.data(), texObjs.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
            std::cout << "Uploaded " << texObjs.size() << " texture to GPU for rendering" << std::endl;
        } else {
            dev_textures = nullptr;
        }
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaDeviceSynchronize();
    
    cudaFree(dev_image); 
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_material_sort_indices);
    cudaFree(dev_vertices);
    cudaFree(dev_triangles);
    cudaFree(dev_bvh_nodes);
    cudaFree(dev_bvh_tri_indices);
    cudaFree(dev_textures);
    
    freeEnv(dev_env);
    
    if (hst_scene) {
        for (auto& tex : hst_scene->textures) {
            freeTexture(tex);
        }
    }

    // checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool antialiasingEnabled)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // Generate random numbers for antialiasing and depth of field
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Stochastic antialiasing by jittering the ray
        float jitter_x = 0.0f;
        float jitter_y = 0.0f;
        
        if (antialiasingEnabled) {
            // Jitter within the pixel
            jitter_x = u01(rng) - 0.5f;
            jitter_y = u01(rng) - 0.5f;
        }
        
        // Compute the point on the image plane
        glm::vec3 imagePoint = cam.position + cam.view * cam.focalDistance
            - cam.right * cam.pixelLength.x * ((float)x + jitter_x - (float)cam.resolution.x * 0.5f) * cam.focalDistance
            - cam.up * cam.pixelLength.y * ((float)y + jitter_y - (float)cam.resolution.y * 0.5f) * cam.focalDistance;

        // Thin lens model for depth of field
        if (cam.lensRadius > 0.0f) {
            // Sample point on lens aperture using concentric disk sampling
            float u1 = u01(rng);
            float u2 = u01(rng);
            
            // Convert to polar coordinates
            float r = cam.lensRadius * sqrtf(u1);
            float theta = 2.0f * PI * u2;
            
            // Convert back to Cartesian coordinates on lens plane
            glm::vec3 lensPoint = cam.position 
                + cam.right * (r * cosf(theta))
                + cam.up * (r * sinf(theta));
            
            // Ray origin is the sampled point on the lens
            segment.ray.origin = lensPoint;
            
            // Ray direction points from lens to the image point
            segment.ray.direction = glm::normalize(imagePoint - lensPoint);
        } else {
            // Pinhole camera model 
            segment.ray.origin = cam.position;
            segment.ray.direction = glm::normalize(cam.view
                - cam.right * cam.pixelLength.x * ((float)x + jitter_x - (float)cam.resolution.x * 0.5f)
                - cam.up * cam.pixelLength.y * ((float)y + jitter_y - (float)cam.resolution.y * 0.5f)
            );
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Vertex* vertices,
    Triangle* triangles,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        int hit_material_id = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        int tmp_material_id = -1;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_material_id = geom.materialid;
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                tmp_material_id = geom.materialid;
            }
            else if (geom.type == MESH)
            {
                // Transform ray to object space
                Ray q;
                q.origin = multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.origin, 1.0f));
                q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(pathSegment.ray.direction, 0.0f)));

                float closestT = FLT_MAX;
                glm::vec3 closestIntersect, closestNormal;
                glm::vec2 closestUV;
                int closestMaterialId = geom.materialid;
                bool hit = false;

#if USE_BVH
                if (d_bvh_nodes != NULL && geom.bvhNodeCount > 0 && d_bvh_tri_indices != NULL) {
                    // printf("Using BVH traversal for mesh %d\n", i);
                    const int root = geom.bvhNodeOffset;
                    int stack[64]; int sp = 0; stack[sp++] = root;

                    auto slabHit = [&q, &closestT](const BVHNodeGPU& n, float& tEnter) {
                        float3 bmin = n.bboxMin;
                        float3 bmax = n.bboxMax;
                        float3 ro = make_float3(q.origin.x, q.origin.y, q.origin.z);
                        float3 rd = make_float3(q.direction.x, q.direction.y, q.direction.z);
                        float3 invD = make_float3(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);
                        
                        // Calculate slab intersections for each axis
                        float tx1 = (bmin.x - ro.x) * invD.x;
                        float tx2 = (bmax.x - ro.x) * invD.x;
                        float ty1 = (bmin.y - ro.y) * invD.y;
                        float ty2 = (bmax.y - ro.y) * invD.y;
                        float tz1 = (bmin.z - ro.z) * invD.z;
                        float tz2 = (bmax.z - ro.z) * invD.z;
                        
                        // Find intersection interval
                        float tmin = fmaxf(fmaxf(fminf(tx1, tx2), fminf(ty1, ty2)), fminf(tz1, tz2));
                        float tmax = fminf(fminf(fmaxf(tx1, tx2), fmaxf(ty1, ty2)), fmaxf(tz1, tz2));
                        
                        tEnter = tmin;
                        return tmax >= fmaxf(tmin, 0.0f) && tmin < closestT;
                    };

                    while (sp > 0) {
                        int ni = stack[--sp];
                        const BVHNodeGPU n = d_bvh_nodes[ni];
                        float tEnter;
                        if (!slabHit(n, tEnter)) continue;
                        if (n.count > 0) {
                            // Leaf node, test triangles
                            int start = n.start;
                            int end = n.start + n.count;
                            for (int j = start; j < end; ++j) {
                                int triIdx = d_bvh_tri_indices[geom.bvhTriIndexOffset + j];
                                Triangle tri = triangles[triIdx];
                                glm::vec3 tempIntersect;
                                glm::vec3 tempNormal;
                                glm::vec2 tempUV;
                                float triT;
                                
                                if (triangleIntersectionTest(tri, vertices, q, tempIntersect, tempNormal, tempUV, triT)) {
                                    if (triT < closestT) {
                                        closestT = triT;
                                        hit = true;
                                        closestIntersect = tempIntersect;
                                        closestNormal = tempNormal;
                                        closestUV = tempUV;
                                        closestMaterialId = tri.materialId;
                                    }
                                }
                            }
                        } else {
                            // Interior node, push children, near first
                            int li = n.left;
                            int ri = n.right;
                            float tL = 0.0f;
                            float tR = 0.0f;
                            bool hL = false;
                            bool hR = false;
                            
                            if (li >= 0) {
                                BVHNodeGPU ln = d_bvh_nodes[li];
                                hL = slabHit(ln, tL);
                            }
                            if (ri >= 0) {
                                BVHNodeGPU rn = d_bvh_nodes[ri];
                                hR = slabHit(rn, tR);
                            }
                            if (hL && hR) {
                                if (tL < tR) {
                                    stack[sp++] = ri;
                                    stack[sp++] = li;
                                } else {
                                    stack[sp++] = li;
                                    stack[sp++] = ri;
                                }
                            } else if (hL) {
                                stack[sp++] = li;
                            } else if (hR) {
                                stack[sp++] = ri;
                            }
                        }
                    }
                } else {
                    // brute force the mesh triangles
                    // printf("BVH fallback: Using brute force for mesh %d\n", i);
                    for (int iTri = 0; iTri < geom.numTriangles; iTri++) {
                        Triangle tri = triangles[geom.triangleOffset + iTri];
                        glm::vec3 tempIntersect;
                        glm::vec3 tempNormal;
                        glm::vec2 tempUV;
                        float triT;
                        
                        if (triangleIntersectionTest(tri, vertices, q, tempIntersect, tempNormal, tempUV, triT)) {
                            if (triT < closestT) {
                                closestT = triT;
                                hit = true;
                                closestIntersect = tempIntersect;
                                closestNormal = tempNormal;
                                closestUV = tempUV;
                                closestMaterialId = tri.materialId;
                            }
                        }
                    }
                }
#else
                // brute force mesh intersection
                // printf("Using brute force intersection for mesh %d (BVH disabled)\n", i);
                for (int i = 0; i < geom.numTriangles; i++) {
                    Triangle tri = triangles[geom.triangleOffset + i];
                    glm::vec3 tempIntersect;
                    glm::vec3 tempNormal;
                    glm::vec2 tempUV;
                    float triT;
                    
                    if (triangleIntersectionTest(tri, vertices, q, tempIntersect, tempNormal, tempUV, triT)) {
                        if (triT < closestT) {
                            closestT = triT;
                            hit = true;
                            closestIntersect = tempIntersect;
                            closestNormal = tempNormal;
                            closestUV = tempUV;
                            closestMaterialId = tri.materialId;
                        }
                    }
                }
#endif

                if (hit) {
                    tmp_intersect = multiplyMV(geom.transform, glm::vec4(closestIntersect, 1.0f));
                    tmp_normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(closestNormal, 0.0f)));
                    tmp_uv = closestUV;
                    tmp_material_id = closestMaterialId;
                    t = glm::length(pathSegment.ray.origin - tmp_intersect);
                    outside = glm::dot(tmp_normal, pathSegment.ray.direction) < 0.0f;
                    
                    if (!outside) {
                        tmp_normal = -tmp_normal;
                    }
                } else {
                    t = -1.0f;
                }
            }
            // TODO: add more intersection tests here... metaball? CSG?

            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                hit_material_id = tmp_material_id;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = hit_material_id;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
// Shader with proper BSDF evaluation for diffuse materials
__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* textures,
    int currentDepth,
    bool russianRouletteEnabled,
    Geom* geoms,
    int geoms_size,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& pathSegment = pathSegments[idx];

        if (pathSegment.remainingBounces == 0) {
            return;
        }
        
        if (intersection.t > 0.0f) // if the intersection exists
        {
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);

            
            Material material = materials[intersection.materialId];
            
            glm::vec3 baseColor = material.color;
            if (material.hasTexture && material.textureID >= 0 && textures != nullptr) {
                cudaTextureObject_t texObj = textures[material.textureID];
                if (texObj != 0) {
                    float4 texSample = tex2D<float4>(texObj, intersection.uv.x, intersection.uv.y);
                    
                    if (texSample.w < 0.5f) {
                        if (d_env.texture != 0) {
                            float u, v; dirToUV(pathSegment.ray.direction, u, v);
                            glm::vec3 Le = envTex(d_env.texture, u, v) * 2.0f;
                            glm::vec3 contrib = clampLuminance(pathSegment.color * Le, 20.0f);
                            atomicAdd(&image[pathSegment.pixelIndex].x, contrib.x);
                            atomicAdd(&image[pathSegment.pixelIndex].y, contrib.y);
                            atomicAdd(&image[pathSegment.pixelIndex].z, contrib.z);
                        }
                        pathSegment.remainingBounces = 0;
                        return;
                    }
                    
                    glm::vec3 texColor(texSample.x, texSample.y, texSample.z);
                    
                    // Apply sRGB to linear conversion
                    auto srgbToLinear = [](float c) {
                        return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
                    };
                    texColor = glm::vec3(srgbToLinear(texColor.x), srgbToLinear(texColor.y), srgbToLinear(texColor.z));
                    
                    baseColor *= texColor;
                }
            }
            
            if (material.emittance > 0.0f) {
                glm::vec3 contrib = pathSegment.color * (baseColor * material.emittance);
                contrib = clampLuminance(contrib, 20.0f);
                atomicAdd(&image[pathSegment.pixelIndex].x, contrib.x);
                atomicAdd(&image[pathSegment.pixelIndex].y, contrib.y);
                atomicAdd(&image[pathSegment.pixelIndex].z, contrib.z);
                pathSegment.remainingBounces = 0; // terminate path at light source
            }
            else {
                glm::vec3 intersectPoint = getPointOnRay(pathSegment.ray, intersection.t);

                Material shadingMaterial = material;
                shadingMaterial.color = baseColor;

                // Scatter the ray using BSDF evaluation
                scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, shadingMaterial, rng);
                
                // Russian Roulette 
                if (russianRouletteEnabled && currentDepth >= RR_MIN_DEPTH) {
                    float maxComponent = fmaxf(fmaxf(pathSegment.color.r, pathSegment.color.g), pathSegment.color.b);
                    float q = fmaxf(0.15f, 1.0f - maxComponent);
                    
                    // Generate random sample
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    float randomSample = u01(rng);
                    
                    if (randomSample < q) {
                        // Terminate the path 
                        pathSegment.remainingBounces = 0;
                        return;
                    } else {
                        // Scale by 1/(1-q) 
                        pathSegment.color /= (1.0f - q);
                    }
                }
                
                // Decrement remaining bounces
                pathSegment.remainingBounces--;
                
                // Terminate path if no more bounces
                if (pathSegment.remainingBounces <= 0) {
                    pathSegment.color = glm::vec3(0.0f);
                    return;
                }
            }
        }
        else {
            // No intersection -> environment miss shading
            if (d_env.texture != 0) {
                float u, v;
                dirToUV(pathSegment.ray.direction, u, v);
                glm::vec3 Le = envTex(d_env.texture, u, v);
                
                // Scale up environment lighting for brighter scene
                float envIntensity = 2.0f;
                Le *= envIntensity;
                
                glm::vec3 contrib = pathSegment.color * Le;
                contrib = clampLuminance(contrib, 20.0f);
                atomicAdd(&image[pathSegment.pixelIndex].x, contrib.x);
                atomicAdd(&image[pathSegment.pixelIndex].y, contrib.y);
                atomicAdd(&image[pathSegment.pixelIndex].z, contrib.z);
            }
            pathSegment.remainingBounces = 0;
        }
    }
}

__global__ void prepareMaterialSortIndices(
    int num_paths,
    ShadeableIntersection* intersections,
    int* sort_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        if (intersections[idx].t > 0.0f) {
            sort_indices[idx] = intersections[idx].materialId;
        } else {
            sort_indices[idx] = -1;
        }
    }
}

__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Performance timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths, 
        guiData != NULL ? guiData->AntialiasingEnabled : true);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    int initial_paths = num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_vertices,
            dev_triangles,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        
        if (guiData != NULL && guiData->MaterialSortingEnabled) {
            prepareMaterialSortIndices<<<numblocksPathSegmentTracing, blockSize1d>>>(
                num_paths,
                dev_intersections,
                dev_material_sort_indices
            );
            checkCUDAError("prepare material sort indices");
            
            thrust::sequence(thrust::device, dev_material_sort_indices, dev_material_sort_indices + num_paths);
            
            // Sort paths and intersections by material ID
            thrust::sort_by_key(thrust::device, 
                dev_material_sort_indices, dev_material_sort_indices + num_paths,
                dev_paths);
            thrust::sort_by_key(thrust::device,
                dev_material_sort_indices, dev_material_sort_indices + num_paths,
                dev_intersections);
            
            checkCUDAError("material sorting");
        }
        
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures,
            depth,
            guiData != NULL ? guiData->RussianRouletteEnabled : false,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_image
        );
        checkCUDAError("shade material");
        cudaDeviceSynchronize();

        // Stream compaction to remove terminated paths
        PathSegment* new_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, IsPathAlive());
        num_paths = new_end - dev_paths;

        // Continue until max depth is reached or no more active paths
        iterationComplete = (depth >= traceDepth) || (num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
            guiData->ActivePaths = num_paths;
            guiData->TotalPaths = initial_paths;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Stop timing and calculate performance metrics
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    if (guiData != NULL)
    {
        guiData->RenderTimeMs = milliseconds;
    }

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    // Clean up timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCUDAError("pathtrace");
}

