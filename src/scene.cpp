#include "scene.h"
#include "bvh.hpp"
#include "texture.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    std::map<std::string, int> texturePathToID;  
    
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        newMaterial.color = glm::vec3(0.0f);
        newMaterial.specular.exponent = 0.0f;
        newMaterial.specular.color = glm::vec3(0.0f);
        newMaterial.hasReflective = 0.0f;
        newMaterial.hasRefractive = 0.0f;
        newMaterial.indexOfRefraction = 1.0f;
        newMaterial.emittance = 0.0f;
        newMaterial.textureID = -1;
        newMaterial.hasTexture = 0;
        
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.0f;
        }
        else if (p["TYPE"] == "Refractive" || p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p.contains("IOR") ? p["IOR"] : 1.5f; 
            std::cout << "Loaded Glass material: IOR=" << newMaterial.indexOfRefraction 
                      << ", Color=(" << newMaterial.color.x << ", " << newMaterial.color.y << ", " << newMaterial.color.z << ")" << std::endl;
        }
        
        if (p.contains("TEXTURE")) {
            std::string texturePath = p["TEXTURE"];
            
            if (texturePathToID.find(texturePath) != texturePathToID.end()) {
                newMaterial.textureID = texturePathToID[texturePath];
                newMaterial.hasTexture = 1;
            } else {
                try {
                    TextureData texData;
                    texData.filepath = texturePath;
                    texData.texObj = 0; 
                    texData.array = nullptr;
                    texData.width = 0;
                    texData.height = 0;
                    
                    int texID = textures.size();
                    textures.push_back(texData);
                    texturePathToID[texturePath] = texID;
                    
                    newMaterial.textureID = texID;
                    newMaterial.hasTexture = 1;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Failed to register texture '" << texturePath << "' for material '" << name << "': " << e.what() << std::endl;
                    newMaterial.textureID = -1;
                    newMaterial.hasTexture = 0;
                }
            }
        }
        
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
        glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        glm::vec3 scaleVec = glm::vec3(scale[0], scale[1], scale[2]);
        
        if (type == "obj" || type == "mesh")
        {
            // Load OBJ file
            if (p.contains("FILE")) {
                std::string objPath = p["FILE"];
                std::string materialName = p["MATERIAL"];
                if (MatNameToID.find(materialName) == MatNameToID.end()) {
                    std::cerr << "Error: Material '" << materialName << "' not found for OBJ object!" << std::endl;
                    exit(-1);
                }
                int materialId = MatNameToID[materialName];
                loadOBJFile(objPath, materialId, translation, rotation, scaleVec);
                // Build BVH for the newly loaded mesh range
                Geom& mesh = geoms.back();
                if (mesh.type == MESH && mesh.numTriangles > 0) {
                    BuiltBVH b = buildMeshBVH(triangles, vertices, mesh.triangleOffset, mesh.numTriangles);
                    
                    // Calculate BVH statistics for this mesh
                    int leafNodes = 0;
                    for (const auto& node : b.nodes) {
                        if (node.count > 0) { // leaf node
                            leafNodes++;
                        }
                    }
                    
                    std::cout << "=== Mesh BVH Statistics ===" << std::endl;
                    std::cout << "Mesh: " << objPath << std::endl;
                    std::cout << "Triangles: " << mesh.numTriangles << std::endl;
                    std::cout << "BVH nodes: " << b.nodes.size() << std::endl;
                    std::cout << "Leaf nodes: " << leafNodes << std::endl;
                    std::cout << "Internal nodes: " << (b.nodes.size() - leafNodes) << std::endl;
                    std::cout << "=========================" << std::endl;
                    
                    mesh.bvhNodeOffset = (int)bvhNodes.size();
                    mesh.bvhNodeCount = (int)b.nodes.size();
                    mesh.bvhTriIndexOffset = (int)bvhTriIndices.size();
                    mesh.bvhTriIndexCount = (int)b.triIndices.size();
                    bvhNodes.insert(bvhNodes.end(), b.nodes.begin(), b.nodes.end());
                    bvhTriIndices.insert(bvhTriIndices.end(), b.triIndices.begin(), b.triIndices.end());
                }
            } else {
                std::cerr << "OBJ object missing FILE parameter" << std::endl;
                exit(-1);
            }
        }
        else
        {
            // Handle primitive shapes (sphere, cube)
            Geom newGeom;
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else
            {
                newGeom.type = SPHERE;
            }
            std::string materialName = p["MATERIAL"];
            if (MatNameToID.find(materialName) == MatNameToID.end()) {
                std::cerr << "Error: Material '" << materialName << "' not found!" << std::endl;
                exit(-1);
            }
            newGeom.materialid = MatNameToID[materialName];
            newGeom.translation = translation;
            newGeom.rotation = rotation;
            newGeom.scale = scaleVec;
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            geoms.push_back(newGeom);
        }
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    if (data.contains("EnvironmentHDR")) {
        environmentHDR = data["EnvironmentHDR"].get<std::string>();
    }
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // Initialize depth of field parameters
    // Default values, lensRadius = 0 (no depth of field), focalDistance = distance to lookAt point
    camera.lensRadius = cameraData.contains("LENS_RADIUS") ? cameraData["LENS_RADIUS"] : 0.0f;
    camera.focalDistance = cameraData.contains("FOCAL_DISTANCE") ? cameraData["FOCAL_DISTANCE"] : glm::length(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadOBJFile(const std::string& objPath, int materialId, const glm::vec3& translation, 
                        const glm::vec3& rotation, const glm::vec3& scale)
{
    std::cout << "Attempting to load OBJ file: " << objPath << std::endl;
    
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./";

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(objPath, reader_config)) {
        std::cerr << "TinyObjReader failed to parse file: " << objPath << std::endl;
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader Error: " << reader.Error() << std::endl;
        }
        std::cerr << "Failed to load OBJ file: " << objPath << std::endl;
        
        // Check if file exists
        std::ifstream fileCheck(objPath);
        if (!fileCheck.good()) {
            std::cerr << "File does not exist or cannot be read: " << objPath << std::endl;
            
            std::vector<std::string> altPaths = {
                "./" + objPath,
                "../" + objPath,
                "models/" + objPath,
                "../models/" + objPath
            };
            
            for (const auto& altPath : altPaths) {
                std::ifstream altCheck(altPath);
                if (altCheck.good()) {
                    std::cout << "Found file at alternative path: " << altPath << std::endl;
                    break;
                }
            }
        } else {
            std::cerr << "File exists but TinyObj couldn't parse it" << std::endl;
        }
        return; 
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader Warning: " << reader.Warning() << std::endl;
    }
    
    std::cout << "Successfully loaded OBJ file: " << objPath << std::endl;

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    // Create transformation matrix
    glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
    glm::mat4 inverseTransform = glm::inverse(transform);
    glm::mat4 invTranspose = glm::inverseTranspose(transform);

    // Store offsets
    int vertexOffset = vertices.size();
    int triangleOffset = triangles.size();
    

    std::vector<Vertex> meshVertices;
    
    // First, create a map to track unique vertex combinations (
    std::map<std::tuple<int, int, int>, int> vertexMap;
    std::vector<int> faceVertexIndices;
    
    for (const auto& shape : shapes) {
        for (size_t i = 0; i < shape.mesh.indices.size(); i++) {
            const auto& idx = shape.mesh.indices[i];
            
            int posIdx = idx.vertex_index;
            int normalIdx = idx.normal_index;
            int uvIdx = idx.texcoord_index;
            
            auto key = std::make_tuple(posIdx, normalIdx, uvIdx);
            
            // Check if we've seen this exact vertex combination before
            if (vertexMap.find(key) == vertexMap.end()) {
                // New unique vertex
                glm::vec3 pos(0.0f);
                if (posIdx >= 0 && posIdx < attrib.vertices.size() / 3) {
                    pos = glm::vec3(
                        attrib.vertices[3 * posIdx + 0],
                        attrib.vertices[3 * posIdx + 1],
                        attrib.vertices[3 * posIdx + 2]
                    );
                }
                
                glm::vec3 normal(0.0f, 1.0f, 0.0f);
                if (normalIdx >= 0 && normalIdx < attrib.normals.size() / 3) {
                    normal = glm::vec3(
                        attrib.normals[3 * normalIdx + 0],
                        attrib.normals[3 * normalIdx + 1],
                        attrib.normals[3 * normalIdx + 2]
                    );
                }
                
                glm::vec2 uv(0.0f, 0.0f);
                if (uvIdx >= 0 && uvIdx < attrib.texcoords.size() / 2) {
                    uv = glm::vec2(
                        attrib.texcoords[2 * uvIdx + 0],
                        attrib.texcoords[2 * uvIdx + 1]
                    );
                }
                
                int newIdx = meshVertices.size();
                meshVertices.push_back(Vertex(pos, normal, uv));
                vertexMap[key] = newIdx;
                faceVertexIndices.push_back(newIdx);
            } else {
                // Reuse existing vertex
                faceVertexIndices.push_back(vertexMap[key]);
            }
        }
    }
    
    std::cout << "  Loaded " << meshVertices.size() << " unique vertices with UV coordinates" << std::endl;
    
    // Add vertices to global array
    vertices.insert(vertices.end(), meshVertices.begin(), meshVertices.end());
    
    // Calculate bounding box in object space, then transform
    glm::vec3 bboxMin(FLT_MAX);
    glm::vec3 bboxMax(-FLT_MAX);
    for (const auto& vertex : meshVertices) {
        bboxMin = glm::min(bboxMin, vertex.pos);
        bboxMax = glm::max(bboxMax, vertex.pos);
    }
    
    // Transform bounding box to world space
    glm::vec4 bboxMinTransformed = transform * glm::vec4(bboxMin, 1.0f);
    glm::vec4 bboxMaxTransformed = transform * glm::vec4(bboxMax, 1.0f);
    bboxMin = glm::vec3(bboxMinTransformed);
    bboxMax = glm::vec3(bboxMaxTransformed);

    // Process triangles
    int numTriangles = 0;
    for (size_t i = 0; i < faceVertexIndices.size(); i += 3) {
        Triangle tri;
        tri.idx_v0 = vertexOffset + faceVertexIndices[i + 0];
        tri.idx_v1 = vertexOffset + faceVertexIndices[i + 1];
        tri.idx_v2 = vertexOffset + faceVertexIndices[i + 2];
        tri.materialId = materialId;
        triangles.push_back(tri);
        numTriangles++;
    }
    

    // Create mesh geometry
    Geom meshGeom;
    meshGeom.type = MESH;
    meshGeom.materialid = materialId;
    meshGeom.translation = translation;
    meshGeom.rotation = rotation;
    meshGeom.scale = scale;
    meshGeom.transform = transform;
    meshGeom.inverseTransform = inverseTransform;
    meshGeom.invTranspose = invTranspose;
    meshGeom.numTriangles = numTriangles;
    meshGeom.triangleOffset = triangleOffset;
    meshGeom.numVertices = meshVertices.size();
    meshGeom.vertexOffset = vertexOffset;
    meshGeom.bboxMin = bboxMin;
    meshGeom.bboxMax = bboxMax;

    geoms.push_back(meshGeom);
}
