#pragma once

#include "sceneStructs.h"
#include <vector>
#include <string>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadOBJFile(const std::string& objPath, int materialId, const glm::vec3& translation, 
                     const glm::vec3& rotation, const glm::vec3& scale);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Vertex> vertices;    
    std::vector<Triangle> triangles;
    RenderState state;
    std::string environmentHDR;
};
