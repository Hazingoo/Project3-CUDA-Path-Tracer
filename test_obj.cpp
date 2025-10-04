#include <iostream>
#include <fstream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "external/include/tiny_obj_loader.h"

int main() {
    std::string objPath = "models/cube.obj";
    
    std::cout << "Testing OBJ file: " << objPath << std::endl;
    
    // Check if file exists
    std::ifstream fileCheck(objPath);
    if (!fileCheck.good()) {
        std::cout << "File does not exist or cannot be read!" << std::endl;
        return 1;
    }
    std::cout << "File exists and can be read." << std::endl;
    
    // Try to parse with TinyObjLoader
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;
    
    bool ret = reader.ParseFromFile(objPath, reader_config);
    if (!ret) {
        std::cout << "TinyObjLoader failed to parse the file!" << std::endl;
        if (!reader.Error().empty()) {
            std::cout << "Error: " << reader.Error() << std::endl;
        }
        return 1;
    }
    
    std::cout << "Successfully parsed OBJ file!" << std::endl;
    
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    
    std::cout << "Vertices: " << attrib.vertices.size() / 3 << std::endl;
    std::cout << "Shapes: " << shapes.size() << std::endl;
    
    return 0;
}
