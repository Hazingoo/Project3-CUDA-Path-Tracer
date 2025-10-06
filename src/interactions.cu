#include "interactions.h"
#include "env.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float schlickApproximation(
    float cosTheta,
    float ior)
{
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosTheta, 5.0f);
}

__host__ __device__ glm::vec3 calculateReflection(
    glm::vec3 incident,
    glm::vec3 normal)
{
    return incident - 2.0f * glm::dot(incident, normal) * normal;
}

__host__ __device__ bool calculateRefraction(
    glm::vec3 incident,
    glm::vec3 normal,
    float ior,
    glm::vec3& refracted)
{
    // Snell's law
    refracted = glm::refract(incident, normal, ior);
    
    // Check if refraction is possible 
    return glm::length(refracted) > 0.001f;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 scatterDirection;
    
    // Determine if we're entering or exiting the material
    bool entering = glm::dot(pathSegment.ray.direction, normal) < 0.0f;
    glm::vec3 outwardNormal = entering ? normal : -normal;
    float ior = entering ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);
    
    // Calculate cosine of incident angle
    float cosTheta = glm::dot(-pathSegment.ray.direction, outwardNormal);
    cosTheta = glm::clamp(cosTheta, -1.0f, 1.0f);
    
    if (m.hasRefractive > 0.0f) {
        // use Fresnel to decide reflection vs refraction
        float fresnel = schlickApproximation(cosTheta, ior);
        
        // Try to refract
        glm::vec3 refracted;
        bool canRefract = calculateRefraction(pathSegment.ray.direction, outwardNormal, ior, refracted);
        
        if (canRefract && u01(rng) > fresnel) {
            // Refraction
            scatterDirection = glm::normalize(refracted);
        } else {
            // Total internal reflection or Fresnel reflection
            scatterDirection = calculateReflection(pathSegment.ray.direction, outwardNormal);
        }
        
        // Ensure the direction is normalized and valid
        scatterDirection = glm::normalize(scatterDirection);
        
        // if direction is invalid, fall back to reflection
        if (glm::length(scatterDirection) < 0.001f) {
            scatterDirection = calculateReflection(pathSegment.ray.direction, normal);
            scatterDirection = glm::normalize(scatterDirection);
        }
    }
    else if (m.hasReflective > 0.0f) {
        // Perfect specular reflection
        scatterDirection = calculateReflection(pathSegment.ray.direction, normal);
    }
    else {
        // Diffuse scattering
        scatterDirection = calculateRandomDirectionInHemisphere(normal, rng);
    }
    
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = scatterDirection;
    
    pathSegment.color *= m.color;
}
