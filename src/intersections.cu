#include "intersections.h"
__device__ bool visibleToInfinity(const glm::vec3& origin, const glm::vec3& dir,
                                  const Geom* geoms, int geoms_size)
{
    Ray r; r.origin = origin + dir * 0.001f; r.direction = glm::normalize(dir);
    glm::vec3 tmp_intersect, tmp_normal;
    bool outside;
    for (int i = 0; i < geoms_size; ++i){
        const Geom& g = geoms[i];
        float t = -1.0f;
        if (g.type == CUBE) {
            t = boxIntersectionTest((Geom&)g, r, tmp_intersect, tmp_normal, outside);
        } else if (g.type == SPHERE) {
            t = sphereIntersectionTest((Geom&)g, r, tmp_intersect, tmp_normal, outside);
        } else if (g.type == MESH) {
            glm::vec3 invD = 1.0f / r.direction;
            glm::vec3 t0 = (g.bboxMin - r.origin) * invD;
            glm::vec3 t1 = (g.bboxMax - r.origin) * invD;
            glm::vec3 tmin = glm::min(t0, t1);
            glm::vec3 tmax = glm::max(t0, t1);
            float tEnter = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
            float tExit  = fminf(fminf(tmax.x, tmax.y), tmax.z);
            if (tExit >= fmaxf(tEnter, 0.0f)) t = tEnter > 0.0f ? tEnter : tExit;
        }
        if (t > 0.0f) return false;
    }
    return true;
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool triangleIntersectionTest(
    Triangle triangle,
    const Vertex* vertices,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    glm::vec2 &uv,
    float &t)
{
    // Get triangle vertices using indices
    glm::vec3 v0 = vertices[triangle.idx_v0].pos;
    glm::vec3 v1 = vertices[triangle.idx_v1].pos;
    glm::vec3 v2 = vertices[triangle.idx_v2].pos;
    
    // Moller-Trumbore algorithm
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(r.direction, edge2);
    float a = glm::dot(edge1, h);
    
    if (a > -0.00001f && a < 0.00001f) {
        return false; // Ray is parallel to triangle
    }
    
    float f = 1.0f / a;
    glm::vec3 s = r.origin - v0;
    float u = f * glm::dot(s, h);
    
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    
    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(r.direction, q);
    
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    
    t = f * glm::dot(edge2, q);
    
    if (t > 0.00001f) { // Ray intersection
        intersectionPoint = r.origin + t * r.direction;
        
        // Interpolate normal using barycentric coordinates
        float w = 1.0f - u - v;
        normal = glm::normalize(w * vertices[triangle.idx_v0].nor + 
                               u * vertices[triangle.idx_v1].nor + 
                               v * vertices[triangle.idx_v2].nor);
        
        // Interpolate UV coordinates using barycentric coordinates
        uv = w * vertices[triangle.idx_v0].uv + 
             u * vertices[triangle.idx_v1].uv + 
             v * vertices[triangle.idx_v2].uv;
        
        return true;
    }
    
    return false;
}
