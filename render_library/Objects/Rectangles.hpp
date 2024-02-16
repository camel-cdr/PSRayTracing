#pragma once

#include "Interfaces/IHittable.hpp"
#include "AABB.hpp"
class IMaterial;


//*== Three types of axis-aligned rectangles

class XYRect FINAL : public IHittable {
private:
    // Data
    rreal _x0, _x1;
    rreal _y0, _y1;
    rreal _k;
    std::shared_ptr<IMaterial> _mat_ptr;

public:
    explicit XYRect(
        const rreal x0,
        const rreal x1,
        const rreal y0,
        const rreal y1,
        const rreal k,
        std::shared_ptr<IMaterial> mat
    ) NOEXCEPT;

    std::shared_ptr<IHittable> deep_copy() const NOEXCEPT override;

    bool hit(RandomGenerator &rng, const Ray &r, const rreal t_min, const rreal t_max, HitRecord &rec) const NOEXCEPT override;
    bool bounding_box(const rreal t0, const rreal t1, AABB &output_box) const NOEXCEPT override;
    rreal pdf_value(RandomGenerator &rng, const Vec3 &origin, const Vec3 &v) const NOEXCEPT override;
    Vec3 random(RandomGenerator &rng, const Vec3 &origin) const NOEXCEPT override;
};


class XZRect FINAL : public IHittable {
private:
    // Data
    rreal _x0, _x1;
    rreal _z0, _z1;
    rreal _k;
    std::shared_ptr<IMaterial> _mat_ptr;

public:
    explicit XZRect(
        const rreal x0,
        const rreal x1,
        const rreal z0,
        const rreal z1,
        const rreal k,
        std::shared_ptr<IMaterial> mat
    ) NOEXCEPT;

    std::shared_ptr<IHittable> deep_copy() const NOEXCEPT override;

    bool hit(RandomGenerator &rng, const Ray &r, const rreal t_min, const rreal t_max, HitRecord &rec) const NOEXCEPT override;
    bool bounding_box(const rreal t0, const rreal t1, AABB &output_box) const NOEXCEPT override;
    rreal pdf_value(RandomGenerator &rng, const Vec3 &origin, const Vec3 &v) const NOEXCEPT override;
    Vec3 random(RandomGenerator &rng, const Vec3 &origin) const NOEXCEPT override;
};


class YZRect FINAL : public IHittable {
private:
    // Data
    rreal _y0, _y1;
    rreal _z0, _z1;
    rreal _k;
    std::shared_ptr<IMaterial> _mat_ptr;

public:
    explicit YZRect(
        const rreal y0,
        const rreal y1,
        const rreal z0,
        const rreal z1,
        const rreal k,
        std::shared_ptr<IMaterial> mat
    ) NOEXCEPT;

    std::shared_ptr<IHittable> deep_copy() const NOEXCEPT override;

    bool hit(RandomGenerator &rng, const Ray &r, const rreal t_min, const rreal t_max, HitRecord &rec) const NOEXCEPT override;
    bool bounding_box(const rreal t0, const rreal t1, AABB &output_box) const NOEXCEPT override;
    rreal pdf_value(RandomGenerator &rng, const Vec3 &origin, const Vec3 &v) const NOEXCEPT override;
    Vec3 random(RandomGenerator &rng, const Vec3 &origin) const NOEXCEPT override;
};
