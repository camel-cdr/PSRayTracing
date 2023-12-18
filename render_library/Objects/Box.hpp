#pragma once

#include "Interfaces/IHittable.hpp"
#include "Objects/HittableList.hpp"
#include "AABB.hpp"
class BVHNode;
class RandomGenerator;


class Box : public IHittable {
private:
    // Data
    Vec3 _box_min = Vec3(0);
    Vec3 _box_max = Vec3(0);

#ifdef USE_BOOK_BOX
    // The book's implementation uses a `HittableList` to store objects
    HittableList _sides{};
#else
    // In out implementation, we need to store an AABB and the material pointer ourselves
    AABB _aabb;
    std::shared_ptr<IMaterial> _mat;
#endif // USE_BOOK_BOX

public:
    explicit Box(const Vec3 &point_min, const Vec3 &point_max, const std::shared_ptr<IMaterial> &mat) NOEXCEPT;

    std::shared_ptr<IHittable> deep_copy() const NOEXCEPT override;

    bool hit(RandomGenerator &rng, const Ray &r, const rreal t_min, const rreal t_max, HitRecord &rec) const NOEXCEPT override;
    bool bounding_box(const rreal t0, const rreal t1, AABB &output_box) const NOEXCEPT override;

    rreal pdf_value(
        [[maybe_unused]] RandomGenerator &rng,
        [[maybe_unused]] const Vec3 &origin,
        [[maybe_unused]] const Vec3 &v
    ) const NOEXCEPT override {
        return 0;
    }

    Vec3 random(
        [[maybe_unused]] RandomGenerator &rng,
        [[maybe_unused]] const Vec3 &origin
    ) const NOEXCEPT override {
        return Vec3(1, 0, 0);
    }
};
