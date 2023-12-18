#include "Objects/Sphere.hpp"
#include <cmath>
#include "Ray.hpp"
#include "AABB.hpp"
#include "Util.hpp"
#include "ONB.hpp"
#include "RandomGenerator.hpp"
using namespace std;


// Get the UV coordinates of a point on a sphere
inline void sphere_get_uv(const Vec3 &p, rreal &u, rreal &v) NOEXCEPT {
    const rreal phi = util::atan2(p.z, p.x);
    const rreal theta = util::asin(p.y);

    u = 1 - ((phi + Pi) / (2 * Pi));
    v = (theta + (Pi / 2)) / Pi;
}


Sphere::Sphere(const Vec3 &cen, const rreal r, const shared_ptr<IMaterial> &mat) NOEXCEPT :
    _center(cen),
    _radius(r),
    _mat_ptr(mat)
{ }

shared_ptr<IHittable> Sphere::deep_copy() const NOEXCEPT {
    auto s = make_shared<Sphere>(*this);
    s->_mat_ptr = _mat_ptr->deep_copy();

    return s;
}

bool Sphere::hit(
    [[maybe_unused]] RandomGenerator &rng,
    const Ray &r,
    const rreal t_min,
    const rreal t_max,
    HitRecord &rec
) const NOEXCEPT {
    const Vec3 oc = r.origin - _center;
    const Vec3 ray_dir = r.direction;
    const rreal a = ray_dir.length_squared();
    const rreal half_b = oc.dot(ray_dir);
    const rreal c = oc.length_squared() - (_radius * _radius);
    const rreal discriminant = (half_b * half_b) - (a * c);

#ifdef USE_BOOK_SPHERE_HIT
    if (discriminant > 0) {
        const rreal root = util::sqrt(discriminant);

        rreal temp = (-half_b - root) / a;
        if ((temp < t_max) && (temp > t_min)) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            const Vec3 outward_normal = (rec.p - _center) / _radius;
            rec.set_face_normal(r, outward_normal);
            rec.set_mat_ptr(_mat_ptr);
            sphere_get_uv((rec.p - _center) / _radius, rec.u, rec.v);
            return true;
        }

        temp = (-half_b + root) / a;
        if ((temp < t_max) && (temp > t_min)) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            const Vec3 outward_normal = (rec.p - _center) / _radius;
            rec.set_face_normal(r, outward_normal);
            rec.set_mat_ptr(_mat_ptr);
            sphere_get_uv((rec.p - _center) / _radius, rec.u, rec.v);
            return true;
        }
    }
#else
    // TODO [performance], this needs some more measuring, not sure if there was a speed
    //      bumpt when I put the calculation of p* and on* back into their constituent `if`s
    //
    //      Also, what if I actually setup both `recs` before the `if`?  Does order of the `if` matter?

    // This one has a slight advantage over the book code since it's able to be
    // vectorized by the compiler,
    if (discriminant > 0) {
        const rreal root = util::sqrt(discriminant);
        const rreal temp1 = (-half_b - root) / a;
        const rreal temp2 = (-half_b + root) / a;
        const bool hit1 = (temp1 < t_max) && (temp1 > t_min);
        const bool hit2 = (temp2 < t_max) && (temp2 > t_min);

        if (hit1) {
            const Vec3 p1 = r.at(temp1);
            const Vec3 on1 = (p1 - _center) / _radius;      // on = outward normal
            sphere_get_uv(on1, rec.u, rec.v);

            rec.t = temp1;
            rec.p = p1;
            rec.set_face_normal(r, on1);
            rec.set_mat_ptr(_mat_ptr);
            return true;
        }

        if (hit2) {
            const Vec3 p2 = r.at(temp2);
            const Vec3 on2 = (p2 - _center) / _radius;
            sphere_get_uv(on2, rec.u, rec.v);

            rec.t = temp2;
            rec.p = p2;
            rec.set_face_normal(r, on2);
            rec.set_mat_ptr(_mat_ptr);
            return true;
        }
    }
#endif

    return false;
}

bool Sphere::bounding_box(
    [[maybe_unused]] const rreal t0,
    [[maybe_unused]] const rreal t1,
    AABB &output_box
) const NOEXCEPT {
    const Vec3 r(abs(_radius));
    output_box = AABB(_center - r, _center + r);
    return true;
}

rreal Sphere::pdf_value(RandomGenerator &rng, const Vec3 &origin, const Vec3 &v) const NOEXCEPT {
    HitRecord rec;
    const bool did_hit = hit(rng, Ray(origin, v), 0.001, Infinity, rec);

    if (!did_hit)
        return 0;

    const rreal dist_sq = (_center - origin).length_squared();
    const rreal cos_theta_max = util::sqrt(1.0 - (_radius * _radius / dist_sq));
    const rreal solid_angle = 2 * Pi * (1 - cos_theta_max);

    return 1.0 / solid_angle;
}

Vec3 Sphere::random(RandomGenerator &rng, const Vec3 &origin) const NOEXCEPT {
    ONB uvw;
    const Vec3 direction = _center - origin;
    const rreal dist_sq = direction.length_squared();

    uvw.build_from_w(direction);

    return uvw.local(rng.get_to_sphere(_radius, dist_sq));
}
