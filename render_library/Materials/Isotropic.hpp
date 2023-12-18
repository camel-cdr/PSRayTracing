#pragma once

#include "Interfaces/IMaterial.hpp"
#include <memory>
class ITexture;


class Isotropic : public IMaterial {
private:
    // Data
    std::shared_ptr<ITexture> _albedo;

public:
    explicit Isotropic(const Vec3 &clr) NOEXCEPT;
    explicit Isotropic(const std::shared_ptr<ITexture> &albedo) NOEXCEPT;

    std::shared_ptr<IMaterial> deep_copy() const NOEXCEPT override;

    bool scatter(
        RandomGenerator &rng,
        const Ray &r_in,
        const HitRecord &h_rec,
        ScatterRecord &s_rec,
        const RenderMethod method
    ) const NOEXCEPT override;

    virtual rreal scattering_pdf(
        const Ray &r_in,
        const HitRecord &h_rec,
        const Ray &scattered
    ) const NOEXCEPT override;

    virtual Vec3 emitted(
        const Ray &r_in,
        const HitRecord &h_rec,
        const rreal u,
        const rreal v,
        const Vec3 &p,
        const RenderMethod method
    ) const NOEXCEPT override;
};
