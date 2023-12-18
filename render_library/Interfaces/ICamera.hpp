#pragma once

#include "Common.hpp"
#include "Interfaces/IDeepCopyable.hpp"

class Ray;
class RandomGenerator;

// An interface so we can have multiple types of cameras
class ICamera : public IDeepCopyable<ICamera> {
public:
    virtual ~ICamera() NOEXCEPT = default;

    virtual Ray get_ray(RandomGenerator &rng, const rreal s, const rreal t) const NOEXCEPT = 0;

    virtual std::shared_ptr<ICamera> deep_copy() const NOEXCEPT = 0;
};
