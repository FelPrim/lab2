#pragma once
#include "useful_math.hpp"
#include <valarray>
#include <string_view>
#include <fstream>

using Valarray = std::valarray<numeric>

class Vector: public Valarray{
public:
    Vector() = default;
    explicit Vector(size_t length): Valarray(length){}
    explicit Vector(const Valarray& other): Valarray(other){}
    explicit Vector(Valarray&& other): Valarray(other){}
    explicit Vector(const Vector&) = default;
    explicit Vector(Vector&&) noexcept = default;
    Vector& operator=(const Vector&) = default;
    Vector& operator=(Vector&&) noexcept = default;
    friend Vector operator*(const Vector&, numeric);
    friend Vector operator*(numeric, const Vector&);
    friend Vector operator/(const Vector&, numeric);
    Vector& operator*=(numeric);
    Vector& operator/=(numeric);

    constexpr numeric norm() const; 
    void load(const std::ifstream file);
    void save(std::string_view filepath) const;
}
