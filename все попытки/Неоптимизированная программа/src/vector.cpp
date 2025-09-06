#include "vector.hpp"
#include <cmath>
#include <string_view>
#include <fstream>


constexpr Vector operator*(const Vector& left, numeric number){
    Vector result(left.size());
    for (unsigned int i = 0; i < left.size(); ++i)
        result[i] = left[i]*numeric;
    
    return result;
}

constexpr Vector operator*(numeric number, const Vector& right){
    Vector result(right.size());
    for (unsigned int i = 0; i < right.size(); ++i)
        result[i] = right[i]*numeric;
    
    return result;
}

constexpr Vector operator/(const Vector& left, numeric number){
    Vector result(left.size());
    for (unsigned int i = 0; i < left.size(); ++i)
        result[i] = left[i]/numeric;
    
    return result;
}

Vector& Vector::operator*=(numeric number){
    for (unsigned int i = 0; i < this->size(); ++i){
        this->operator[](i) *= number; 
    }
    return *this;
}

Vector& Vector::operator/=(numeric number){
    for (unsigned int i = 0; i < this->size(); ++i){
        this->operator[](i) /= number; 
    }
    return *this;
}

constexpr numeric Vector::norm() const{
    numeric result = 0;
    for (unsigned int i = 0; i < this->size(); ++i){
        const numeric number = this->operator[](i);
        result += number*number;
    }
    return std::sqrt(result);
}

void Vector::save(std::string_view filepath) const{
    std::ofstream file(filepath, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char*>(&this->size), sizeof(this->size));
    for (unsigned int i = 0; i < this->size; ++i){
        file.write(reinterpret_cast<const char*>(&this->operator[](i), sizeof(numeric));
    }
    file.close();
}

void Vector::load(const std::ifstream file){
    size_t size;
    file.read(reinterpret_cast<const char*>(&size), sizeof(size_t));
    this->resize(size);
    for (unsigned int i = 0; i < size; ++i){
        file.read(reinterpret_cast<char *>(&this->operator[](i)), sizeof(numeric));
    }
}
