#pragma once

// SubdomainInfo: identity of one macro-block (subdomain) of the icosahedral shell -- (diamond, x, y, r)
// indices plus a packed 64-bit global_id(). Extracted from spherical_shell.hpp so it can be shared with
// dependency-free consumers (e.g. the AMR forest) without pulling in Kokkos/MPI.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>

namespace terra::grid::shell {

class SubdomainInfo
{
  public:
    /// @brief Creates invalid ID.
    SubdomainInfo()
    : diamond_id_( -1 )
    , subdomain_x_( -1 )
    , subdomain_y_( -1 )
    , subdomain_r_( -1 )
    {}

    /// @brief Creates unique subdomain ID.
    SubdomainInfo( int diamond_id, int subdomain_x, int subdomain_y, int subdomain_r )
    : diamond_id_( diamond_id )
    , subdomain_x_( subdomain_x )
    , subdomain_y_( subdomain_y )
    , subdomain_r_( subdomain_r )
    {}

    /// @brief Read from encoded 64-bit integer.
    ///
    /// See \ref global_id() for format.
    explicit SubdomainInfo( const int64_t global_id )
    : diamond_id_( static_cast< int >( ( global_id >> 57 ) ) )
    , subdomain_x_( static_cast< int >( ( global_id >> 0 ) & ( ( 1 << 19 ) - 1 ) ) )
    , subdomain_y_( static_cast< int >( ( global_id >> 19 ) & ( ( 1 << 19 ) - 1 ) ) )
    , subdomain_r_( static_cast< int >( ( global_id >> 38 ) & ( ( 1 << 19 ) - 1 ) ) )
    {
        if ( global_id != this->global_id() )
        {
            throw std::logic_error( "Invalid global ID conversion." );
        }
    }

    /// @brief Diamond that subdomain is part of.
    int diamond_id() const { return diamond_id_; }

    /// @brief Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x() const { return subdomain_x_; }

    /// @brief Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y() const { return subdomain_y_; }

    /// @brief Subdomain index in the radial direction (local to the diamond).
    int subdomain_r() const { return subdomain_r_; }

    bool operator<( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) <
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

    bool operator==( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) ==
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

    /// @brief Scrambles the four indices (diamond ID, x, y, r) into a single integer.
    ///
    /// Format
    /// @code
    ///
    /// bits (LSB)  0-18        (19 bits): subdomain_x
    /// bits       19-37        (19 bits): subdomain_y
    /// bits       38-56        (19 bits): subdomain_r
    /// bits       57-63 (MSB)  ( 7 bits): diamond_id (in [0, ..., 9])
    ///
    /// @endcode
    [[nodiscard]] int64_t global_id() const
    {
        if ( diamond_id_ >= 10 )
        {
            throw std::logic_error( "Diamond ID must be less than 10." );
        }

        if ( subdomain_x_ > ( 1 << 19 ) - 1 || subdomain_y_ > ( 1 << 19 ) - 1 || subdomain_r_ > ( 1 << 19 ) - 1 )
        {
            throw std::logic_error( "Subdomain indices too large." );
        }

        return ( static_cast< int64_t >( diamond_id_ ) << 57 ) | ( static_cast< int64_t >( subdomain_r_ ) << 38 ) |
               ( static_cast< int64_t >( subdomain_y_ ) << 19 ) | ( static_cast< int64_t >( subdomain_x_ ) );
    }

  private:
    /// Diamond that subdomain is part of.
    int diamond_id_;

    /// Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x_;

    /// Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y_;

    /// Subdomain index in radial direction.
    int subdomain_r_;
};

inline std::ostream& operator<<( std::ostream& os, const SubdomainInfo& si )
{
    os << "Diamond ID: " << si.diamond_id() << ", subdomains (" << si.subdomain_x() << ", " << si.subdomain_y() << ", "
       << si.subdomain_r() << ")";
    return os;
}

} // namespace terra::grid::shell
