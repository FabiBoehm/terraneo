
#pragma once
#include <iostream>
#include <mpi.h>

#include <Kokkos_Core.hpp>

namespace terra::mpi {

using MPIRank = int;

inline MPIRank rank()
{
    MPIRank rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    return rank;
}

inline int num_processes()
{
    int num_processes;
    MPI_Comm_size( MPI_COMM_WORLD, &num_processes );
    return num_processes;
}

inline void barrier()
{  
    MPI_Barrier(MPI_COMM_WORLD);
}

namespace detail {
class MPIContext
{
  public:
    MPIContext( const MPIContext& )            = delete;
    MPIContext& operator=( const MPIContext& ) = delete;
    MPIContext( MPIContext&& )                 = delete;
    MPIContext& operator=( MPIContext&& )      = delete;

    /// Initialize MPI once. Safe to call only once.
    static void initialize( int* argc, char*** argv ) { instance( argc, argv ); }

    /// Query whether MPI is initialized
    static bool is_initialized()
    {
        int flag = 0;
        MPI_Initialized( &flag );
        return flag != 0;
    }

    /// Query whether MPI is finalized
    static bool is_finalized()
    {
        int flag = 0;
        MPI_Finalized( &flag );
        return flag != 0;
    }

  private:
    bool mpi_initialized_ = false;

    // private constructor
    MPIContext( int* argc, char*** argv )
    {
        if ( is_initialized() )
        {
            throw std::runtime_error( "MPI already initialized!" );
        }

        int err = MPI_Init( argc, argv );
        if ( err != MPI_SUCCESS )
        {
            char errstr[MPI_MAX_ERROR_STRING];
            int  len = 0;
            MPI_Error_string( err, errstr, &len );
            throw std::runtime_error( std::string( "MPI_Init failed: " ) + std::string( errstr, len ) );
        }

        mpi_initialized_ = true;

        MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI_ERRORS_RETURN );
    }

    // private destructor
    ~MPIContext()
    {
        if ( mpi_initialized_ && !is_finalized() )
        {
            int err = MPI_Finalize();
            if ( err != MPI_SUCCESS )
            {
                char errstr[MPI_MAX_ERROR_STRING];
                int  len = 0;
                MPI_Error_string( err, errstr, &len );
                std::cerr << "[MPI] MPI_Finalize failed: " << std::string( errstr, len ) << std::endl;
            }
        }
    }

    // singleton instance accessor
    static MPIContext& instance( int* argc = nullptr, char*** argv = nullptr )
    {
        static MPIContext guard( argc, argv );
        return guard;
    }
};

} // namespace detail

inline std::string mpi_error_string( int err )
{
    char errstr[MPI_MAX_ERROR_STRING];
    int  len = 0;
    MPI_Error_string( err, errstr, &len );
    return { errstr, static_cast< size_t >( len ) };
}

template < typename T >
MPI_Datatype mpi_datatype()
{
    static_assert( sizeof( T ) == 0, "No MPI datatype mapping for this type." );
    return MPI_DATATYPE_NULL;
}

template <>
inline MPI_Datatype mpi_datatype< char >()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype mpi_datatype< signed char >()
{
    return MPI_SIGNED_CHAR;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned char >()
{
    return MPI_UNSIGNED_CHAR;
}

template <>
inline MPI_Datatype mpi_datatype< int >()
{
    return MPI_INT;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned int >()
{
    return MPI_UNSIGNED;
}

template <>
inline MPI_Datatype mpi_datatype< short >()
{
    return MPI_SHORT;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned short >()
{
    return MPI_UNSIGNED_SHORT;
}

template <>
inline MPI_Datatype mpi_datatype< long >()
{
    return MPI_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned long >()
{
    return MPI_UNSIGNED_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< long long >()
{
    return MPI_LONG_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned long long >()
{
    return MPI_UNSIGNED_LONG_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< float >()
{
    return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_datatype< double >()
{
    return MPI_DOUBLE;
}

template <>
inline MPI_Datatype mpi_datatype< long double >()
{
    return MPI_LONG_DOUBLE;
}

// MPI defines no 16-bit floating-point datatype. half_t and bhalf_t are therefore
// transported as opaque 2-byte words, which is correct for the halo exchange because
// that is a pure copy: the additive reduction of ghost contributions happens in a
// Kokkos kernel after the receive, not inside MPI.
//
// These must never be handed to MPI_Allreduce with an arithmetic op -- MPI would add
// 16-bit patterns as unsigned integers. Reduce scalars with allreduce_scalar() below,
// which promotes to double.
// Guarded: where a backend has no native 16-bit type Kokkos aliases half_t/bhalf_t to
// float, and the float specialisation above already covers them.
#if !( defined( KOKKOS_HALF_T_IS_FLOAT ) && KOKKOS_HALF_T_IS_FLOAT )
template <>
inline MPI_Datatype mpi_datatype< Kokkos::Experimental::half_t >()
{
    return MPI_UINT16_T;
}
#endif

#if !( defined( KOKKOS_BHALF_T_IS_FLOAT ) && KOKKOS_BHALF_T_IS_FLOAT )
template <>
inline MPI_Datatype mpi_datatype< Kokkos::Experimental::bhalf_t >()
{
    return MPI_UINT16_T;
}
#endif

/// @brief Allreduce a single scalar, promoting to double for the reduction itself.
///
/// Needed because MPI has no 16-bit float datatype, and desirable anyway: a global
/// sum of per-rank partials accumulated in double loses nothing, whereas reducing in
/// the field precision would.
template < typename T >
inline void allreduce_scalar( T& value, MPI_Op op, MPI_Comm comm = MPI_COMM_WORLD )
{
    double promoted = static_cast< double >( value );
    MPI_Allreduce( MPI_IN_PLACE, &promoted, 1, MPI_DOUBLE, op, comm );
    value = static_cast< T >( promoted );
}

template <>
inline MPI_Datatype mpi_datatype< bool >()
{
    return MPI_CXX_BOOL;
}

} // namespace terra::mpi