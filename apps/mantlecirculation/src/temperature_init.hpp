#pragma once

#include <string>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "interpolators.hpp"
#include "io.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"
#include "parameters.hpp"
#include "shell/spherical_harmonics.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

/// Set the Q1 temperature field T from the configured initial-condition profile.
///
/// Three profiles are supported:
///   * FROM_FILE:   radial reference profile read from CSV, broadcast to Q1 nodes.
///   * CONDUCTIVE:  spherical steady-state conduction solution + optional
///                  spherical-harmonic perturbation (Y_l^m + factor·Y_l2^m2).
///   * power-law + noise: radial power-law profile with a per-node noise add.
///
/// The interpolators write a position-dependent value identically to every
/// subdomain copy of a shared node, so T is consistent without a halo exchange.
template < typename ScalarType >
void initialize_temperature_fields(
    linalg::VectorQ1Scalar< ScalarType >&                           T,
    grid::Grid2DDataScalar< ScalarType >&                           T_ref,
    const grid::shell::DistributedDomain&                           domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const Parameters&                                               prm )
{
    using util::logroot;
    const auto& init_temp = prm.physics_parameters.initial_temperature;

    if ( init_temp.profile == InitialTemperatureProfile::FROM_FILE )
    {
        util::logroot << "Reading reference temperature from: '" << init_temp.Tref_profile_csv_path << "'" << std::endl;

        // Read and populate reference temperature profile
        T_ref = shell::interpolate_radial_profile_into_subdomains_from_csv(
            init_temp.Tref_profile_csv_path,
            init_temp.Tref_profile_radii_key,
            init_temp.Tref_profile_temperature_key,
            coords_radii,
            1.0 / prm.mesh_parameters.mantle_thickness_m,
            1.0 / prm.boundary_parameters.delta_T_K );

        // Broadcast to Q1 nodes
        Kokkos::parallel_for(
            "initial temperature from profile",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            RadialProfileToQ1{ T_ref, T.grid_data() } );
        Kokkos::fence();

        // Add noise
        //        Kokkos::parallel_for(
        //            "noise to Q1 temperature",
        //            grid::shell::local_domain_md_range_policy_nodes( domain ),
        //            NoiseAdder{
        //                prm.boundary_parameters.temperature_min,
        //                prm.boundary_parameters.temperature_max,
        //                coords_shell,
        //                coords_radii,
        //                T.grid_data(),
        //                ownership_mask } );
        //        Kokkos::fence();

    }

    else if ( init_temp.profile == InitialTemperatureProfile::CONDUCTIVE )
    {
        const bool has_sph_2 =
            ( init_temp.sph_degree_l_2 > 0 && init_temp.sph_factor_2 != 0.0 && init_temp.sph_epsilon != 0.0 );

        logroot << "Initial temperature: conductive profile";
        if ( init_temp.sph_degree_l > 0 && init_temp.sph_epsilon != 0.0 )
        {
            logroot << " + eps * (Y_" << init_temp.sph_degree_l << "^" << init_temp.sph_order_m;
            if ( has_sph_2 )
            {
                logroot << " + " << init_temp.sph_factor_2 << " * Y_" << init_temp.sph_degree_l_2 << "^"
                        << init_temp.sph_order_m_2;
            }
            logroot << ") (eps=" << init_temp.sph_epsilon << ")";
        }
        logroot << std::endl;

        // Compute spherical-harmonic coefficients on unit-sphere Q1 nodes.
        grid::Grid3DDataScalar< ScalarType > sph_coeffs;
        const bool                           has_sph = ( init_temp.sph_degree_l > 0 && init_temp.sph_epsilon != 0.0 );
        if ( has_sph )
        {
            sph_coeffs = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                init_temp.sph_degree_l, init_temp.sph_order_m, coords_shell );

            if ( has_sph_2 )
            {
                auto sph_coeffs_2 = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                    init_temp.sph_degree_l_2, init_temp.sph_order_m_2, coords_shell );
                const ScalarType factor_2 = static_cast< ScalarType >( init_temp.sph_factor_2 );
                Kokkos::parallel_for(
                    "combine spherical harmonics",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                        { 0, 0, 0 },
                        { static_cast< int >( sph_coeffs.extent( 0 ) ),
                          static_cast< int >( sph_coeffs.extent( 1 ) ),
                          static_cast< int >( sph_coeffs.extent( 2 ) ) } ),
                    KOKKOS_LAMBDA( int sd, int x, int y ) {
                        sph_coeffs( sd, x, y ) += factor_2 * sph_coeffs_2( sd, x, y );
                    } );
                Kokkos::fence();
            }
        }

        // Fill Q1 T with the conductive profile + spherical-harmonic perturbation.
        Kokkos::parallel_for(
            "initial temp (conductive + sph. harm.)",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            ConductiveProfileInterpolator{
                domain.domain_info().radii().front(),
                domain.domain_info().radii().back(),
                init_temp.sph_epsilon,
                prm.boundary_parameters.temperature_min,
                coords_shell,
                coords_radii,
                T.grid_data(),
                sph_coeffs,
                has_sph } );
        Kokkos::fence();
        // NOTE: do NOT call send_recv here.  The kernel writes the same value to every
        // subdomain copy of each shared node already; send_recv (SUM) would accumulate
        // them and produce N*value at shared nodes.

        // Populate the radial reference profile T_ref used for the buoyancy deviation
        // Tdev = T - T_ref(r). For a conductive background the reference IS the conductive
        // profile (the Y_l^m perturbation has zero horizontal mean). Without this, T_ref
        // stays a default-constructed (null) Grid2DDataScalar and subtract_radial_profile()
        // dereferences it on device -> GPU null-pointer fault (only the FROM_FILE branch
        // populated it before).
        {
            const ScalarType r_min = domain.domain_info().radii().front();
            const ScalarType r_max = domain.domain_info().radii().back();
            const ScalarType T_min = static_cast< ScalarType >( prm.boundary_parameters.temperature_min );
            const int        nsub  = static_cast< int >( coords_radii.extent( 0 ) );
            const int        nr    = static_cast< int >( coords_radii.extent( 1 ) );
            T_ref     = grid::Grid2DDataScalar< ScalarType >( "T_ref_conductive", nsub, nr );
            auto tref = T_ref;
            auto rad  = coords_radii;
            Kokkos::parallel_for(
                "T_ref conductive profile",
                Kokkos::MDRangePolicy< Kokkos::Rank< 2 > >( { 0, 0 }, { nsub, nr } ),
                KOKKOS_LAMBDA( const int sd, const int r ) {
                    const ScalarType radius = rad( sd, r );
                    tref( sd, r )           = ( radius < ScalarType( 1e-15 ) )
                                                  ? ScalarType( 0 )
                                                  : ( r_min * r_max / radius - r_min ) / ( r_max - r_min ) + T_min;
                } );
            Kokkos::fence();
        }
    }
    else
    {
        logroot << "Initial temperature: power-law + noise" << std::endl;

        Kokkos::parallel_for(
            "initial temp interpolation (power-law)",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            InitialConditionInterpolator{
                domain.domain_info().radii().front(),
                domain.domain_info().radii().back(),
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                coords_shell,
                coords_radii,
                T.grid_data(),
                boundary_mask,
                /*only_boundary=*/false } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "adding noise to temp (power-law)",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            NoiseAdder{
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                coords_shell,
                coords_radii,
                T.grid_data(),
                ownership_mask } );
        Kokkos::fence();
    }
}

template < typename ScalarType >
void subtract_radial_profile(
    linalg::VectorQ1Scalar< ScalarType >&       dst,
    const linalg::VectorQ1Scalar< ScalarType >& src,
    const grid::Grid2DDataScalar< ScalarType >& profile,
    const grid::shell::DistributedDomain&       domain )
{
    Kokkos::parallel_for(
        "subtract_radial_profile",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        SubtractRadialProfile{ profile, src.grid_data(), dst.grid_data() } );
    Kokkos::fence();
}

/// Spherical steady-state conduction profile  T_cond(r) = r_min·r_max/r − r_min.
/// Used as the reference temperature for the Nusselt-number diagnostic and
/// added to XDMF output for visualisation.
template < typename ScalarType >
void compute_reference_conductive_profile(
    linalg::VectorQ1Scalar< ScalarType >&       T_cond,
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii,
    const Parameters&                           prm )
{
    Kokkos::parallel_for(
        "conductive profile T_cond",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        ConductiveProfileInterpolator{
            domain.domain_info().radii().front(),
            domain.domain_info().radii().back(),
            ScalarType( 0 ),
            prm.boundary_parameters.temperature_min,
            coords_shell,
            coords_radii,
            T_cond.grid_data(),
            {},
            false } );
    Kokkos::fence();
    // NOTE: do NOT call send_recv here.  Same rationale as in the IC kernel:
    // every subdomain copy of a shared node already gets the correct value, so
    // a SUM exchange would multiply it by the sharing count.
}

/// Load (u, T) from an XDMF checkpoint.
template < typename ScalarType >
void load_temperature_checkpoint(
    linalg::VectorQ1Vec< ScalarType, 3 >& u_velocity,
    linalg::VectorQ1Scalar< ScalarType >& T,
    const grid::shell::DistributedDomain&  domain,
    const Parameters&                      prm )
{
    using util::logroot;

    logroot << "Loading checkpoint from " << prm.io_parameters.checkpoint_dir << " at simulation step "
            << prm.io_parameters.checkpoint_step << std::endl;

    // Checking if checkpoint is dimensional or nondimensional
    auto metadata_result = io::read_xdmf_checkpoint_metadata( prm.io_parameters.checkpoint_dir );
    if ( metadata_result.is_err() )
    {
        Kokkos::abort( metadata_result.error().c_str() );
    }
    const auto& metadata = metadata_result.unwrap();

    if ( metadata.is_dimensional == -1 )
    {
        // Checkpoint version 0 or 1 - no flag present.
        logroot << "\nWARNING: Checkpoint predates is_dimensional flag (version " << metadata.version
                << "). Assuming nondimensional.\n"
                << std::endl;
    }

    else if ( static_cast< bool >( metadata.is_dimensional ) != prm.devel_parameters.output_dimensional )
    {
        logroot
            << "\nWARNING: Read and write checkpoint details are inconsistent - one is dimensional,  one is nondimensional.\n"
            << std::endl;
    }

    auto success_vel = io::read_xdmf_checkpoint_grid(
        prm.io_parameters.checkpoint_dir,
        std::string( "u_u" ),
        prm.io_parameters.checkpoint_step,
        domain,
        u_velocity.grid_data() );
    if ( success_vel.is_err() )
    {
        Kokkos::abort( success_vel.error().c_str() );
    }

    auto success_temp = io::read_xdmf_checkpoint_grid(
        prm.io_parameters.checkpoint_dir,
        std::string( "T" ),
        prm.io_parameters.checkpoint_step,
        domain,
        T.grid_data() );
    if ( success_temp.is_err() )
    {
        Kokkos::abort( success_temp.error().c_str() );
    }

    // Nondimensionalise checkpoint, if necessary
    if ( metadata.is_dimensional )
    {
        scale( T.grid_data(), 1.0 / prm.boundary_parameters.delta_T_K );
        scale( u_velocity.grid_data(), 1.0 / prm.physics_parameters.calc_cm_per_year );
    }
}

} // namespace terra::mantlecirculation
