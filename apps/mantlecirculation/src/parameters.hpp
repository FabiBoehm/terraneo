#pragma once

#include <string>
#include <variant>

#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/result.hpp"

namespace terra::mantlecirculation {

using ScalarType = double;

struct MeshParameters
{
    int refinement_level_mesh_min   = 1;
    int refinement_level_mesh_max   = 4;
    int refinement_level_subdomains = 0;

    // Nondimensional radii
    double radius_min = 0.5;
    double radius_max = 1.0;
    // Dimensional radii in meter
    double radius_surface_m   = 6371000.0;
    double radius_cmb_m       = 3480000.0;
    double mantle_thickness_m = 2891000.0;

    // Anisotropic refinement (Path B.1): radial diamond level is set to
    // (refinement_level_mesh_* + radial_extra_levels) at every MG level, so the
    // full hierarchy stays consistent (both axes halve per step). lat_sdr/rad_sdr
    // override refinement_level_subdomains per axis when >= 0.
    int radial_extra_levels = -1;
    int lat_sdr             = -1;
    int rad_sdr             = -1;

    /// Selector for the radial-shell distribution.  All non-uniform variants
    /// use a tanh map (see grid::shell::make_tanh_*_cluster) parameterised by
    /// `radial_cluster_k`; `radial_cluster_k <= 0` collapses each variant to
    /// the uniform distribution.
    enum class RadialDistribution
    {
        UNIFORM,      ///< equispaced shells (default).
        TANH_BOTH,    ///< both-side clustering at CMB and surface.
        TANH_CMB,     ///< one-side clustering at the inner boundary (CMB).
        TANH_SURFACE, ///< one-side clustering at the outer boundary (surface).
    };

    RadialDistribution radial_distribution = RadialDistribution::UNIFORM;
    double             radial_cluster_k    = 1.0;
};

struct PlateParameters
{
    bool apply_plate_velocities = false; // This does nothing yet
    int  initial_plate_age      = 400;
    int  final_plate_age        = 0;

    double plate_velocity_scaling = 1.0;
};
struct BoundaryConditionsParameters
{
    enum class VelocityBC
    {
        NO_SLIP,
        FREE_SLIP,
    };

    VelocityBC velocity_bc_cmb     = VelocityBC::NO_SLIP;
    VelocityBC velocity_bc_surface = VelocityBC::NO_SLIP;

    // Nondimensional temperatures
    double temperature_min = 0.0;
    double temperature_max = 1.0;
    // Dimensional temperatures in Kelvin
    double temperature_cmb_K     = 3800.0;
    double temperature_surface_K = 300.0;
    double delta_T_K             = temperature_cmb_K - temperature_surface_K;

    PlateParameters plate_parameters{};
};

/// Choice of viscosity law for the temperature-dependent viscosity field.
///   CONSTANT          : eta(T) = const (taken from `reference_viscosity`, optionally
///                       multiplied by a radial profile if `radial_profile_enabled`).
///   FRANK_KAMENETSKII : eta(T) = rmu^(0.5 - T)  (Zhong et al. 2008).  T in [0,1].
///                       Total cold/hot viscosity contrast = rmu (rmu = 1 → constant
///                       viscosity).  See `ViscosityParameters::rmu`.
enum class ViscosityLaw
{
    CONSTANT,
    FRANK_KAMENETSKII,
};

struct ViscosityParameters
{
    /// Viscosity law selector — see ViscosityLaw above.
    ViscosityLaw law = ViscosityLaw::CONSTANT;

    /// Base of the Frank-Kamenetskii viscosity law (Zhong et al. 2008): eta = rmu^(0.5 - T).
    /// Total cold/hot viscosity contrast = rmu (rmu = 1 → constant viscosity).
    /// Ignored when `law == CONSTANT`.
    double rmu = 1.0;

    /// If true, multiply the temperature-dependent viscosity by a radial reference profile
    /// eta_ref(r) read from CSV.  The on-disk file gives a 1D profile (radii column +
    /// viscosity column); cell-wise viscosity is linearly interpolated to the cell radius
    /// and *multiplied* into the temperature-dependent factor.  Useful for laterally
    /// uniform but radially varying viscosity (e.g. realistic mantle profiles).
    bool radial_profile_enabled = false;

    /// Path to the CSV file containing the radial viscosity profile.
    std::string radial_profile_csv_filename = "radial_viscosity_profile.csv";
    /// CSV column name for the radii (first column).  Radii must be in the same units
    /// as `radius_min`/`radius_max` (i.e. non-dimensional shell radii).
    std::string radial_profile_radii_key = "radii";
    /// CSV column name for the viscosity values (second column), in units of
    /// `reference_viscosity`.
    std::string radial_profile_viscosity_key = "viscosity";

    /// Multiplicative reference viscosity scale.  Final viscosity is
    /// `reference_viscosity * eta(T) * eta_ref(r)`.
    /// When `law == CONSTANT` and `radial_profile_enabled == false`, this is just the
    /// constant viscosity value.
    double reference_viscosity = 1e23;
    double viscosity           = 1.0;
};

/// Initial temperature distribution.
///   POWER_LAW  : T_init from a radial power-law profile (legacy default).
///   CONDUCTIVE : T_init = analytic conduction profile (r_min*r_max/r - r_min) / D.
///                Use this in combination with a spherical harmonic perturbation for the standard mantle
///                convection benchmarks (Zhong et al. 2008 A/C cases).
///   FROM_FILE  : Custom T_init read from csv file with path set in 'Tref_profile_csv_path'.
enum class InitialTemperatureProfile
{
    POWER_LAW,
    CONDUCTIVE,
    FROM_FILE
};

/// Initial temperature perturbation.
/// Choice between random noise and a spherical harmonic perturbation pattern, defined with parameters 'sph_degree_l' and 'sph_order_m'.
/// Amplitudes for both cases are set with the parameter 'perturbation_amplitude', in fraction of global temperature difference.
enum class InitialPerturbation
{
    NOISE,
    SPHERICAL_HARMONICS
};

struct InitialTemperatureParameters
{
    /// Selectors for the initial temperature distribution — see InitialTemperatureProfile and InitialPerturbation.
    InitialTemperatureProfile profile      = InitialTemperatureProfile::FROM_FILE;
    InitialPerturbation       perturbation = InitialPerturbation::SPHERICAL_HARMONICS;

    // Reference temperature from file
    std::string Tref_profile_csv_path        = "TemperatureProfile_3800K.csv";
    std::string Tref_profile_radii_key       = "radius";
    std::string Tref_profile_temperature_key = "temperature";

    double perturbation_amplitude = 5e-2;

    /// Spherical-harmonic perturbation degree l and order m (l >= 0, |m| <= l). Set l to 0 to
    /// disable the SH perturbation entirely (then perturbation_amplitude is ignored).
    int sph_degree_l = 0;
    int sph_order_m  = 0;

    /// Optional second spherical harmonic for combined modes (e.g. cubic symmetry
    /// T_perturb = Y_4^0 + (5/7) * Y_4^4).  Set sph_degree_l_2 > 0 to enable.
    /// The total perturbation is perturbation_amplitude * (Y_l1^m1 + sph_factor_2 * Y_l2^m2).
    int sph_degree_l_2 = 0;
    int sph_order_m_2  = 0;
    /// Relative amplitude of the second harmonic (typical values are O(1); e.g. 5/7 for
    /// the Zhong C3 cubic-symmetry benchmark).
    double sph_factor_2 = 0.0;
};

struct PhysicsParameters
{
    double gravity = 9.81;

    // Non-dimensional numbers
    double rayleigh_number    = 1e5;
    double peclet_number      = 1.0;
    double dissipation_number = 1.0;
    double h_number           = 1.0;

    double thermal_diffusivity_dim    = 1.0;
    double thermal_diffusivity_nondim = 1.0;
    double characteristic_velocity    = 1e-10; // diffusive velocity

    double reference_density      = 4500;
    double surface_density_dim    = 3300;
    double surface_density_nondim = 1.0;

    double thermal_expansivity    = 2.5e-5;
    double thermal_conductivity   = 3.0;
    double specific_heat_capacity = 1230;
    double grueneisen_parameter   = 1.1;

    bool   internal_heating      = false;
    double internal_heating_rate = 1.0;

    bool compressible = false;

    double calc_cm_per_year = 3e-4; // from non-dim velocity to cm/a
    double calc_time_Ma     = 1e6;  // from non-dim time to Ma

    ViscosityParameters          viscosity_parameters{};
    InitialTemperatureParameters initial_temperature{};
};

/// Storage/working precision of the velocity-block multigrid V-cycle preconditioner.
/// The outer Stokes FGMRES, block preconditioner and operators stay double; only the
/// MG hierarchy (operators, smoothers, transfers, coarse solve, level vectors) runs in
/// this precision, with convert at the preconditioner boundary. Lower precision trims
/// the MG memory; the outer double Krylov solve absorbs the preconditioner inexactness.
enum class MGPrecision
{
    DOUBLE,
    FLOAT,
    HALF,
};

struct StokesSolverParameters
{
    int    krylov_restart            = 10;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    /// Store the outer FGMRES Krylov basis in single precision while the operator,
    /// preconditioner and orthogonalization stay double. Roughly halves the FGMRES
    /// workspace memory; convergence is unaffected (the operator never sees float).
    bool   float_krylov_basis        = false;

    /// Precision of the velocity-block multigrid V-cycle preconditioner (see MGPrecision).
    MGPrecision mg_precision = MGPrecision::DOUBLE;

    int viscous_pc_num_vcycles                 = 1;
    int viscous_pc_chebyshev_order             = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
    int viscous_pc_num_power_iterations        = 10;

    /// Galerkin coarse-grid approximation mode for the multigrid preconditioner of the
    /// viscous Stokes block.
    ///   0 : disabled — coarse operators are re-discretised on every level (cheap setup,
    ///       but coarse smoothers see a different operator than the fine one).  Default.
    ///   1 : full GCA — store and assemble the full coarse-grid Galerkin matrices for
    ///       every coarse level.  Most robust setup; usually faster overall at high
    ///       viscosity contrast / variable viscosity.
    ///   2 : adaptive GCA — only store/assemble the coarse-grid matrices for elements
    ///       flagged by `GCAElementsCollector`, leaving the rest re-discretised.
    ///       Uses less memory than mode 1 but slightly less robust.
    int gca = 0;

    /// Per-descent agglomeration factors for the viscous MG preconditioner.
    /// Length must equal num_mg_levels - 1 (one factor per coarse descent step).
    /// Empty = classical MG, all levels on MPI_COMM_WORLD.
    /// Factor f > 1 at descent i means the comm shrinks by f ranks going from
    /// MG level max-i-1 to level max-i-2. Factor 1 = identity (no shrink).
    std::vector< int > viscous_pc_agglom_factors = {};
};

/// Time-discretization scheme for the energy (temperature) equation.
///   FCT  : explicit Flux-Corrected Transport on the FV mesh.  Low-order upwind
///          predictor + Zalesak limiter (monotone, no over/undershoots).
///          Stability bound: dt <= dt_stable (computed from advective + diffusive
///          face fluxes).  Cheap per step but requires small dt at high velocity / Pe.
///   SUPG : implicit SUPG-stabilised Galerkin advection-diffusion on the Q1 mesh,
///          solved by FGMRES.  Unconditionally stable (dt only bounded by the
///          *advection* CFL for accuracy), so allows much larger dt at moderate Pe.
///          Linear-solver convergence degrades at high Pe (Ra >> 1e6).
enum class EnergySolverType
{
    FCT,
    SUPG,
    ENTROPY_VISCOSITY,
};
struct EnergySolverParameters
{
    int    krylov_restart            = 5;
    int    krylov_max_iterations     = 100;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;
    int    energy_substeps           = 1;

    /// Store the energy FGMRES Krylov basis in single precision (operator stays
    /// double). The energy advection-diffusion solve is well-conditioned and runs
    /// fine in reduced precision; this trims the energy FGMRES workspace.
    bool   float_krylov_basis        = false;

    /// Entropy-viscosity stabilization parameters (only used when
    /// `energy_solver == ENTROPY_VISCOSITY`).  Defaults match ASPECT.
    double ev_alpha_max = 0.078; ///< First-order upwind cap on ν_h (= 0.026·d in 3D).
    double ev_alpha_E   = 1.0;   ///< Residual-branch scale.

    /// If true, log global min/max/mean of the per-wedge ν_h field once per
    /// output_frequency to <outdir>/nu_h_stats.csv (timestep, min, max, mean).
    bool ev_dump_nu_h = false;

    EnergySolverType energy_solver = EnergySolverType::ENTROPY_VISCOSITY;
};

struct TimeSteppingParameters
{
    double dt_scaling = 0.5;
    double t_end_Ma   = 100.0;
    double t_end      = 1.0;
    double dt_max_Ma  = 5.0;
    double dt_min_Ma  = 0.1;
    double dt_max     = 1.0;
    double dt_min     = 1.0;

    int max_timesteps         = 10;
    int timestep_initial      = 0;
    int initial_dt_ramp_steps = 20;

    int picard_iterations = 1;
};

struct IOParameters
{
    std::string outdir          = "output";
    bool        overwrite       = false;
    bool        output_pressure = true;

    std::string xdmf_dir                = "xdmf";
    std::string radial_profiles_out_dir = "radial_profiles";
    std::string timer_trees_dir         = "timer_trees";

    bool        load_checkpoint = false;
    std::string checkpoint_dir;
    int         checkpoint_step = -1;

    int output_frequency = 1;

    bool no_xdmf            = false;
    bool no_radial_profiles = false;
};

// This struct holds options that might be useful for debugging, benchmarking, etc., but are not intended for 'standard' use.
struct DeveloperOptions
{
    bool set_nondimensional_numbers = false;
    bool output_dimensional         = true;
};

struct Parameters
{
    MeshParameters               mesh_parameters;
    BoundaryConditionsParameters boundary_parameters;
    StokesSolverParameters       stokes_solver_parameters;
    EnergySolverParameters       energy_solver_parameters;
    PhysicsParameters            physics_parameters;
    TimeSteppingParameters       time_stepping_parameters;
    IOParameters                 io_parameters;
    DeveloperOptions             devel_parameters;

    std::string output_config_file;
};

struct CLIHelp
{};

inline void nondimensionalise( Parameters& prm )
{
    auto& phys     = prm.physics_parameters;
    auto& mesh     = prm.mesh_parameters;
    auto& boundary = prm.boundary_parameters;
    auto& devel    = prm.devel_parameters;
    auto& time     = prm.time_stepping_parameters;

    // --- Domain ---

    // Nondimensionalise radii with mantle thickness -- rescaled to unit sphere for output
    mesh.mantle_thickness_m = mesh.radius_surface_m - mesh.radius_cmb_m;
    mesh.radius_max         = mesh.radius_surface_m / mesh.mantle_thickness_m;
    mesh.radius_min         = mesh.radius_cmb_m / mesh.mantle_thickness_m;

    // --- Boundary conditions ---

    boundary.temperature_min = boundary.temperature_surface_K / boundary.delta_T_K;
    boundary.temperature_max = boundary.temperature_cmb_K / boundary.delta_T_K;

    // Surface density
    phys.surface_density_nondim = phys.surface_density_dim / phys.reference_density;

    // Compute characteristic velocity and thermal diffusivity
    phys.characteristic_velocity =
        phys.thermal_conductivity / ( phys.reference_density * phys.specific_heat_capacity * mesh.mantle_thickness_m );

    phys.thermal_diffusivity_dim = phys.thermal_conductivity / ( phys.reference_density * phys.specific_heat_capacity );

    // Precompute conversion factors from non-dim to dimensional quantities
    phys.calc_cm_per_year = phys.characteristic_velocity * 60 * 60 * 24 * 365 * 100; // Velocity in cm/a

    phys.calc_time_Ma = mesh.mantle_thickness_m / ( phys.calc_cm_per_year * 1e4 ); // Time in Ma

    // Acount for plate velocity scaling
    if ( boundary.plate_parameters.apply_plate_velocities )
    {
        phys.calc_time_Ma /= boundary.plate_parameters.plate_velocity_scaling;
    }

    // Nondimensionalise time
    time.t_end  = time.t_end_Ma / phys.calc_time_Ma;
    time.dt_max = time.dt_max_Ma / phys.calc_time_Ma;
    time.dt_min = time.dt_min_Ma / phys.calc_time_Ma;

    if ( !devel.set_nondimensional_numbers )
    {
        // Compute nondimensional numbers
        // Rayleigh number = ( rho * alpha * g * L^3 * dT ) / ( eta * kappa )
        phys.rayleigh_number = ( phys.reference_density * phys.gravity * phys.thermal_expansivity *
                                 std::pow( mesh.mantle_thickness_m, 3 ) * boundary.delta_T_K ) /
                               ( phys.viscosity_parameters.reference_viscosity * phys.thermal_diffusivity_dim );

        // Peclet number = ( U * L ) / kappa -> should be 1
        phys.peclet_number = ( phys.characteristic_velocity * mesh.mantle_thickness_m ) / phys.thermal_diffusivity_dim;

        // Dissipation number = ( alpha * g * L ) / Cp
        phys.dissipation_number =
            ( phys.thermal_expansivity * phys.gravity * mesh.mantle_thickness_m ) / phys.specific_heat_capacity;

        // H-number = ( H * L ) / ( Cp * U * dT )
        phys.h_number = ( phys.internal_heating_rate * mesh.mantle_thickness_m ) /
                        ( phys.specific_heat_capacity * phys.characteristic_velocity * boundary.delta_T_K );
    }
}

inline util::Result< std::variant< CLIHelp, Parameters > > parse_parameters( int argc, char** argv )
{
    CLI::App app{ "Mantle circulation simulation." };

    Parameters parameters{};

    using util::add_flag_with_default;
    using util::add_option_with_default;

    // Allow config files
    app.set_config( "--config" );

    // --low-mem preset. Detect it BEFORE registering the options below so its five
    // constituent settings become the captured CLI *defaults*: FP16 Krylov basis for the
    // Stokes and energy solves, Stokes restart lowered 10 -> 5 (energy is already 5), and a
    // single pre/post velocity-MG smoothing step (vs 2). These minimise the FGMRES
    // workspace, the dominant memory term at high dofs/GCD. Making them defaults (rather
    // than patching after parse) means an explicit --stokes-*/--energy-* flag still
    // overrides through normal parsing, and --write-config-and-exit emits the expanded
    // preset so a generated config round-trips correctly.
    bool low_mem = false;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string a{ argv[i] };
        if ( a == "--low-mem" || a.rfind( "--low-mem=", 0 ) == 0 )
        {
            low_mem = true;
            break;
        }
    }
    if ( low_mem )
    {
        parameters.stokes_solver_parameters.float_krylov_basis                     = true;
        parameters.energy_solver_parameters.float_krylov_basis                     = true;
        parameters.stokes_solver_parameters.krylov_restart                         = 5;
        parameters.energy_solver_parameters.krylov_restart                         = 5;
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost = 1;
    }

    ///////////////
    /// General ///
    ///////////////

    add_option_with_default(
        app,
        "--write-config-and-exit",
        parameters.output_config_file,
        "Writes a config file with the passed (or default arguments) to the desired location to be then modified and passed. E.g., '--write-config-and-exit my-config.toml'.\n"
        "IMPORTANT: THIS OPTION MUST BE REMOVED IN THE GENERATED CONFIG OR ELSE YOU WILL OVERWRITE IT AGAIN" )
        ->group( "General" );

    ///////////////////////
    /// Domain and mesh ///
    ///////////////////////

    add_option_with_default( app, "--refinement-level-mesh-min", parameters.mesh_parameters.refinement_level_mesh_min )
        ->group( "Domain" );
    add_option_with_default( app, "--refinement-level-mesh-max", parameters.mesh_parameters.refinement_level_mesh_max )
        ->group( "Domain" );

    add_option_with_default(
        app, "--refinement-level-subdomains", parameters.mesh_parameters.refinement_level_subdomains )
        ->group( "Domain" );

    add_option_with_default( app, "--radius-cmb", parameters.mesh_parameters.radius_cmb_m )->group( "Domain" );
    add_option_with_default( app, "--radius-surface", parameters.mesh_parameters.radius_surface_m )->group( "Domain" );

    add_option_with_default( app, "--radial-extra-levels", parameters.mesh_parameters.radial_extra_levels )
        ->group( "Domain" )
        ->description( "Per-MG-level offset added to the radial diamond refinement level relative to the "
                       "lateral one. Radial level at each MG level L becomes L + radial_extra_levels, so the "
                       "hierarchy coarsens uniformly in both axes. Default 0 = isotropic." );
    add_option_with_default( app, "--lat-sdr", parameters.mesh_parameters.lat_sdr )
        ->group( "Domain" )
        ->description(
            "Override the lateral subdomain refinement level (otherwise --refinement-level-subdomains is used)." );
    add_option_with_default( app, "--rad-sdr", parameters.mesh_parameters.rad_sdr )
        ->group( "Domain" )
        ->description(
            "Override the radial subdomain refinement level (otherwise --refinement-level-subdomains is used)." );

    std::map< std::string, MeshParameters::RadialDistribution > radial_distribution_map{
        { "uniform", MeshParameters::RadialDistribution::UNIFORM },
        { "tanh-both", MeshParameters::RadialDistribution::TANH_BOTH },
        { "tanh-cmb", MeshParameters::RadialDistribution::TANH_CMB },
        { "tanh-surface", MeshParameters::RadialDistribution::TANH_SURFACE },
    };
    add_option_with_default( app, "--radial-distribution", parameters.mesh_parameters.radial_distribution )
        ->transform( CLI::CheckedTransformer( radial_distribution_map, CLI::ignore_case ) )
        ->group( "Domain" )
        ->description( "Radial shell distribution: 'uniform' (equispaced, default), 'tanh-both' "
                       "(cluster at both CMB and surface), 'tanh-cmb' (cluster at CMB only), "
                       "'tanh-surface' (cluster at surface only).  Cluster strength is set by "
                       "--radial-cluster-k." );
    add_option_with_default( app, "--radial-cluster-k", parameters.mesh_parameters.radial_cluster_k )
        ->group( "Domain" )
        ->description( "Cluster-strength k for the tanh-based radial distributions.  k <= 0 "
                       "collapses to uniform; k ~ 1 mild clustering, k ~ 2 strong clustering." );

    ///////////////////////////
    /// Boundary conditions ///
    ///////////////////////////

    std::map< std::string, BoundaryConditionsParameters::VelocityBC > velocity_bc_cmb_map{
        { "noslip", BoundaryConditionsParameters::VelocityBC::NO_SLIP },
        { "freeslip", BoundaryConditionsParameters::VelocityBC::FREE_SLIP },
    };

    std::map< std::string, BoundaryConditionsParameters::VelocityBC > velocity_bc_surface_map{
        { "noslip", BoundaryConditionsParameters::VelocityBC::NO_SLIP },
        { "freeslip", BoundaryConditionsParameters::VelocityBC::FREE_SLIP },
    };

    add_option_with_default( app, "--velocity-bc-cmb", parameters.boundary_parameters.velocity_bc_cmb )
        ->transform( CLI::CheckedTransformer( velocity_bc_cmb_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--velocity-bc-surface", parameters.boundary_parameters.velocity_bc_surface )
        ->transform( CLI::CheckedTransformer( velocity_bc_surface_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--temperature-cmb", parameters.boundary_parameters.temperature_cmb_K )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--temperature-surface", parameters.boundary_parameters.temperature_surface_K )
        ->group( "Boundary Conditions" );

    //////////////////////////////
    /// Geophysical parameters ///
    //////////////////////////////
    add_flag_with_default( app, "--compressible", parameters.physics_parameters.compressible );
    add_flag_with_default( app, "--internal-heating-enabled", parameters.physics_parameters.internal_heating );
    add_option_with_default( app, "--internal-heating-rate", parameters.physics_parameters.internal_heating_rate );

    add_option_with_default( app, "--reference-density", parameters.physics_parameters.reference_density );
    add_option_with_default( app, "--thermal-expansivity", parameters.physics_parameters.thermal_expansivity );
    add_option_with_default( app, "--thermal-conductivity", parameters.physics_parameters.thermal_conductivity );
    add_option_with_default( app, "--specific-heat-capacity", parameters.physics_parameters.specific_heat_capacity );

    // Viscosity parameters
    add_option_with_default(
        app, "--reference-viscosity", parameters.physics_parameters.viscosity_parameters.reference_viscosity )
        ->group( "Viscosity" );
    const auto radial_profile_enabled =
        add_flag_with_default(
            app,
            "--viscosity-radial-profile",
            parameters.physics_parameters.viscosity_parameters.radial_profile_enabled )
            ->group( "Viscosity" )
            ->description(
                "Add this flag if you want to supply a radial viscosity profile. "
                "Then use further flags/arguments (starting with --viscosity-radial-profile-<...>) to specify the file path etc. "
                "If you omit this flag, the viscosity is set to const (eta = 1)." );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-csv-filename",
        parameters.physics_parameters.viscosity_parameters.radial_profile_csv_filename )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-radii-key",
        parameters.physics_parameters.viscosity_parameters.radial_profile_radii_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-value-key",
        parameters.physics_parameters.viscosity_parameters.radial_profile_viscosity_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );

    std::map< std::string, ViscosityLaw > viscosity_law_map{
        { "constant", ViscosityLaw::CONSTANT },
        { "frank-kamenetskii", ViscosityLaw::FRANK_KAMENETSKII },
    };

    add_option_with_default( app, "--viscosity-law", parameters.physics_parameters.viscosity_parameters.law )
        ->transform( CLI::CheckedTransformer( viscosity_law_map, CLI::ignore_case ) )
        ->default_val( "constant" )
        ->group( "Viscosity" )
        ->description( "Viscosity law to use. 'constant' uses a constant or radial profile. "
                       "'frank-kamenetskii' computes eta = rmu^(0.5 - T) (Zhong et al. 2008)." );

    add_option_with_default( app, "--viscosity-rmu", parameters.physics_parameters.viscosity_parameters.rmu )
        ->group( "Viscosity" )
        ->description( "Base of the Frank-Kamenetskii viscosity law: eta = rmu^(0.5 - T) "
                       "(Zhong et al. 2008). Cold/hot contrast = rmu; rmu = 1 gives constant viscosity." );

    ///////////////////////////////
    /// Initial temperature      ///
    ///////////////////////////////

    std::map< std::string, InitialTemperatureProfile > init_temp_profile_map{
        { "power-law", InitialTemperatureProfile::POWER_LAW },
        { "conductive", InitialTemperatureProfile::CONDUCTIVE },
        { "from-file", InitialTemperatureProfile::FROM_FILE } };

    add_option_with_default(
        app, "--initial-temperature-profile", parameters.physics_parameters.initial_temperature.profile )
        ->transform( CLI::CheckedTransformer( init_temp_profile_map, CLI::ignore_case ) )
        ->default_val( "from-file" )
        ->group( "Initial Temperature" )
        ->description( "'power-law': T = ((r_max-r)/(r_max-r_min))^5 (default). "
                       "'from-file': read custom temperature profile from file."
                       "'conductive': T_ref = (r_min*r_max/r - r_min)/(r_max - r_min)." );

    add_option_with_default(
        app, "--temperature-profile-path", parameters.physics_parameters.initial_temperature.Tref_profile_csv_path )
        ->group( "Initial Temperature" )
        ->description( "Path to file for '--initial-temperature-profile == from-file'." );

    std::map< std::string, InitialPerturbation > init_temp_perturbation_map{
        { "noise", InitialPerturbation::NOISE },
        { "spherical-harmonics", InitialPerturbation::SPHERICAL_HARMONICS },
        { "sph", InitialPerturbation::SPHERICAL_HARMONICS } };

    add_option_with_default(
        app, "--initial-temperature-perturbation", parameters.physics_parameters.initial_temperature.perturbation )
        ->transform( CLI::CheckedTransformer( init_temp_perturbation_map, CLI::ignore_case ) )
        ->default_val( "spherical-harmonics" )
        ->group( "Initial Temperature" )
        ->description( "'noise' or 'spherical-harmonics' / 'sph' perturbation." );

    add_option_with_default(
        app, "--perturbation-amplitude", parameters.physics_parameters.initial_temperature.perturbation_amplitude )
        ->group( "Initial Temperature" )
        ->description( "Fractional amplitude of added noise/spherical harmonic perturbation." );

    add_option_with_default(
        app, "--initial-temperature-sph-degree", parameters.physics_parameters.initial_temperature.sph_degree_l )
        ->group( "Initial Temperature" )
        ->description( "Spherical harmonic degree l for initial temperature perturbation (0 = none)." );

    add_option_with_default(
        app, "--initial-temperature-sph-order", parameters.physics_parameters.initial_temperature.sph_order_m )
        ->group( "Initial Temperature" )
        ->description( "Spherical harmonic order m for initial temperature perturbation." );

    add_option_with_default(
        app, "--initial-temperature-sph-degree-2", parameters.physics_parameters.initial_temperature.sph_degree_l_2 )
        ->group( "Initial Temperature" )
        ->description( "Optional second spherical harmonic degree l2 (0 = none). For combined modes." );

    add_option_with_default(
        app, "--initial-temperature-sph-order-2", parameters.physics_parameters.initial_temperature.sph_order_m_2 )
        ->group( "Initial Temperature" )
        ->description( "Optional second spherical harmonic order m2." );

    add_option_with_default(
        app, "--initial-temperature-sph-factor-2", parameters.physics_parameters.initial_temperature.sph_factor_2 )
        ->group( "Initial Temperature" )
        ->description(
            "Weight factor for second spherical harmonic: T += eps * envelope * (Y_l1^m1 + factor_2 * Y_l2^m2)." );

    ///////////////////////////
    /// Time discretization ///
    ///////////////////////////

    add_option_with_default( app, "--dt-scaling", parameters.time_stepping_parameters.dt_scaling )
        ->description(
            "A robust (stable) dt is computed the the actual face-normal velocity fluxes and cell volumes via a "
            "parallel reduce over all cells. However, a smaller value might still be desired due to accuracy "
            "considerations. You can scale the computed dt using this value (e.g. set to 0.5 to half the estimated dt, "
            "set to 1.0 to just use the estimated dt)." )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--t-end", parameters.time_stepping_parameters.t_end_Ma )
        ->group( "Time Discretization" )
        ->description( "Final time in Ma." );
    add_option_with_default( app, "--dt-max", parameters.time_stepping_parameters.dt_max_Ma )
        ->group( "Time Discretization" )
        ->description( "Maximum/minimum time step size in Ma" );
    add_option_with_default( app, "--dt-min", parameters.time_stepping_parameters.dt_min_Ma )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--max-timesteps", parameters.time_stepping_parameters.max_timesteps )
        ->group( "Time Discretization" )
        ->description(
            "Simulation aborts when this time step index is reached. "
            "If a checkpoint is loaded, the simulation will start at the next step after the loaded checkpoint. "
            "This means the number of time steps executed might be smaller than what is passed in here." );
    add_option_with_default( app, "--initial-ramp-steps", parameters.time_stepping_parameters.initial_dt_ramp_steps )
        ->group( "Time Discretization" )
        ->description( "Amount of ramp-up timesteps at the beginning of the run." );
    add_option_with_default( app, "--picard-iterations", parameters.time_stepping_parameters.picard_iterations )
        ->group( "Time Discretization" )
        ->description( "Number of Picard (fixed-point) iterations per timestep. "
                       "Each iteration re-solves Stokes and energy from the same starting temperature. "
                       "Default: 1 (no iteration, current behavior)." );

    /////////////////////
    /// Stokes solver ///
    /////////////////////

    // --low-mem preset flag (its effect is applied up front in the pre-scan near the top
    // of this function, before the option defaults below are captured).
    add_flag_with_default( app, "--low-mem", low_mem )
        ->group( "Stokes Solver" )
        ->description(
            "Low-memory solver preset. Equivalent to --stokes-float-krylov-basis "
            "--energy-float-krylov-basis --stokes-krylov-restart 5 --energy-krylov-restart 5 "
            "--stokes-viscous-pc-num-smoothing-steps-prepost 1. Individual flags passed "
            "explicitly override the corresponding preset value." );

    add_option_with_default( app, "--stokes-krylov-restart", parameters.stokes_solver_parameters.krylov_restart )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-max-iterations", parameters.stokes_solver_parameters.krylov_max_iterations )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-relative-tolerance", parameters.stokes_solver_parameters.krylov_relative_tolerance )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-absolute-tolerance", parameters.stokes_solver_parameters.krylov_absolute_tolerance )
        ->group( "Stokes Solver" );
    add_flag_with_default(
        app, "--stokes-float-krylov-basis", parameters.stokes_solver_parameters.float_krylov_basis )
        ->group( "Stokes Solver" );
    static const std::map< std::string, MGPrecision > mg_precision_map{
        { "double", MGPrecision::DOUBLE },
        { "float", MGPrecision::FLOAT },
        { "single", MGPrecision::FLOAT },
        { "half", MGPrecision::HALF },
        { "fp16", MGPrecision::HALF },
    };
    add_option_with_default( app, "--stokes-mg-precision", parameters.stokes_solver_parameters.mg_precision )
        ->transform( CLI::CheckedTransformer( mg_precision_map, CLI::ignore_case ) )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-num-vcycles", parameters.stokes_solver_parameters.viscous_pc_num_vcycles )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-cheby-order", parameters.stokes_solver_parameters.viscous_pc_chebyshev_order )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-smoothing-steps-prepost",
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-power-iterations",
        parameters.stokes_solver_parameters.viscous_pc_num_power_iterations )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-gca", parameters.stokes_solver_parameters.gca )
        ->group( "Stokes Solver" )
        ->description( "Galerkin coarse-grid approximation mode for the viscous-block multigrid "
                       "preconditioner. 0 = disabled (default; coarse operators rediscretised), "
                       "1 = full GCA (more robust at variable viscosity), "
                       "2 = adaptive GCA (memory-saving, slightly less robust)." );
    app.add_option(
           "--stokes-viscous-pc-agglom-factors",
           parameters.stokes_solver_parameters.viscous_pc_agglom_factors,
           "Per-descent agglomeration factors for the viscous MG preconditioner. "
           "Space-separated list of length num_mg_levels-1. Example: \"2 2 1 1\". "
           "Empty (default) = classical MG with all levels on MPI_COMM_WORLD." )
        ->group( "Stokes Solver" )
        ->expected( 0, -1 )
        ->default_val( parameters.stokes_solver_parameters.viscous_pc_agglom_factors );

    /////////////////////
    /// Energy solver ///
    /////////////////////

    std::map< std::string, EnergySolverType > energy_solver_map{
        { "fct", EnergySolverType::FCT },
        { "supg", EnergySolverType::SUPG },
        { "entropy_viscosity", EnergySolverType::ENTROPY_VISCOSITY },
        { "ev", EnergySolverType::ENTROPY_VISCOSITY },
    };

    add_option_with_default( app, "--energy-solver", parameters.energy_solver_parameters.energy_solver )
        ->transform( CLI::CheckedTransformer( energy_solver_map, CLI::ignore_case ) )
        ->default_val( "ev" )
        ->group( "Energy Solver" )
        ->description( "'fct': Explicit FCT advection-diffusion (default). "
                       "'supg': Implicit SUPG advection-diffusion with FGMRES solver." );

    add_option_with_default( app, "--energy-krylov-restart", parameters.energy_solver_parameters.krylov_restart )
        ->group( "Energy Solver" );
    add_flag_with_default(
        app, "--energy-float-krylov-basis", parameters.energy_solver_parameters.float_krylov_basis )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-max-iterations", parameters.energy_solver_parameters.krylov_max_iterations )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-relative-tolerance", parameters.energy_solver_parameters.krylov_relative_tolerance )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-absolute-tolerance", parameters.energy_solver_parameters.krylov_absolute_tolerance )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--energy-substeps", parameters.energy_solver_parameters.energy_substeps )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--ev-alpha-max", parameters.energy_solver_parameters.ev_alpha_max )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--ev-alpha-E", parameters.energy_solver_parameters.ev_alpha_E )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--ev-dump-nu-h", parameters.energy_solver_parameters.ev_dump_nu_h )
        ->group( "Energy Solver" );

    //////////////////////
    /// Input / output ///
    //////////////////////

    add_option_with_default( app, "--outdir", parameters.io_parameters.outdir )->group( "I/O" );
    add_flag_with_default( app, "--outdir-overwrite", parameters.io_parameters.overwrite )->group( "I/O" );
    add_option_with_default( app, "--output-pressure", parameters.io_parameters.output_pressure )->group( "I/O" );

    // Checkpoint loading
    add_option_with_default( app, "--load-checkpoint", parameters.io_parameters.load_checkpoint )
        ->group( "I/O" )
        ->description( "Starting from checkpoint" );
    add_option_with_default( app, "--checkpoint-dir", parameters.io_parameters.checkpoint_dir )->group( "I/O" );
    add_option_with_default( app, "--checkpoint-step", parameters.io_parameters.checkpoint_step )->group( "I/O" );

    add_option_with_default( app, "--output-frequency", parameters.io_parameters.output_frequency )
        ->group( "I/O" )
        ->description( "Write XDMF and radial profile output every N timesteps. Default: 1 (every timestep)." );

    add_flag_with_default( app, "--no-xdmf", parameters.io_parameters.no_xdmf )
        ->group( "I/O" )
        ->description( "Disable XDMF output." );

    add_flag_with_default( app, "--no-radial-profiles", parameters.io_parameters.no_radial_profiles )
        ->group( "I/O" )
        ->description( "Disable radial profile output." );

    try
    {
        app.parse( argc, argv );
    }
    catch ( const CLI::ParseError& e )
    {
        app.exit( e );
        if ( e.get_exit_code() == static_cast< int >( CLI::ExitCodes::Success ) )
        {
            return { CLIHelp{} };
        }
        return { "CLI parse error" };
    }

    // Cross-flag validation for anisotropic refinement.  The radial diamond
    // level at MG level L is (L + radial_extra_levels); we need that level to
    // be non-negative (otherwise (1 << rad_level) is UB) and to be at least the
    // radial subdomain refinement level so each subdomain holds >= 1 cell.
    {
        const auto& mp            = parameters.mesh_parameters;
        const int   mesh_min      = mp.refinement_level_mesh_min;
        const int   extra         = mp.radial_extra_levels;
        const int   lat_sdr_eff   = ( mp.lat_sdr >= 0 ) ? mp.lat_sdr : mp.refinement_level_subdomains;
        const int   rad_sdr_eff   = ( mp.rad_sdr >= 0 ) ? mp.rad_sdr : mp.refinement_level_subdomains;
        const int   rad_level_min = mesh_min + extra;

        if ( rad_level_min < 0 )
        {
            return {
                "Invalid refinement: refinement_level_mesh_min (" + std::to_string( mesh_min ) +
                ") + radial_extra_levels (" + std::to_string( extra ) + ") = " + std::to_string( rad_level_min ) +
                " is negative.  Radial mesh refinement level must be >= 0 at the coarsest "
                "MG level." };
        }
        if ( mesh_min < lat_sdr_eff )
        {
            return {
                "Invalid refinement: refinement_level_mesh_min (" + std::to_string( mesh_min ) +
                ") is less than the effective lateral subdomain refinement level (" + std::to_string( lat_sdr_eff ) +
                ").  Each lateral subdomain needs at least one cell at the coarsest MG level." };
        }
        if ( rad_level_min < rad_sdr_eff )
        {
            return {
                "Invalid refinement: refinement_level_mesh_min + radial_extra_levels (" +
                std::to_string( rad_level_min ) + ") is less than the effective radial subdomain refinement level (" +
                std::to_string( rad_sdr_eff ) +
                ").  Each radial subdomain needs at least one cell at the coarsest MG level. "
                "Consider lowering --rad-sdr or raising --radial-extra-levels." };
        }
    }

    // Nondimensionalise all relevant input parameters
    nondimensionalise( parameters );

    // Determine timestep_initial from checkpointing parameters
    if ( parameters.io_parameters.load_checkpoint )
    {
        parameters.time_stepping_parameters.timestep_initial = parameters.io_parameters.checkpoint_step;
    }

    util::logroot << "=========================================\n";
    util::logroot << "     Starting mantle circulation app     \n";
    util::logroot << "     Run with -h or --help for help      \n";
    util::logroot << "=========================================\n";

    util::print_general_info( argc, argv, util::logroot );
    util::print_cli_summary( app, util::logroot );
    util::logroot << std::endl;

    if ( !parameters.output_config_file.empty() )
    {
        util::logroot << "Writing config file to " << parameters.output_config_file << " and exiting." << std::endl;
        std::ofstream config_file( parameters.output_config_file );
        config_file << app.config_to_str( true, true );
    }

    return { parameters };
}

}; // namespace terra::mantlecirculation
