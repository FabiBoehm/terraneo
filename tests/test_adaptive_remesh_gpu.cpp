// Dynamic-remeshing field transfer: correctness of the forest-to-forest remap (adaptive_remesh.hpp).
//
// The old and new meshes are built at the SAME intra-block LDR but on DIFFERENT forests (some blocks Refined,
// some Coarsened, some Same). The definitive correctness bar is FINEST-FRAME AFFINE REPRODUCTION: a field that
// is affine in finest-frame coordinates,  T = A*fx + B*fy + C*fr + D,  must be reproduced to MACHINE ZERO by
// every transfer rule -- copy (Same), trilinear prolong (Refined), and injection (Coarsened) -- and must be
// preserved by the post-transfer hanging-node reconciliation (an affine field satisfies the 2:1 constraint
// exactly). A nonzero result exposes a mis-mapped block or a wrong interpolation weight.
//
// Also exercises rebuild_forest (indicator -> global decision + margin + 2:1 balance) and the plan classifier.
//
// Needs a GPU (the field vectors live in device memory) -- run on an H100 node.

#include "terra/grid/shell/adaptive_remesh.hpp"

#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/adaptive_distribute.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using namespace terra;
using grid::Grid4DDataScalar;
using grid::NodeOwnershipFlag;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using namespace terra::grid::shell::amr;

static int g_failures = 0, g_checks = 0;
#define CHECK( cond )                                                       \
    do                                                                      \
    {                                                                       \
        ++g_checks;                                                         \
        if ( !( cond ) )                                                    \
        {                                                                   \
            std::printf( "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond ); \
            ++g_failures;                                                   \
        }                                                                   \
    } while ( 0 )

static constexpr int    M = 4, S_LAT = 2, S_RAD = 2, LDR = 2;
static constexpr double AA = 0.37, BB = -0.51, CC = 0.29, DD = 0.13; // affine coeffs (finest-frame)

static std::vector< double > radii_nodes()
{
    return grid::shell::uniform_shell_radii< double >( 0.5, 1.0, S_RAD * ( 1 << LDR ) + 1 );
}

static DistributedAdaptiveMesh make_mesh( const AdaptiveForest& f )
{
    return build_distributed_adaptive_mesh( MPI_COMM_WORLD, LDR, radii_nodes(), f,
                                            grid::shell::subdomain_to_rank_by_diamond );
}

// finest-frame coordinate of local node (s,i,j,r): anchor + index * span/(n-1).
template < typename V >
static void fill_affine( const DistributedAdaptiveMesh& mesh, V& field, double a, double b, double c, double d,
                         int comp )
{
    const int nx = mesh.nx, ny = mesh.ny, nr = mesh.nr, MM = mesh.domain.max_subdivision();
    auto      hv = Kokkos::create_mirror_view( field.grid_data().comp_[comp] );
    const int nsub = static_cast< int >( mesh.domain.subdomains().size() );
    for ( int s = 0; s < nsub; ++s )
    {
        const auto& info = mesh.domain.subdomain_info_from_local_idx( s );
        const int   sub  = mesh.domain.subdivision_of( info );
        const long  span = 1L << ( MM - sub );
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < ny; ++j )
                for ( int r = 0; r < nr; ++r )
                {
                    const double fx = info.subdomain_x() + (double) i * span / ( nx - 1 );
                    const double fy = info.subdomain_y() + (double) j * span / ( ny - 1 );
                    const double fr = info.subdomain_r() + (double) r * span / ( nr - 1 );
                    hv( s, i, j, r ) = a * fx + b * fy + c * fr + d;
                }
    }
    Kokkos::deep_copy( field.grid_data().comp_[comp], hv );
}

// max | field - affine(a,b,c,d) | over all local nodes of one component.
template < typename V >
static double affine_error( const DistributedAdaptiveMesh& mesh, const V& field, double a, double b, double c,
                            double d, int comp )
{
    const int nx = mesh.nx, ny = mesh.ny, nr = mesh.nr, MM = mesh.domain.max_subdivision();
    auto      hv = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, field.grid_data().comp_[comp] );
    const int nsub = static_cast< int >( mesh.domain.subdomains().size() );
    double    e = 0.0;
    for ( int s = 0; s < nsub; ++s )
    {
        const auto& info = mesh.domain.subdomain_info_from_local_idx( s );
        const int   sub  = mesh.domain.subdivision_of( info );
        const long  span = 1L << ( MM - sub );
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < ny; ++j )
                for ( int r = 0; r < nr; ++r )
                {
                    const double fx  = info.subdomain_x() + (double) i * span / ( nx - 1 );
                    const double fy  = info.subdomain_y() + (double) j * span / ( ny - 1 );
                    const double fr  = info.subdomain_r() + (double) r * span / ( nr - 1 );
                    const double ref = a * fx + b * fy + c * fr + d;
                    e                = std::max( e, std::fabs( (double) hv( s, i, j, r ) - ref ) );
                }
    }
    return e;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const int rank = mpi::rank( MPI_COMM_WORLD );

        // ---- OLD forest: uniform subdivision 2. -------------------------------------------------------
        AdaptiveForest fold( M, S_LAT, S_RAD );
        for ( int lvl = 0; lvl < 2; ++lvl )
        {
            fold.refine( fold.leaves() );
            fold.balance_2to1();
        }

        // ---- NEW forest: Refine a spread of blocks (-> Refined) and collapse a full sibling group
        //      (-> Coarsened); everything else stays (-> Same). Then 2:1-balance. ------------------------
        AdaptiveForest fnew = fold;
        {
            std::vector< ForestLeaf > ref;
            const auto&               lv = fnew.leaves();
            const std::size_t         stride = std::max< std::size_t >( 1, lv.size() / 8 );
            for ( std::size_t i = 0; i < lv.size(); i += stride )
                if ( lv[i].subdivision < M ) ref.push_back( lv[i] );
            fnew.refine( ref );
        }
        {
            // collapse the first complete, unrefined 8-sibling group we find.
            std::vector< ForestLeaf > co;
            const auto&               lv = fnew.leaves();
            for ( const auto& l : lv )
            {
                if ( l.subdivision != 2 ) continue;
                const ForestLeaf par = fnew.parent( l );
                const auto       ch  = fnew.children( par );
                bool             all = true;
                for ( const auto& c : ch )
                    if ( !fnew.contains( c ) ) { all = false; break; }
                if ( all ) { co.push_back( ch[0] ); break; }
            }
            if ( !co.empty() ) fnew.coarsen( co );
        }
        fnew.balance_2to1();
        CHECK( fold.validate() );
        CHECK( fnew.validate() );

        DistributedAdaptiveMesh old_mesh = make_mesh( fold );
        DistributedAdaptiveMesh new_mesh = make_mesh( fnew );
        auto                    old_mask = adaptive_ownership_mask( old_mesh );
        auto                    new_mask = adaptive_ownership_mask( new_mesh );

        // ---- classify + report relation counts --------------------------------------------------------
        const RemeshPlan plan = plan_remesh( old_mesh, new_mesh );
        int              nsame = 0, nref = 0, ncoa = 0, nbad = 0;
        for ( const auto& e : plan.blocks )
        {
            if ( e.relation == BlockRelation::Same ) ++nsame;
            else if ( e.relation == BlockRelation::Refined ) ++nref;
            else ++ncoa;
            if ( e.old_local < 0 && e.old_children.empty() ) ++nbad; // an unmapped new block
        }
        if ( rank == 0 )
            std::printf( "  plan: old blocks %d, new blocks %zu  |  Same %d  Refined %d  Coarsened %d  unmapped %d\n",
                         (int) old_mesh.domain.subdomains().size(), plan.blocks.size(), nsame, nref, ncoa, nbad );
        CHECK( nbad == 0 );          // every new block must map to something
        CHECK( nref > 0 );           // the test constructed refinements
        CHECK( ncoa > 0 );           // ... and a coarsening
        CHECK( nsame > 0 );          // ... and unchanged blocks

        // ---- SCALAR transfer: affine field must reproduce to machine zero ------------------------------
        VectorQ1Scalar< double > T_old( "T_old", old_mesh.domain, old_mask );
        VectorQ1Scalar< double > T_new( "T_new", new_mesh.domain, new_mask );
        // fill_affine writes comp_[0]; VectorQ1Scalar's grid_data() is a single scalar view, so wrap access:
        {
            const int nx = old_mesh.nx, ny = old_mesh.ny, nr = old_mesh.nr, MM = old_mesh.domain.max_subdivision();
            auto      hv = Kokkos::create_mirror_view( T_old.grid_data() );
            const int nsub = static_cast< int >( old_mesh.domain.subdomains().size() );
            for ( int s = 0; s < nsub; ++s )
            {
                const auto& info = old_mesh.domain.subdomain_info_from_local_idx( s );
                const int   sub  = old_mesh.domain.subdivision_of( info );
                const long  span = 1L << ( MM - sub );
                for ( int i = 0; i < nx; ++i )
                    for ( int j = 0; j < ny; ++j )
                        for ( int r = 0; r < nr; ++r )
                        {
                            const double fx = info.subdomain_x() + (double) i * span / ( nx - 1 );
                            const double fy = info.subdomain_y() + (double) j * span / ( ny - 1 );
                            const double fr = info.subdomain_r() + (double) r * span / ( nr - 1 );
                            hv( s, i, j, r ) = AA * fx + BB * fy + CC * fr + DD;
                        }
            }
            Kokkos::deep_copy( T_old.grid_data(), hv );
        }

        FieldRemapper< double > rm( old_mesh, new_mesh, plan );
        rm.remap( T_old, T_new, TransferPolicy{ TransferKind::Geometric } );

        // reuse the vector affine_error on the scalar view via a temporary single-component wrapper.
        double t_err = 0.0;
        {
            const int nx = new_mesh.nx, ny = new_mesh.ny, nr = new_mesh.nr, MM = new_mesh.domain.max_subdivision();
            auto      hv = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, T_new.grid_data() );
            const int nsub = static_cast< int >( new_mesh.domain.subdomains().size() );
            for ( int s = 0; s < nsub; ++s )
            {
                const auto& info = new_mesh.domain.subdomain_info_from_local_idx( s );
                const int   sub  = new_mesh.domain.subdivision_of( info );
                const long  span = 1L << ( MM - sub );
                for ( int i = 0; i < nx; ++i )
                    for ( int j = 0; j < ny; ++j )
                        for ( int r = 0; r < nr; ++r )
                        {
                            const double fx  = info.subdomain_x() + (double) i * span / ( nx - 1 );
                            const double fy  = info.subdomain_y() + (double) j * span / ( ny - 1 );
                            const double fr  = info.subdomain_r() + (double) r * span / ( nr - 1 );
                            const double ref = AA * fx + BB * fy + CC * fr + DD;
                            t_err            = std::max( t_err, std::fabs( (double) hv( s, i, j, r ) - ref ) );
                        }
            }
        }
        if ( rank == 0 ) std::printf( "  scalar T: max affine-reproduction error = %.3e\n", t_err );
        CHECK( t_err < 1e-10 );

        // ---- VECTOR transfer: each component a different affine field ----------------------------------
        VectorQ1Vec< double, 3 > u_old( "u_old", old_mesh.domain, old_mask );
        VectorQ1Vec< double, 3 > u_new( "u_new", new_mesh.domain, new_mask );
        const double             cf[3][4] = { { AA, BB, CC, DD }, { BB, CC, AA, -DD }, { CC, AA, BB, 0.2 } };
        for ( int c = 0; c < 3; ++c ) fill_affine( old_mesh, u_old, cf[c][0], cf[c][1], cf[c][2], cf[c][3], c );

        rm.remap< 3 >( u_old, u_new, TransferPolicy{ TransferKind::Geometric } );

        double u_err = 0.0;
        for ( int c = 0; c < 3; ++c )
            u_err = std::max( u_err, affine_error( new_mesh, u_new, cf[c][0], cf[c][1], cf[c][2], cf[c][3], c ) );
        if ( rank == 0 ) std::printf( "  vector u: max affine-reproduction error = %.3e\n", u_err );
        CHECK( u_err < 1e-10 );

        // ---- rebuild_forest: a peaked indicator must produce a valid, refined forest -------------------
        {
            std::vector< double > ind( old_mesh.domain.subdomains().size(), 0.0 );
            if ( !ind.empty() ) ind[0] = 1.0; // peak on block 0
            RemeshOptions o;
            o.remesh_every  = 1;
            o.max_subdiv    = M;
            o.margin_rings  = 1;
            o.refine_frac   = 0.5;
            AdaptiveForest rebuilt = rebuild_forest( fold, old_mesh, ind, MPI_COMM_WORLD, o );
            CHECK( rebuilt.validate() );
            if ( rank == 0 )
                std::printf( "  rebuild_forest: %zu leaves -> %zu leaves (peaked indicator)\n", fold.leaves().size(),
                             rebuilt.leaves().size() );
            CHECK( rebuilt.leaves().size() >= fold.leaves().size() ); // refining a region grows the forest
        }
    }

    const bool ok = ( g_failures == 0 );
    if ( mpi::rank( MPI_COMM_WORLD ) == 0 )
        std::printf( "test_adaptive_remesh_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
