// Compile-only instantiation check for multigrid_mp.hpp: forces a hierarchy with a
// genuine precision change across a level (fine float over coarse double).
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid_mp.hpp"
#include "linalg/solvers/pcg.hpp"

using namespace terra;
using namespace terra::linalg::solvers;

template < typename S >
using Lap = fe::wedge::operators::shell::LaplaceSimple< S >;
template < typename S >
using Res = fe::wedge::operators::shell::RestrictionConstant< S >;
template < typename S >
using Pro = fe::wedge::operators::shell::ProlongationConstant< S >;

// Coarsest level in double.
using CoarseD = MGCoarseLevel< Lap< double >, PCG< Lap< double > > >;

// Uniform-precision level: float over float (no conversion, kSamePrec == true).
using LevelFF = MGLevel< Lap< float >, Jacobi< Lap< float > >, Res< float >, Pro< float >,
                         linalg::VectorQ1Scalar< float > >;

// Mixed level: this level in float, the coarser one in double -> conversion path.
using LevelFD = MGLevel< Lap< float >, Jacobi< Lap< float > >, Res< float >, Pro< float >,
                         linalg::VectorQ1Scalar< double > >;

// Mixed the other way: double fine over float coarse.
using LevelDF = MGLevel< Lap< double >, Jacobi< Lap< double > >, Res< double >, Pro< double >,
                         linalg::VectorQ1Scalar< float > >;

static_assert( LevelFF::kSamePrec, "float-over-float must take the no-conversion path" );
static_assert( !LevelFD::kSamePrec, "float-over-double must take the conversion path" );
static_assert( !LevelDF::kSamePrec, "double-over-float must take the conversion path" );

// Force member instantiation of cycle() on every variant.
void force( LevelFF& a, LevelFD& b, LevelDF& c, CoarseD& d,
            linalg::VectorQ1Scalar< float >& xf, const linalg::VectorQ1Scalar< float >& bf,
            linalg::VectorQ1Scalar< double >& xd, const linalg::VectorQ1Scalar< double >& bd )
{
    a.cycle( xf, bf );
    b.cycle( xf, bf );
    c.cycle( xd, bd );
    d.cycle( xd, bd );
    (void) a.precision();
}

// Force the driver, in both precisions.
void force_driver( MultigridMP< Lap< double > >& md, Lap< double >& Ad,
                   linalg::VectorQ1Scalar< double >& xd, const linalg::VectorQ1Scalar< double >& bd,
                   MultigridMP< Lap< float > >& mf, Lap< float >& Af,
                   linalg::VectorQ1Scalar< float >& xf, const linalg::VectorQ1Scalar< float >& bf )
{
    md.solve_impl( Ad, xd, bd );
    mf.solve_impl( Af, xf, bf );
}

// The driver must be usable as a preconditioner.
static_assert( SolverLike< MultigridMP< Lap< double > > >,
               "MultigridMP must satisfy SolverLike to serve as a preconditioner" );

// Precision-list parsing.
static_assert( sizeof( void* ) > 0 );
int main()
{
    const auto l = parse_mg_precision_list( "d,d,f,f", 6 );
    if ( !l || l->size() != 6 ) return 1;
    if ( ( *l )[0] != MGPrec::Double || ( *l )[2] != MGPrec::Float || ( *l )[5] != MGPrec::Float ) return 2;
    const auto one = parse_mg_precision_list( "bf16", 4 );            // broadcast
    if ( !one || ( *one )[3] != MGPrec::BHalf ) return 3;
    if ( parse_mg_precision_list( "nonsense", 2 ) ) return 4;         // rejected
    if ( mg_prec_bytes( MGPrec::Double ) != 8 || mg_prec_bytes( MGPrec::Float ) != 4 ) return 5;
    return 0;
}
