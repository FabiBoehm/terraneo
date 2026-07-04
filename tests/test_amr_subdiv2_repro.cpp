// Pure-forest diagnostic: does refine/balance on a uniform S_rad=2 forest produce out-of-range leaves or
// duplicate finest anchors (the partition invariant)? Isolates where the bad leaf (anchor.x=8) is created.
#include "terra/grid/shell/adaptive_forest.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <array>
#include <cstdio>
#include <map>

using namespace terra;
using namespace terra::grid::shell::amr;

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const int S_lat = 2, S_rad = 2, NU = 2;

        auto check = [&]( AdaptiveForest& f, const char* stage ) {
            auto                            lv = f.leaves();
            std::map< std::array< int, 4 >, int > seen;
            int                             bad_range = 0, dup = 0;
            for ( const auto& l : lv )
            {
                const int nx = S_lat << l.subdivision, nr = S_rad << l.subdivision;
                if ( l.id.subdomain_x() < 0 || l.id.subdomain_x() >= nx || l.id.subdomain_y() < 0 ||
                     l.id.subdomain_y() >= nx || l.id.subdomain_r() < 0 || l.id.subdomain_r() >= nr )
                {
                    if ( bad_range < 8 )
                        std::printf( "    BAD RANGE [%s]: d=%d x=%d y=%d r=%d subdiv=%d (nx=%d nr=%d)\n", stage,
                                     l.id.diamond_id(), l.id.subdomain_x(), l.id.subdomain_y(), l.id.subdomain_r(),
                                     l.subdivision, nx, nr );
                    bad_range++;
                }
                const auto           a = f.finest_anchor( l );
                std::array< int, 4 > k{ a.diamond_id(), a.subdomain_x(), a.subdomain_y(), a.subdomain_r() };
                if ( !seen.emplace( k, 1 ).second )
                    dup++;
            }
            std::printf( "  [%s] %zu leaves  valid=%d  bad_range=%d  dup_anchors=%d\n", stage, lv.size(),
                         (int) f.validate(), bad_range, dup );
            std::fflush( stdout );
        };

        AdaptiveForest fu( NU, S_lat, S_rad );
        check( fu, "subdiv0" );
        auto all = fu.leaves(); // COPY (refine mutates leaves_)
        fu.refine( all );
        check( fu, "after refine (copied)" );
        fu.balance_2to1();
        check( fu, "after balance" );
    }
    return 0;
}
