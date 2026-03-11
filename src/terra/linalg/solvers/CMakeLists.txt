target_sources(terra
        PRIVATE
        block_jacobi.hpp
        identity_solver.hpp
        diagonal_solver.hpp
        iterative_solver_info.hpp
        jacobi.hpp
        multigrid.hpp
        pbicgstab.hpp
        pcg.hpp
        pminres.hpp
        richardson.hpp
        solver.hpp)

add_subdirectory(gca)
