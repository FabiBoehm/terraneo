function(add_terra_performance_benchmark name)
    add_executable(${name} ${name}.cpp)
    if (MPI_FOUND)
        target_link_libraries(${name} terra MPI::MPI_CXX)
    else ()
        target_link_libraries(${name} terra)
    endif ()
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_terra_performance_benchmark(benchmark_alt)
add_terra_performance_benchmark(benchmark_alt_fe)
add_terra_performance_benchmark(benchmark_baseline)
add_terra_performance_benchmark(benchmark_operators)
# One executable per EpsDivDivKerngen sweep, so each instantiates only its own operator types.
add_terra_performance_benchmark(benchmark_epsdivdiv_precision_sweep)
add_terra_performance_benchmark(benchmark_epsdivdiv_precision_matrix)
add_terra_performance_benchmark(benchmark_epsdivdiv_precision_ablation)
add_terra_performance_benchmark(benchmark_epsdivdiv_tiles)
add_terra_performance_benchmark(benchmark_epsdivdiv_stored)
add_terra_performance_benchmark(benchmark_stokes_precision_sweep)

# Device line info on the precision sweep only, for ncu source-level profiling of the
# EpsDivDiv kernels (ncu --section SourceCounters / --import-source). Off by default:
# it is a profiling build, not a measurement build.
option(TERRA_BENCHMARK_LINEINFO "Compile benchmark_epsdivdiv_precision_sweep with device line info" OFF)
if (TERRA_BENCHMARK_LINEINFO)
    target_compile_options(benchmark_epsdivdiv_precision_sweep PRIVATE
        $<$<CXX_COMPILER_ID:NVHPC>:-gpu=lineinfo>
        $<$<CXX_COMPILER_ID:NVIDIA>:-lineinfo>)
endif ()
add_terra_performance_benchmark(benchmark_communication)

