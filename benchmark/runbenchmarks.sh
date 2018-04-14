#!/usr/bin/env sh

echo ""
echo "              Threaded               "
echo "-------------------------------------"
echo " Workers | Forward (s) | Adjoint (s) "
echo "-------------------------------------"
counter=1
while [ $counter -le `nproc` ]
do
    JULIA_NUM_THREADS=$counter julia benchmarks.jl threaded
    counter=`expr $counter + 1`
done

echo ""
echo "              Parallel               "
echo "-------------------------------------"
echo " Workers | Forward (s) | Adjoint (s) "
echo "-------------------------------------"
counter=1
while [ $counter -le `nproc` ]
do
    julia -p $counter benchmarks.jl parallel
    counter=`expr $counter + 1`
done

echo ""
echo "               Serial                "
echo "-------------------------------------"
echo " Workers | Forward (s) | Adjoint (s) "
echo "-------------------------------------"
julia benchmarks.jl
