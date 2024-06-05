using ITensors, ITensorMPS
using PastaQ
import PastaQ: gate
using ITensorUnicodePlots
using Statistics
using Plots



function CircuitCreation(depth::Int, N::Int)
    Circuit = []
    for j in 1:depth
        OddEvenLayer = randomlayer("RandomUnitary", [(i, i+1) for i in 1:2:N-1])
        EvenOddLayer = randomlayer("RandomUnitary", [(i, i+1) for i in 2:2:N-2])
        push!(Circuit, [OddEvenLayer, EvenOddLayer])
    end
    return Circuit
end

function OrthogonalizeCheck(rhoSys::MPO, N::Int)

    testOrthoSys = MPO(N)
    for i in 1:N
        testOrthoSys[i] = copy(rhoSys[i])
    end

    println(tr(rhoSys))

    println("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

    for i in 1:10
        println(rhoSys[1][i])
        println(testOrthoSys[1][i])
        println("Breakkkkkk")
    end

    rhoA = copy(rhoSys[2])
    orthogonalize!(testOrthoSys, 2)

    println(tr(testOrthoSys))
    OrthoRhoA = copy(testOrthoSys[2])
    println("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

    for i in 1:10
        println(rhoSys[1][i])
        println(testOrthoSys[1][i])
        println("Breakkkkkk")
    end




    traceval = tr(rhoSys[1])
    rhoA *= traceval
    traceval = tr(rhoSys[4])
    traceval = tr(traceval * rhoSys[3])
    rhoA *= traceval

    orthogonalize!(testOrthoSys, 2)
    OrthoRhoA = copy(testOrthoSys[2])
    Orthotraceval = tr(testOrthoSys[1]) * tr(testOrthoSys[3]) * tr(testOrthoSys[4])
    OrthoRhoA *= Orthotraceval








    





    #println(testOrthoSys[1])
    #println(rhoSys[1])

    #testOrthoSys[1][1] = 100

    #println(testOrthoSys[1][1])
    #println(rhoSys[1][1])




    #for i in 1:N
    #    eachcol
    #end


    #println(tr(rhoSys[1]))

    #orthogonalize!(testOrthoSys, 2)
    
    #println(tr(rhoSys[1]))
    #println(tr(testOrthoSys[1]))

    #Orthotraceval = tr(testOrthoSys[1]) * tr(testOrthoSys[3]) * tr(testOrthoSys[4])
    #OrthoRhoA = copy(testOrthoSys[2])
    #OrthoRhoA *= Orthotraceval

    #println(size(traceval))
    #println(size(Orthotraceval))

    return rhoA, OrthoRhoA
end


function RunMonitoredCircuit(circuit::Vector)

    # Number of Qubits
    N = nqubits(circuit)

    # Produce MPS state then take it's outer product to get MPO state
    MPSSys = productstate(N)
    rhoSys = outer(MPSSys', MPSSys)

    # Apply gates and measurements
    for layer in circuit
        # Applying Gates
        rhoSys = runcircuit(rhoSys, layer, cutoff=tol)
    end

    RhoA, OrthoRhoA = OrthogonalizeCheck(rhoSys, N)

    return RhoA, OrthoRhoA
end

let 
# Number of Qubits
    N = 4
    # Circuit Depth
    CircuitDepth = 2
    # Tolerance
    global tol = 1e-8

    Circuit = CircuitCreation(CircuitDepth, N)
    RhoA, OrthoRhoA = RunMonitoredCircuit(Circuit)

    #println(real.(RhoA))
    #println("Break")
    #println(real.(OrthoRhoA))
    return nothing
end
