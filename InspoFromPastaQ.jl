using ITensors, ITensorMPS
import Pkg; Pkg.add("PastaQ") 
using PastaQ
import PastaQ: gate
using ITensorUnicodePlots
using Statistics
import Pkg; Pkg.add("Plots")
using Plots

# Projective Measurements onto Computational Basis
gate(::GateName"P0") = [1 0 ; 0 0]
gate(::GateName"P1") = [0 0 ; 0 1]

function gate(::GateName"ChargeConservingUnitary")
    phi0 = rand()*2*pi
    phi1 = rand()*2*pi
    phi2 = rand()*2*pi
    phi3 = rand()*2*pi
    theta = rand()*pi

    return [exp(1im*phi0) 0 0 0; 0 exp(1im*(phi1+phi2))*cos(theta) exp(1im*(phi1-phi2))*sin(theta) 0; 0 -exp(1im*(phi1-phi2))*sin(theta) exp(-1im*(phi1+phi2))*cos(theta) 0; 0 0 0 exp(1im * phi3)]
end


# Circuit Creation
function CircuitCreation(depth::Int, N::Int)
    Circuit = []
    for j in 1:depth
        OddEvenLayer = gatelayer("ChargeConservingUnitary", [(i, i+1) for i in 1:2:N-1])
        EvenOddLayer = gatelayer("ChargeConservingUnitary", [(i, i+1) for i in 2:2:N-2])
        push!(Circuit, [OddEvenLayer, EvenOddLayer])
    end
    return Circuit
end

# Performs a Projective Measurement onto Computational Basis
function Measurement!(rhoSys::MPO, site::Int)
    orthogonalize!(rhoSys, site)
    # I shouldn't have to trace out the other sites because its orthogonalized correct?
    rhoMeasuring = rhoSys[site]
    # Grab probabilities from the diagonal
    probabilities = real.(diag(rhoMeasuring))
    # Computer randomly decides if it gets projected up or down (based on probability of down)
    if rand() < probabilities[1]
        rhoSys = runcircuit(rhoSys, ("P0", site), maxdim=20)
    else 
        rhoSys = runcircuit(rhoSys, ("P1", site), maxdim=20)
    end
    return normalize!(rhoSys)

    # Do I need to normalize after measurement??? I would think so but I remember you saying about not normallizing 
    # and thinking of it as a noise channel
    # Without normalizing entropy increases, it shouldn't
end

function RunMonitoredCircuit(circuit::Vector, p::Float64)

    # Number of Qubits
    N = nqubits(circuit)

    # Produce MPS state then take it's outer product to get MPO state
    MPSSys = productstate(N)
    rhoSys = outer(MPSSys', MPSSys)

    # Apply gates and measurements
    for layer in circuit
        # Applying Gates
        rhoSys = runcircuit(rhoSys, layer, cutoff=tol, maxdim=20)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 
        for i in 1:N
            if rand() < p
                rhoSys = Measurement!(rhoSys, i)
            end
        end
    end
    return rhoSys, SecondRenyiEntropy(rhoSys)
end

# Help from Forum 
function MPOTraceOut(rhoSys::MPO)
    N = length(rhoSys)
    L = tr(rhoSys[1])
    subsyscutoff = Int(round(N/2))
    for i in 2:subsyscutoff
        L *= rhoSys[i]
        L = tr(L)
    end

    rhoSys[subsyscutoff + 1] *= L
    rhoA = MPO(N-subsyscutoff)

    for i in (subsyscutoff + 1):N
        rhoA[i-subsyscutoff] = copy(rhoSys[i])
    end
    return rhoA
end

function SecondRenyiEntropy(rhoSys::MPO)
    # Orthoganlize in the center???
    rhoA = MPOTraceOut(rhoSys)
    entropy = -log(tr(apply(rhoA, rhoA)))
    return entropy
end

# Setup
let 
    # Circuit Depth
    CircuitDepth = 10
    # Tolerance
    global tol = 1e-8
    # Number of trials
    ntrials = 50
    # Measurement probabilities
    global problist = LinRange(0, 0.5, 25)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # Array of System Sizes
    global SystemSizes = [8]


    for N in SystemSizes
        # Create Circuit of size N
        Circuit = CircuitCreation(CircuitDepth, N)
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        
        # Entropy fluctuating at some points (Assuming this is because im only running each circuit once) (still fluctuates when averaging)
        for p in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = [] 
            @time for _ in 1:ntrials
                rhoSys, entropy = RunMonitoredCircuit(Circuit, p)
                push!(entropytrialarray, entropy)
            end
            push!(AvgEntropyArray, mean(entropytrialarray))
        end
        
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        
        push!(BigEntropyArray, AvgEntropyArray)
    end
end

println(length(BigEntropyArray))

plot(problist, real.(BigEntropyArray))
scatter!(problist, real.(BigEntropyArray))
xlabel!("p")
ylabel!("Average Entropy")