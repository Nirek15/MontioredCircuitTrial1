using ITensors, ITensorMPS
import Pkg; Pkg.add("PastaQ") 
using PastaQ
import PastaQ: gate
using ITensorUnicodePlots
using Statistics
import Pkg; Pkg.add("Plots")
using Plots
using Random

ITensors.op(::OpName"ProjectUp",::SiteType"S=1/2") = [1 0; 0 0]
ITensors.op(::OpName"ProjectDown",::SiteType"S=1/2") = [0 0; 0 1]


function VonNeumannEntropy(MPSSys::MPS)
    N = length(MPSSys)
    SubSysCutoff = Int(round(N/2))
    orthogonalize!(MPSSys, SubSysCutoff)
    U,S,V = svd(MPSSys[SubSysCutoff], (linkinds(MPSSys, SubSysCutoff-1)..., siteinds(MPSSys, SubSysCutoff)...), cutoff=tol)

    entropy = 0
    for n=1:ITensors.dim(S,1)
        pn = (S[n,n])^2
        entropy -= pn * log(pn)
    end
    return entropy
end

function CreateCircuit(depth::Int, N::Int)
    Circuit = []
    for i in 1:depth
        if isodd(i)
            oddeventemplayer = []
            for j in 1:2:N-1
                push!(oddeventemplayer, ("RandomUnitary", j, j+1))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 2:2:N-2
                push!(evenoddtemplayer, ("RandomUnitary", j, j+1))
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end

function PerformMeasurement!(MPSSys::MPS, MeasurementSites::Vector)
    for site in MeasurementSites
        
        orthogonalize!(MPSSys, site)
        # Create a density matrix for the qubit we are measuring
        MPSMeasuring = MPSSys[site]
        SiteDensityMatrix = prime(MPSMeasuring, tags="Site") * dag(MPSMeasuring)
        
        # Grab probabilities from the diagonal of the density matrix
        probabilities = real.(diag(SiteDensityMatrix))

        # Computer decides if it gets projected up or down based on probability of down
        if rand() < probabilities[1]
            ProjectUpMPO = MPO("ProjectUp", site)
            newMPSSite = ProjectUpMPO * MPSSys[site]
            noprime!(newMPSSite)
            MPSSys[site] = newMPSSite
        else 
            ProjectDownMPO = MPO("ProjectDown", site)
            newMPSSite = ProjectDownMPO * MPSSys[site]
            noprime!(newMPSSite)
            MPSSys[site] = newMPSSite
        end
    end
    return normalize!(MPSSys)
end

function RunMonitoredCircuit(circuit::Vector, p::Float64)

    # Number of Qubits
    N = nqubits(circuit)

    # Create Site
    s = siteinds("S=1/2", N)

    # Produce MPS state
    MPSSys = MPS(s, n -> iseven(n) ? "↓" : "↑")

    # Apply gates and measurements
    for layer in circuit
        # Applying Gates
        MPSSys = apply(ITensors.ops(layer,s), MPSSys, cutoff=tol)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 

        MeasuringSites = []

        for i in 1:N
            if rand() < p
                push!(MeasuringSites,i)
            end
        end

        MPSSys = PerformMeasurement!(MPSSys, MeasuringSites)

    end
    return MPSSys, VonNeumannEntropy(MPSSys)
end

let 
    # Tolerance
    global tol = 1e-150
    # Measurement probabilities
    global problist = [0.1] #LinRange(0.1, 0.2, 2)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # Array of System Sizes
    global SystemSizes = [6]


    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 2 * N
        # Number of trials
        ntrials = 1
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        
        # Entropy fluctuating at some points (Assuming this is because im only running each circuit once) (still fluctuates when averaging)
        for prob in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = [] 
            @time for _ in 1:ntrials   
                # Create Circuit of size N
                Circuit = CreateCircuit(CircuitDepth, N)
                MPSSys, entropy = RunMonitoredCircuit(Circuit, prob)
                push!(entropytrialarray, entropy)
            end
            push!(AvgEntropyArray, mean(entropytrialarray))
        end
        
        # Printing Entropy
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        
        # Bring the entropy array to the array for each system size
        push!(BigEntropyArray, AvgEntropyArray)
    end
end

