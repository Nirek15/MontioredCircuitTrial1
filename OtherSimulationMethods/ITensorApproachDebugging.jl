using ITensors, ITensorMPS
using PastaQ
import PastaQ: gate
using ITensorUnicodePlots
using Statistics
using Plots
using Random
using LinearAlgebra
using StatsBase
using StaticArrays
import ITensors: op
using Distributions


ITensors.op(::OpName"ProjectUp",::SiteType"S=1/2") = [1 0; 0 0]
ITensors.op(::OpName"ProjectDown",::SiteType"S=1/2") = [0 0; 0 1]

# Define a custom sine distribution
struct SinDistribution <: ContinuousUnivariateDistribution
end

# Sampling method (inverse transform sampling)
function Base.rand(d::SinDistribution, rng::AbstractRNG=Random.GLOBAL_RNG)
    return acos(1 - 2 * rand(rng))
end

sin_dist = SinDistribution()

function ITensors.op(::OpName"ChargeConservingUnitaryGate", ::SiteType"S=1/2", s1::Index, s2::Index)

    phi0 = rand()*2*pi
    phi1 = rand()*2*pi
    phi2 = rand()*2*pi
    phi3 = rand()*2*pi
    theta = rand(sin_dist)

    Matrix = [
        exp(1im * phi0)  0                               0                               0;
        0               exp(1im * (phi1 + phi2))*cos(theta)  exp(1im * (phi1 - phi2))*sin(theta)  0;
        0              -exp(1im * (phi2 - phi1))*sin(theta)  exp(-1im * (phi1 + phi2))*cos(theta)  0;
        0               0                               0                               exp(1im*phi3)]

    return itensor(Matrix, s2', s1', s2, s1)
end


function rand_haar2(::Val{n}) where n
    M = randn(SMatrix{n,n,ComplexF64}) 
    q = qr(M).Q
    L = cispi.(2 .* @SVector(rand(n)))
    return q*diagm(L)
end

function op(::OpName"haar2",::SiteType"S=1/2",s1::Index,s2::Index)
    h = rand_haar2(Val{4}())
     return itensor(Matrix(h),s2',s1',s2,s1)
  end



function VonNeumannEntropy(MPSSys::MPS)
    N = length(MPSSys)
    SubSysCutoff = Int(round(N/2))
    orthogonalize!(MPSSys, SubSysCutoff)
    U,S,V = svd(MPSSys[SubSysCutoff], (linkinds(MPSSys, SubSysCutoff-1)..., siteinds(MPSSys, SubSysCutoff)...), cutoff=1e-150)

    entropy = 0
    for n in 1:ITensors.dim(S,1)
        if n == 1 && S[n,n] == 0
            continue
        else
            pn = (S[n,n])^2
            entropy -= pn * log(pn)
        end
    end
    return entropy
end

function CreateCircuit(depth::Int, N::Int)
    Circuit = []
    for i in 1:depth
        if isodd(i)
            oddeventemplayer = []
            for j in 1:2:N-1
                push!(oddeventemplayer, ("haar2", j, j+1))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 2:2:N-2
                push!(evenoddtemplayer, ("haar2", j, j+1))
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end


function PerformMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)
    for site in MeasurementSites
        
        orthogonalize!(MPSSys, site)
        # Create a density matrix for the qubit we are measuring
        MPSMeasuring = MPSSys[site]
        SiteDensityMatrix = prime(MPSMeasuring, tags="Site") * dag(MPSMeasuring)
        
        # Grab probabilities from the diagonal of the density matrix
        probabilities = real.(diag(SiteDensityMatrix))

        # Computer decides if it gets projected up or down based on probability of down
        if rand() < probabilities[1]
            os = OpSum()
            os += "ProjectUp", site
            ProjectUpMPO = MPO(os, s)
            MPSSys = apply(ProjectUpMPO, MPSSys, cutoff=tol)
        else 
            os = OpSum()
            os += "ProjectDown", site
            ProjectDownMPO = MPO(os, s)
            MPSSys = apply(ProjectDownMPO, MPSSys, cutoff=tol)
        end
    end
    noprime!(MPSSys)
    return normalize!(MPSSys)
end

function RunMonitoredCircuit(circuit::Vector, p::Float64)

    # Number of Qubits
    N = nqubits(circuit)

    # Create Site
    s = siteinds("S=1/2", N)

    # Produce MPS state
    MPSSys = MPS(s, n -> iseven(n) ? "↓" : "↑")
    LayerEntropyList = []

    # Apply gates and measurements
    
    for layer in circuit

        # Applying Gates
        MPSSys = apply(ITensors.ops(layer, s), MPSSys)
        LayerEntropy = VonNeumannEntropy(MPSSys)
        push!(LayerEntropyList, LayerEntropy)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 
        
        MeasuringSites = []

        for i in 1:N
            if layer == circuit[1] && i == 2 #rand() < p
                push!(MeasuringSites,i)
            end
        end


        #MPSSys = PerformMeasurement!(MPSSys, MeasuringSites, s)

    end
    return MPSSys, LayerEntropyList
end


let 
    # Tolerance
    global tol = 1e-8
    # Measurement probabilities
    global problist = [0.0] #LinRange(0.1, 0.8, 13)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # Array of System Sizes
    global SystemSizes = [16]

    prob = problist[1]


    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 40
        # Number of trials
        ntrials = 1
        # Average Entropy Array per System size

        Circuit = CreateCircuit(CircuitDepth, N)
        MPSSys, EntropyList = RunMonitoredCircuit(Circuit, prob)
        BigEntropyArray = EntropyList
    end
end

print(BigEntropyArray)


scatter(1:2:40, BigEntropyArray[1:2:end])
xlabel!("Circuit Depth Level")
ylabel!("Average Entropy")

plot(problist, real.(BigEntropyArray[1]), ylim=(0,5), label="L=6")
plot!(problist, real.(BigEntropyArray[2]), label="L=8")
scatter!(problist, real.(BigEntropyArray), label="")
xlabel!("p")
ylabel!("Average Entropy")

