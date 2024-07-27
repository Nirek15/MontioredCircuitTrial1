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
using JLD2


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

function ITensors.op(::OpName"ChargeConservingUnitaryGate2", ::SiteType"S=1/2", s1::Index, s2::Index)

    phi0 = rand()*2*pi
    phi3 = rand()*2*pi
    TwoTwoHaarRandom = RandomHaar(Val{2}())

    Matrix = [
        exp(1im * phi0)        0                     0               0;
        0              TwoTwoHaarRandom[1]  TwoTwoHaarRandom[3]      0;
        0              TwoTwoHaarRandom[2]  TwoTwoHaarRandom[4]      0;
        0                      0                     0         exp(1im*phi3)]

    return itensor(Matrix, s2', s1', s2, s1)
end


function RandomHaar(::Val{n}) where n
    M = randn(SMatrix{n,n,ComplexF64}) 
    Q, R = qr(M)
    Lambda = diagm([x/abs(x) for x in diag(R)])
    return Q * Lambda
end

function op(::OpName"HaarRandomGate",::SiteType"S=1/2",s1::Index,s2::Index)
    h = RandomHaar(Val{4}())
    return itensor(Matrix(h),s2',s1',s2,s1)
  end


function addzero()
    global zerocount += 1
end

function addone()
    global onecount += 1
end

function MPOEntanglementEntropy(MPOSys::MPO, s)
    N = length(MPOSys)
    SubSysCutoff = Int(round(N/2)) + 1

    # Trace MPO Tensors
    for i in SubSysCutoff:N
        MPOSys[i] *= delta(s[i], s[i]')
    end

    # Hold value of contracted Tensors
    TempTensor = ITensor(1)
    for j in SubSysCutoff:N
        TempTensor *= MPOSys[j]
    end

    MPOSys[SubSysCutoff - 1] *= TempTensor

    ReducedMPO = MPO(SubSysCutoff - 1)

    for k in 1:(SubSysCutoff - 1)
        ReducedMPO[k] = MPOSys[k]
    end

    SecondRenyiEntropy = tr(apply(ReducedMPO,ReducedMPO))

    return SecondRenyiEntropy
end

function CreateCircuit(depth::Int, N::Int)
    Circuit = []
    for i in 1:depth
        if isodd(i)
            oddeventemplayer = []
            for j in 1:2:N-1
                push!(oddeventemplayer, ("ChargeConservingUnitaryGate2", j, j+1))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 2:2:N-2
                push!(evenoddtemplayer, ("ChargeConservingUnitaryGate2", j, j+1))
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end


function PerformMeasurement!(MPOSys::MPO, MeasurementSites::Vector, s)
    for site in MeasurementSites
        normalize!(MPOSys)
        orthogonalize!(MPOSys, site)
        # Create a density matrix for the qubit we are measuring
        MPOMeasuring = MPOSys[site]

        # Redo This
        SiteDensityMatrix = prime(MPSMeasuring, s) * dag(MPSMeasuring)
        DensityMatrix = [SiteDensityMatrix[1] SiteDensityMatrix[2]; SiteDensityMatrix[3] SiteDensityMatrix[4]]

        # Computer decides if it gets projected up or down based on probability of down
        if rand() < something #real.(DensityMatrix[1])
            Projection = [("ProjUp", site)]
            addzero()
        else 
            Projection = [("ProjDn", site)]
            addone()
        end

        MPOSys = apply(ITensors.ops(Projection, s), MPOSys, cutoff=tol, apply_dag=true)

    end
    noprime!(MPOSys)
    return normalize!(MPOSys)
end

function RunMonitoredCircuit(circuit::Vector, p::Float64)

    # Number of Qubits
    N = nqubits(circuit)

    # Create Site
    s = siteinds("S=1/2", N)

    # Produce MPS state
    MPOSys = MPO(s, n -> iseven(n) ? "↓" : "↑")

    # Apply gates and measurements
    for layer in circuit

        # Applying Gates
        MPOSys = apply(ITensors.ops(layer, s), MPOSys, cutoff = tol)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 

        MeasuringSites = []

        for i in 1:N
            if rand() < p
                push!(MeasuringSites,i)
            end
        end


        MPOSys = PerformMeasurement!(MPOSys, MeasuringSites, s)

    end
    return MPOSys, VonNeumannEntropy(MPOSys)
end

zerocount = 0
onecount = 0

tol = 0

let 
    # Tolerance
    global tol = 1e-150
    # Measurement probabilities
    global problist = LinRange(0, 0.5, 10)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # SE Array
    global BigSEArray = []
    # Array of System Sizes
    global SystemSizes = [8,10]
    # Variance
    global BigVarArray = []

    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 2 * N
        # Number of trials
        ntrials = 1000
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        smallSEArray = []
        SmallVarArray = []
        
        # Entropy fluctuating at some points (Assuming this is because im only running each circuit once) (still fluctuates when averaging)
        for prob in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = zeros(ntrials)
            @time @Threads.threads for i in 1:ntrials   
                # Create Circuit of size N
                Circuit = CreateCircuit(CircuitDepth, N)
                MPOSys, entropy = RunMonitoredCircuit(Circuit, prob)
                entropytrialarray[i] = entropy
            end
            push!(smallSEArray, std(entropytrialarray))
            push!(AvgEntropyArray, mean(entropytrialarray))
            push!(SmallVarArray, var(entropytrialarray))
        end
        
        
        # Printing Entropy
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        
        # Bring the entropy array to the array for each system size
        push!(BigEntropyArray, AvgEntropyArray)
        push!(BigSEArray, smallSEArray)
        push!(BigVarArray, SmallVarArray)
    end
end

save_object("L68EntanglementBipartateSingleSiteWideEntropies.jld2", BigEntropyArray)
save_object("L68EntanglementBipartateSingleSiteWideError.jld2", BigSEArray)
save_object("L68EntanglementBipartateSingleSiteWideVariance.jld2", BigVarArray)
save_object("L68EntanglementBipartateSingleSiteProbList.jld2", problist)


plot(problist, real.(BigEntropyArray[1]), label="L=8")
plot!(problist, real.(BigEntropyArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Average Entropy")
title!("Bipartate Single-Site Measurements ; ntrials = 1000")
png("Single Site Measurement, UpDn Setup, Bipartate, 1000 trials")

plot(problist, real.(BigVarArray[1]), label="L=8")
plot!(problist, real.(BigVarArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Variance in Entropy")
title!("Bipartate Single-Site Measurements Entropy Variance ; ntrials = 1000")
png("Single Site Measurement, UpDn Setup, Bipartate,Variance, 1000 trials")
