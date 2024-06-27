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

    push!(phi0Array, phi0)
    push!(phi1Array, phi1)
    push!(phi2Array, phi2)
    push!(phi3Array, phi3)
    push!(thetaArray, theta)


    Matrix = [
        exp(1im * phi0)  0                               0                               0;
        0               exp(1im * (phi1 + phi2))*cos(theta)  exp(1im * (phi1 - phi2))*sin(theta)  0;
        0              -exp(1im * (phi2 - phi1))*sin(theta)  exp(-1im * (phi1 + phi2))*cos(theta)  0;
        0               0                               0                               exp(1im*phi3)]

    return itensor(Matrix, s2', s1', s2, s1)
end


function ITensors.op(::OpName"FixedChargeConservingUnitaryGate", ::SiteType"S=1/2", s1::Index, s2::Index)

    global GateCount += 1

    phi0 = phi0Array[GateCount]
    phi1 = phi1Array[GateCount]
    phi2 = phi2Array[GateCount]
    phi3 = phi3Array[GateCount]
    theta = thetaArray[GateCount]

    Matrix = [
        exp(1im * phi0)  0                               0                               0;
        0               exp(1im * (phi1 + phi2))*cos(theta)  exp(1im * (phi1 - phi2))*sin(theta)  0;
        0              -exp(1im * (phi2 - phi1))*sin(theta)  exp(-1im * (phi1 + phi2))*cos(theta)  0;
        0               0                               0                               exp(1im*phi3)]

    return itensor(Matrix, s2', s1', s2, s1)
end


function VonNeumannEntropy(MPSSys::MPS)
    normalize!(MPSSys)
    N = length(MPSSys)
    SubSysCutoff = Int(round(N/2))
    orthogonalize!(MPSSys, SubSysCutoff)
    U,S,V = svd(MPSSys[SubSysCutoff], (linkinds(MPSSys, SubSysCutoff-1)..., siteinds(MPSSys, SubSysCutoff)...), cutoff=1e-150)


    entropy = 0
    for n in 1:ITensors.dim(S,1)
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
                push!(oddeventemplayer, ("FixedChargeConservingUnitaryGate", j, j+1))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 2:2:N-2
                push!(evenoddtemplayer, ("FixedChargeConservingUnitaryGate", j, j+1))
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end


function PerformMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)

    ###
    ForcedLayer = []
    ###

    for site in MeasurementSites
        normalize!(MPSSys)
        orthogonalize!(MPSSys, site)
        # Create a density matrix for the qubit we are measuring
        MPSMeasuring = MPSSys[site]
        SiteDensityMatrix = prime(MPSMeasuring, s) * dag(MPSMeasuring)
        DensityMatrix = [SiteDensityMatrix[1] SiteDensityMatrix[2]; SiteDensityMatrix[3] SiteDensityMatrix[4]]

        # Computer decides if it gets projected up or down based on probability of down
        if rand() < real.(DensityMatrix[1])
            os = OpSum()
            os += "ProjectUp", site
            ProjectUpMPO = MPO(os, s)
            MPSSys = apply(ProjectUpMPO, MPSSys, cutoff=tol)
            push!(ForcedLayer, (site, 0))
        else 
            os = OpSum()
            os += "ProjectDown", site
            ProjectDownMPO = MPO(os, s)
            MPSSys = apply(ProjectDownMPO, MPSSys, cutoff=tol)
            push!(ForcedLayer, (site, 1))
        end
    end
    noprime!(MPSSys)
    push!(MeasurementTrajectoryForced, ForcedLayer)
    return normalize!(MPSSys)
end

function PerformForcedMeasurement!(MPSSys::MPS, MeasurementTuple::Vector, s)

    for mtuple in MeasurementTuple
        normalize!(MPSSys)
        orthogonalize!(MPSSys, mtuple[1])

        # Force Measurement
        if mtuple[2] == 0
            os = OpSum()
            os += "ProjectUp", mtuple[1]
            ProjectUpMPO = MPO(os, s)
            MPSSys = apply(ProjectUpMPO, MPSSys, cutoff=tol)
        else 
            os = OpSum()
            os += "ProjectDown", mtuple[1]
            ProjectDownMPO = MPO(os, s)
            MPSSys = apply(ProjectDownMPO, MPSSys, cutoff=tol)
        end
    end
    noprime!(MPSSys)
    return normalize!(MPSSys)
end

function RunMonitoredCircuit(circuit::Vector, p::Float64, ForcedTrajectory=nothing)

    # Number of Qubits
    N = nqubits(circuit)

    # Create Site
    s = siteinds("S=1/2", N)

    # Produce MPS state
    MPSSys = MPS(s, n -> iseven(n) ? "↓" : "↑")

    # Apply gates and measurements
    for (layer, i) in zip(circuit, 1:length(circuit))

        # Applying Gates
        MPSSys = apply(ITensors.ops(layer, s), MPSSys, cutoff = tol)

        if ForcedTrajectory !== nothing
            MPSSys = PerformForcedMeasurement!(MPSSys, ForcedTrajectory[i], s)
        else 

            # Possible Measurement
            # For each qubit, according to a given probability of measurement, measure it. 
            
            MeasuringSites = []
            
            for i in 1:N
                if rand() < p
                    push!(MeasuringSites,i)
                end
            end
            
            MPSSys = PerformMeasurement!(MPSSys, MeasuringSites, s)
        end
    
    end
    return MPSSys, VonNeumannEntropy(MPSSys)
end

zerocount = 0
onecount = 0
tol = 0
MeasurementTrajectoryForced = [] 

phi0Array = load_object("phi0s.jld2")
phi1Array = load_object("phi1s.jld2")
phi2Array = load_object("phi2s.jld2")
phi3Array = load_object("phi3s.jld2")
thetaArray = load_object("thetas.jld2")

GateCount = 0

let 
    # Tolerance
    #global tol = 1e-300
    # Measurement probabilities
    global problist = [0.05] #LinRange(0, 0.5, 10)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # SE Array
    global BigSEArray = []
    # Array of System Sizes
    global SystemSizes = [8]
    # Tolerances
    global ToleranceArray = 10 .^ range(-10, stop=-1, length=100) 
    # Max Bond Dimensions
    global BigDimArray = []

    for N in SystemSizes

        # Circuit Depth
        CircuitDepth = 2 * N
        # Number of trials
        ntrials = 1
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        smallSEArray = []

        # Create Circuit Once, then save and load

        
        Circuit = CreateCircuit(CircuitDepth, N)
        #save_object("ToleranceTrajectoryCircuit.jld2", Circuit)
        
        #Circuit = load_object("ToleranceTrajectoryUnitaryGates2.jld2")
        CircuitMTrajectory = load_object("ToleranceTrajectoryMTrajectory.jld2")

        @time for tolerance in ToleranceArray
            global tol = tolerance
            entropytrialarray = [] 
            SmallDimArray = []
            for _ in 1:ntrials 
                MPSSys, entropy = RunMonitoredCircuit(Circuit, problist[1], CircuitMTrajectory)
                push!(entropytrialarray, entropy)
                push!(SmallDimArray, maxlinkdim(MPSSys))
                global GateCount = 0
            end
            push!(BigEntropyArray, mean(entropytrialarray))
            push!(BigDimArray, mean(SmallDimArray))
        end

        
        #=
        # Entropy fluctuating at some points (Assuming this is because im only running each circuit once) (still fluctuates when averaging)
        for prob in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = [] 
            @time for _ in 1:ntrials   
                # Run Circuit
                MPSSys, entropy = RunMonitoredCircuit(Circuit, prob, CircuitMTrajectory)
                push!(entropytrialarray, entropy)
                println(entropy)
                global GateCount = 0
            end
            push!(smallSEArray, std(entropytrialarray))
            push!(AvgEntropyArray, mean(entropytrialarray))
        end
        
        
        
        # Printing Entropy
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        =#
        
        
        # Bring the entropy array to the array for each system size
        #push!(BigEntropyArray, AvgEntropyArray)
        #push!(BigSEArray, smallSEArray)
    end
end

print(BigEntropyArray)

plot(log10.(ToleranceArray), BigEntropyArray, label="L=8")
scatter!(log10.(ToleranceArray), BigEntropyArray, label="")
xlabel!("log(Tolerance)")
ylabel!("Average Entropy for p = 0.05")
title!("Ancilla One Circuit Entropy, Trajectory")
png("AncillaFixedCircuitToleranceVsEntropy(1e-10to1e-1)LogSpaced100points")


plot(log10.(ToleranceArray), BigDimArray, label="L=8")
scatter!(log10.(ToleranceArray), BigDimArray, label="")
xlabel!("log(Tolerance)")
ylabel!("Max Bond Dimension")
title!("Ancilla One Circuit BondDim, Trajectory Setup")
png("AncillaFixedCircuitToleranceVsBondDim(1e-10to1e-1)LogSpaced100points")


print(MeasurementTrajectoryForced)
print(phi0Array)

print(BigEntropyArray)
#save_object("ToleranceTrajectoryMTrajectory.jld2", MeasurementTrajectoryForced)
#save_object("phi0s.jld2", phi0Array)
#save_object("phi1s.jld2", phi1Array)
#save_object("phi2s.jld2", phi2Array)
#save_object("phi3s.jld2", phi3Array)
#save_object("thetas.jld2", thetaArray)



plot(problist, real.(BigEntropyArray[1]), ylim=(-0.5,5), label="L=8")
plot!(problist, real.(BigEntropyArray[2]), label="L=12")
xlabel!("Measurement Rate")
ylabel!("Average Entropy")
title!("U(1) Gates; ntrials = 200")

png("U(1) Gates with 1e-5 Tolerance")
