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

ForcedMeasurementArray = []

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
        #end
    end

    return entropy
end

function CreateCircuit(depth::Int, N::Int)
    Circuit = []
    for i in 1:depth
        if isodd(i)
            oddeventemplayer = []
            for j in 1:2:N-1
                push!(oddeventemplayer, ("ChargeConservingUnitaryGate", j, j+1))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 2:2:N-2
                push!(evenoddtemplayer, ("ChargeConservingUnitaryGate", j, j+1))
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end


function PerformMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)

    ###
    #LayerForcedMeasurements = []

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
            addzero()

            ###
            #push!(LayerForcedMeasurements, (site, 0))
        else 
            os = OpSum()
            os += "ProjectDown", site
            ProjectDownMPO = MPO(os, s)
            MPSSys = apply(ProjectDownMPO, MPSSys, cutoff=tol)
            addone()

            ###
            #push!(LayerForcedMeasurements, (site, 1))
        end
    end
    noprime!(MPSSys)

    ### 
    #push!(ForcedMeasurementArray, LayerForcedMeasurements)

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
            addzero()
        else 
            os = OpSum()
            os += "ProjectDown", mtuple[1]
            ProjectDownMPO = MPO(os, s)
            MPSSys = apply(ProjectDownMPO, MPSSys, cutoff=tol)
            addone()
        end
    end
    noprime!(MPSSys)
    return normalize!(MPSSys)
end

function CreateMeasurementTrajectory(N::Int, depth::Int, p::Float64)
    MeasurementTrajectory = []
    for _ in 1:depth
        LayerMeasurementsSites = []
        for i in 1:N
            if rand() < p
                push!(LayerMeasurementsSites, i)
            end
        end
        push!(MeasurementTrajectory, LayerMeasurementsSites)
    end
    return MeasurementTrajectory
end

function RunMonitoredCircuit(circuit::Vector, MeasurementTrajectory::Vector)

    # Number of Qubits
    N = nqubits(circuit)

    # Create Site
    s = siteinds("S=1/2", N)

    # Produce MPS state
    MPSSys = MPS(s, n -> iseven(n) ? "↓" : "↑")

    # Apply gates and measurements
    for (layer, i) in zip(circuit, 1:length(circuit))

        # Applying Gates
        MPSSys = apply(ITensors.ops(layer, s), MPSSys)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 
        MPSSys = PerformMeasurement!(MPSSys, MeasurementTrajectory[i], s)

        ###
        #MPSSys = PerformForcedMeasurement!(MPSSys, MeasurementTrajectory[i], s)

    end
    return MPSSys, VonNeumannEntropy(MPSSys)
end

zerocount = 0
onecount = 0
tol = 0


let 
    # Tolerance
    #global tol = 1e-300
    # Measurement probabilities
    global problist = [0.05] #LinRange(0.1, 0.8, 13)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # SE Array
    global BigSEArray = []
    # Array of System Sizes
    global SystemSizes = [8]
    
    global ToleranceArray = 10 .^ range(-40, stop=-1, length=100)

    global BigDimArray = []


    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 2 * N
        # Number of trials
        ntrials = 500
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        SmallSEArray = []


         Circuit = load_object("ToleranceCheckCircuit.jld2")
         MeasurementTrajectory = load_object("ToleranceCheckTrajectory.jld2") #ForcedMeasurementArray

         

        for tolerance in ToleranceArray
            global tol = tolerance
            entropytrialarray = [] 
            SmallDimArray = []
            @time for _ in 1:ntrials 
                MPSSys, entropy = RunMonitoredCircuit(Circuit, MeasurementTrajectory)
                push!(entropytrialarray, entropy)
                push!(SmallDimArray, maxlinkdim(MPSSys))
            end
            push!(BigEntropyArray, mean(entropytrialarray))
            push!(BigDimArray, mean(SmallDimArray))
        end
        
        
        #=
        # Entropy fluctuating at some points (Assuming this is because im only running each circuit once) (still fluctuates when averaging)
        for prob in problist
            entropytrialarray = []
            @time for _ in 1:ntrials   
                # Run Circuit
                MPSSys, entropy = RunMonitoredCircuit(Circuit, ForcedMeasurementArray)
                push!(entropytrialarray, entropy)
                println(entropy)
            end
            push!(SmallSEArray, std(entropytrialarray))
            push!(AvgEntropyArray, mean(entropytrialarray))
        end
        =#
        
        # Printing Entropy
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        
        # Bring the entropy array to the array for each system size
        #push!(BigEntropyArray, AvgEntropyArray)
        #push!(BigSEArray, smallSEArray)
    end
end


print(ForcedMeasurementArray)

println(zerocount)
println(onecount)

print(length(BigEntropyArray))

plot(log10.(ToleranceArray), BigEntropyArray, label="L=8")
scatter!(log10.(ToleranceArray), BigEntropyArray, label="")
xlabel!("log(Tolerance)")
ylabel!("Average Entropy for p = 0.05")
title!("One Circuit Entropy, Measurement Trajectory Setup; ntrials = 100")
png("ToleranceVsEntropy2(1e-50to1e-1)LogSpaced100points")


plot(log10.(ToleranceArray), BigDimArray, label="L=8")
scatter!(log10.(ToleranceArray), BigDimArray, label="")
xlabel!("log(Tolerance)")
ylabel!("Max Bond Dimension")
title!("One Circuit BondDim, Measurement Trajectory Setup; ntrials = 100")
png("ToleranceVsBondDim2(1e-50to1e-1)LogSpaced100points")


ToleranceEntropyTuples = collect(zip(ToleranceArray, BigEntropyArray))
ToleranceBondDimTuples = collect(zip(ToleranceArray, BigDimArray))


save_object("ToleranceEntropyTuples2(1e-50to1e-1)LogSpaced100.jld2", ToleranceEntropyTuples)
save_object("ToleranceBondDimTuples2(1e-50to1e-1)LogSpaced100.jld2", ToleranceBondDimTuples)

png("ToleranceVsEntropy(1e-150to1e-50)LogSpaced100points")


start_exponent = -50
end_exponent = -10
num_points = 100  # Adjust this to the desired number of points

points = 10 .^ range(start_exponent, stop=end_exponent, length=num_points)

println(points)

println(ToleranceArray[end-10])
