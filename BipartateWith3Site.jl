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

function ITensors.op(::OpName"ThreeSiteProjectThreeCharge", ::SiteType"S=1/2", s1::Index, s2::Index, s3::Index)

    TempMatrix = zeros((8,8))
    TempMatrix[1,1] = 1

    return itensor(TempMatrix, s3', s2', s1', s3, s2, s1)
end

function ITensors.op(::OpName"ThreeSiteProjectNoCharge", ::SiteType"S=1/2", s1::Index, s2::Index, s3::Index)

    TempMatrix = zeros((8,8))
    TempMatrix[8,8] = 1

    return itensor(TempMatrix, s3', s2', s1', s3, s2, s1)
end



function ITensors.op(::OpName"ThreeSiteProjectTwoCharge", ::SiteType"S=1/2", s1::Index, s2::Index, s3::Index)

    TempMatrix = zeros((8,8))
    TempMatrix[2,2] = 1
    TempMatrix[3,3] = 1
    TempMatrix[5,5] = 1

    return itensor(TempMatrix, s3', s2', s1', s3, s2, s1)
end

function ITensors.op(::OpName"ThreeSiteProjectOneCharge", ::SiteType"S=1/2", s1::Index, s2::Index, s3::Index)

    TempMatrix = zeros((8,8))
    TempMatrix[4,4] = 1
    TempMatrix[6,6] = 1
    TempMatrix[7,7] = 1

    return itensor(TempMatrix, s3', s2', s1', s3, s2, s1)
end

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


function PerformThreeSiteMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)
    for MeasurementTuple in MeasurementSites
        Site1 = MeasurementTuple[1]
        Site2 = MeasurementTuple[2]
        Site3 = MeasurementTuple[3]
        normalize!(MPSSys)
        orthogonalize!(MPSSys, MeasurementTuple[1])

        MPSSysDagger = prime(dag(MPSSys), linkinds(MPSSys))

        DensityMatrix = prime(MPSSys[Site1], linkinds(MPSSys, Site1-1)) * prime(MPSSysDagger[Site1], s[Site1])
        DensityMatrix *= MPSSys[Site2]
        DensityMatrix *= prime(MPSSysDagger[Site2], s[Site2])
        DensityMatrix *= prime(MPSSys[Site3], linkinds(MPSSys, Site3)) * prime(MPSSysDagger[Site3], s[Site3])
        
        BornProbabilities = []
        for i = 1:2, j = 1:2, k = 1:2
            push!(BornProbabilities, DensityMatrix[s[Site1] => i, s[Site1]' => i, s[Site2] => j, s[Site2]' => j, s[Site3] => k, s[Site3]' => k])
    
        end
        BornProbabilities = real.(BornProbabilities)

        NoChargeProb = BornProbabilities[8]
        OneChargeProb = BornProbabilities[4] + BornProbabilities[6] + BornProbabilities[7] 
        TwoChargeProb = BornProbabilities[2] + BornProbabilities[3] + BornProbabilities[5] 
        ThreeChargeProb = BornProbabilities[1]

        items = ["3Charge", "2Charge", "1Charge", "0Charge"]
        weights = [ThreeChargeProb, TwoChargeProb, OneChargeProb, NoChargeProb]
        ProjectionDecision = StatsBase.sample(items, Weights(weights))


        if ProjectionDecision == "3Charge"
            Projection = [("ThreeSiteProjectThreeCharge", Site1, Site2, Site3)]
            #Projection = [("ProjUp", Site1), ("ProjUp", Site2), ("ProjUp", Site3)]
        elseif ProjectionDecision == "2Charge"
            Projection = [("ThreeSiteProjectTwoCharge", Site1, Site2, Site3)]
        elseif ProjectionDecision == "1Charge"
            Projection = [("ThreeSiteProjectOneCharge", Site1, Site2, Site3)]
        elseif ProjectionDecision == "0Charge"
            Projection = [("ThreeSiteProjectNoCharge", Site1, Site2, Site3)]
            #Projection = [("ProjDn", Site1), ("ProjDn", Site2), ("ProjDn", Site3)]
        end
        MPSSys = apply(ITensors.ops(Projection, s), MPSSys, cutoff=tol)
        normalize!(MPSSys)
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

    # Apply gates and measurements
    for (layer, i) in zip(circuit, 1:length(circuit))

        # Applying Gates
        MPSSys = apply(ITensors.ops(layer, s), MPSSys, cutoff=tol)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 

        MultiSiteMeasurements = []

        if rem(N,4) == 0 
            if iseven(i)
                for site in 3:3:N
                    if rand() < p
                        push!(MultiSiteMeasurements, (site, site+1, site+2))
                    end
                end
            else 
                for site in 1:3:N-2
                    if rand() < p
                        push!(MultiSiteMeasurements, (site, site+1, site+2))
                    end
                end
            end
            MPSSys = PerformThreeSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        else
            if iseven(i)
                for site in 2:3:N-1
                    if rand() < p
                        push!(MultiSiteMeasurements, (site, site+1, site+2))
                    end
                end
            else 
                for site in 1:3:N-1
                    if rand() < p
                        push!(MultiSiteMeasurements, (site, site+1, site+2))
                    end
                end
            end
            MPSSys = PerformThreeSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        end
    end

    return MPSSys, VonNeumannEntropy(MPSSys)
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
            @time Threads.@threads for i in 1:ntrials   
                # Create Circuit of size N
                Circuit = CreateCircuit(CircuitDepth, N)
                MPSSys, entropy = RunMonitoredCircuit(Circuit, prob)
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

save_object("L68EntanglementBipartateThreeSiteWideEntropies.jld2", BigEntropyArray)
save_object("L68EntanglementBipartateThreeSiteWideError.jld2", BigSEArray)
save_object("L68EntanglementBipartateThreeSiteWideVariance.jld2", BigVarArray)
save_object("L68EntanglementBipartateThreeSiteProbList.jld2", problist)


plot(problist, real.(BigEntropyArray[1]), label="L=8")
plot!(problist, real.(BigEntropyArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Average Entropy")
title!("Bipartate 3-Site Measurements ; ntrials = 1000")
png("Three Site Measurement, UpDn Setup, Bipartate, 1000 trials")

plot(problist, real.(BigVarArray[1]), label="L=8")
plot!(problist, real.(BigVarArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Variance in Entropy")
title!("Bipartate 3-Site Measurements Entropy Variance ; ntrials = 1000")
png("Three Site Measurement, UpDn Setup, Bipartate,Variance, 1000 trials")
