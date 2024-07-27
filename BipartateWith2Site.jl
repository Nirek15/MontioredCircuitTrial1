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

function ITensors.op(::OpName"TwoSiteProjectUp", ::SiteType"S=1/2", s1::Index, s2::Index)

    Matrix = zeros((4,4))
    Matrix[1,1] = 1

    return itensor(Matrix, s2', s1', s2, s1)
end

function ITensors.op(::OpName"TwoSiteProjectDown", ::SiteType"S=1/2", s1::Index, s2::Index)

    Matrix = zeros((4,4))
    Matrix[4,4] = 1

    return itensor(Matrix, s2', s1', s2, s1)
end

function ITensors.op(::OpName"TwoSiteProjectOneCharge", ::SiteType"S=1/2", s1::Index, s2::Index)

    Matrix = zeros((4,4))
    Matrix[2,2] = 1
    Matrix[3,3] = 1

    return itensor(Matrix, s2', s1', s2, s1)
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

function HalfSystemCharge(MPSSys::MPS, s)
    halfsystem = []

    for i in 1:2:(Int(length(MPSSys)/2))
        push!(halfsystem, (i, i+1))
    end

    MPSSys, charge = PerformTwoSiteMeasurement!(MPSSys, halfsystem, s)
    return charge
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


function PerformTwoSiteMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)
    charge = 0
    for MeasurementTuple in MeasurementSites
        Site1 = MeasurementTuple[1]
        Site2 = MeasurementTuple[2]
        normalize!(MPSSys)
        orthogonalize!(MPSSys, MeasurementTuple[1])

        MPSSysDagger = prime(dag(MPSSys), linkinds(MPSSys))

        DensityMatrix = prime(MPSSys[Site1], linkinds(MPSSys, Site1-1)) * prime(MPSSysDagger[Site1], s[Site1])
        DensityMatrix *= prime(MPSSys[Site2],linkinds(MPSSys, Site2)) * prime(MPSSysDagger[Site2], s[Site2])
        
        BornProbabilities = []
        for i = 1:2, j = 1:2
            push!(BornProbabilities, DensityMatrix[s[Site1] => i, s[Site1]' => i, s[Site2] => j, s[Site2]' => j])
        end
        BornProbabilities = real.(BornProbabilities)

        SingleChargeProb = BornProbabilities[2] + BornProbabilities[3]

        items = ["2Charge", "1Charge", "0Charge"]
        weights = [BornProbabilities[1], SingleChargeProb, BornProbabilities[4]]
        ProjectionDecision = StatsBase.sample(items, Weights(weights))

        if ProjectionDecision == "2Charge"
            Projection = [("TwoSiteProjectUp", Site1, Site2)]
            charge += 2
        elseif ProjectionDecision == "1Charge"
            Projection = [("TwoSiteProjectOneCharge", Site1, Site2)]
            charge += 1
        elseif ProjectionDecision == "0Charge"
            Projection = [("TwoSiteProjectDown", Site1, Site2)]
            charge += 0
        end
        MPSSys = apply(ITensors.ops(Projection, s), MPSSys, cutoff=tol)
        normalize!(MPSSys)
    end
    noprime!(MPSSys)
    return normalize!(MPSSys), charge
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
        
        if isodd(i)
            for site in 1:2:N
                if rand() < p
                    push!(MultiSiteMeasurements, (site, site+1))
                end
            end
            MPSSys, chargeuseless = PerformTwoSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        else
            for site in 2:2:N-1
                if rand() < p
                    push!(MultiSiteMeasurements, (site, site+1))
                end
            end
            MPSSys, chargeuseless = PerformTwoSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        end
    end
    return MPSSys, VonNeumannEntropy(MPSSys), HalfSystemCharge(MPSSys, s)
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
    global SystemSizes = [8, 10]
    # Variance
    global BigVarArray = []
    # Charge Variance
    global BigChargeVarArray = []

    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 2 * N
        # Number of trials
        ntrials = 1000
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        smallSEArray = []
        SmallVarArray = []
        SmallChargeVarArray = []
        
        # Entropy fluctuating at some points (Assuming this is because im only running each circuit once) (still fluctuates when averaging)
        for prob in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = zeros(ntrials)
            chargetrialarray = zeros(ntrials)
            @time @Threads.threads for i in 1:ntrials   
                # Create Circuit of size N
                Circuit = CreateCircuit(CircuitDepth, N)
                MPSSys, entropy, halfsyscharge = RunMonitoredCircuit(Circuit, prob)
                entropytrialarray[i] = entropy
                chargetrialarray[i] = halfsyscharge
            end
            push!(smallSEArray, std(entropytrialarray)/sqrt(ntrials))
            push!(AvgEntropyArray, mean(entropytrialarray))
            push!(SmallVarArray, var(entropytrialarray))
            push!(SmallChargeVarArray, var(chargetrialarray))
        end
        
        
        # Printing Entropy
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        
        # Bring the entropy array to the array for each system size
        push!(BigEntropyArray, AvgEntropyArray)
        push!(BigSEArray, smallSEArray)
        push!(BigVarArray, SmallVarArray)
        push!(BigChargeVarArray, SmallChargeVarArray)
    end
end

println(zerocount)
println(onecount)

save_object("HalfSysChargeL810EntanglementBipartateTwoSiteWideEntropies.jld2", BigEntropyArray)
save_object("HalfSysChargeL810EntanglementBipartateTwoSiteWideError.jld2", BigSEArray)
save_object("HalfSysChargeL810EntanglementBipartateTwoSiteWideVariance.jld2", BigVarArray)
save_object("HalfSysChargeL810EntanglementBipartateTwoSiteProbList.jld2", problist)
save_object("HalfSysChargeL810EntanglementBipartateTwoSiteChargeVar.jld2", BigChargeVarArray)



plot(problist, real.(BigEntropyArray[1]), label="L=8")
plot!(problist, real.(BigEntropyArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Average Entropy")
title!("Bipartate 2-Site Measurements ; ntrials = 1000")
png("Two Site Measurement, UpDn Setup, Bipartate, 1000 trials")

plot(problist, real.(BigVarArray[1]), label="L=6")
plot!(problist, real.(BigVarArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Variance in Entropy")
title!("Bipartate 2-Site Measurements Entropy Variance ; ntrials = 1000")
png("Two Site Measurement, UpDn Setup, Bipartate,Variance, 1000 trials")

plot(problist, real.(BigChargeVarArray[1]), label="L=8")
plot!(problist, real.(BigChargeVarArray[2]), label="L=10")
xlabel!("Measurement Rate p")
ylabel!("Variance In Half System Charge")
title!("Bipartate Two-Site Measurement; ntrials = 1000")
png("Two Site Measurement, UpDn Setup, Bipartate,Final Half Charge Variance, 1000 trials")
