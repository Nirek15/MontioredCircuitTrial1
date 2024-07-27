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

function ITensors.op(::OpName"ProjectUpTest", ::SiteType"S=1/2", s1::Index)

    Matrix = zeros((2,2))
    Matrix[1,1] = 1

    return itensor(Matrix, s1', s1)
end

function ITensors.op(::OpName"ProjectDownTest", ::SiteType"S=1/2", s1::Index)

    Matrix = zeros((2,2))
    Matrix[2,2] = 1

    return itensor(Matrix, s1', s1)
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

    push!(phi0Array, phi0)
    push!(TwoSiteHaarRandomArray, TwoTwoHaarRandom)
    push!(phi3Array, phi3)

    Matrix = [
        exp(1im * phi0)        0                     0               0;
        0              TwoTwoHaarRandom[1]  TwoTwoHaarRandom[3]      0;
        0              TwoTwoHaarRandom[2]  TwoTwoHaarRandom[4]      0;
        0                      0                     0         exp(1im*phi3)]

    return itensor(Matrix, s2', s1', s2, s1)
end

function ITensors.op(::OpName"FixedChargeConservingUnitaryGate2", ::SiteType"S=1/2", s1::Index, s2::Index, LocalGateCount)

    global GateCount += 1

    phi0 = phi0Array[GateCount]
    TwoTwoHaarRandom = TwoSiteHaarRandomArray[GateCount]
    phi3 = phi3Array[GateCount]

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

    # Cutoff at Ancilla
    SubSysCutoff = 1 
    orthogonalize!(MPSSys, SubSysCutoff)
    U,S,V = svd(MPSSys[SubSysCutoff], (linkinds(MPSSys, SubSysCutoff-1)..., siteinds(MPSSys, SubSysCutoff)...), cutoff=1e-150)


    entropy = 0
    for n in 1:ITensors.dim(S,1)
        pn = (S[n,n])^2
        entropy -= pn * log(pn)
    end

    return entropy
end

function HalfSystemCharge(MPSSys::MPS, s)
    halfsystem = []

    for i in 2:2:Int(round(length(MPSSys)/2))
        push!(halfsystem, (i, i+1))
    end

    MPSSys, charge = PerformTwoSiteMeasurement!(MPSSys, halfsystem, s)
    return charge
end

# Adjusted to allow ancilla
function CreateCircuitAncilla(depth::Int, N::Int)
    Circuit = []
    for i in 1:depth
        if isodd(i)
            oddeventemplayer = []
            for j in 2:2:N-1
                push!(oddeventemplayer, ("ChargeConservingUnitaryGate2", j, j+1, GateCounter))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 3:2:N-2
                push!(evenoddtemplayer, ("ChargeConservingUnitaryGate2", j, j+1, GateCounter))
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end

# Uses the fixed gates from the first trial of the circuit
function CreateFixedAncillaCircuit(depth::Int, N::Int, Scramble::Bool, ScrambleDepth::Int)
    Circuit = []
    if Scramble == true
        GateCounter = 1
    else 
        GateCounter = Int((N/2 + (N/2-1)) * (ScrambleDepth/2)) + 1
    end
    for i in 1:depth
        if isodd(i)
            oddeventemplayer = []
            for j in 2:2:N-1
                push!(oddeventemplayer, ("FixedChargeConservingUnitaryGate2", j, j+1, GateCounter))
                GateCounter += 1
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 3:2:N-2
                push!(evenoddtemplayer, ("FixedChargeConservingUnitaryGate2", j, j+1, GateCounter))
                GateCounter += 1
            end
            push!(Circuit, evenoddtemplayer)
        end
    end
    return Circuit
end

function PerformMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)

    for site in MeasurementSites
        normalize!(MPSSys)
        orthogonalize!(MPSSys, site)
        # Create a density matrix for the qubit we are measuring
        MPSMeasuring = MPSSys[site]
        SiteDensityMatrix = prime(MPSMeasuring, s) * dag(MPSMeasuring)

        DiagProababilities = []
        for i = 1:2
            push!(DiagProababilities, SiteDensityMatrix[s[site] => i, s[site]' => i])
        end

        # Computer decides if it gets projected up or down based on probability of down
        if rand() < real.(DiagProababilities[1])
            Projection = [("ProjUp", site)]
            MPSSys = apply(ITensors.ops(Projection, s), MPSSys, cutoff=tol)
        else 
            Projection = [("ProjDn", site)]
            MPSSys = apply(ITensors.ops(Projection, s), MPSSys, cutoff=tol)
        end
        normalize!(MPSSys)
    end

    noprime!(MPSSys)

    return normalize!(MPSSys)
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

function Scramble!(MPSSys::MPS, ScrambleDepth::Int, N::Int, s)

    # Entangle Ancilla and First Qubit
    entangleancilla1 = [("H", 1)]
    MPSSys = apply(ITensors.ops(entangleancilla1, s), MPSSys)
    entangleancilla2 = [("CNOT", 1, 2)]
    MPSSys = apply(ITensors.ops(entangleancilla2, s), MPSSys)
    entangleancilla3 = [("CNOT", 2, 3)]
    MPSSys = apply(ITensors.ops(entangleancilla3, s), MPSSys)

    # Create scramble layer gates
    if ScrambleDepth != 0
        ScrambleCircuit = CreateCircuitAncilla(ScrambleDepth, N+1, Scramble == true, ScrambleDepth)
        
        # Applying the scrambling
        
        for scramblelayer in ScrambleCircuit
            MPSSys = apply(ITensors.ops(scramblelayer, s), MPSSys)
        end
    end

    return MPSSys
end

function RunMonitoredCircuit(circuit::Vector, p::Float64, N::Int, ScrambleDepth::Int, FixedMeasurementPosition::Bool)

    # Number of Qubits N

    # Create Sites for MPS and ancilla
    s = siteinds("S=1/2", N+1)

    # Produce MPS state with ancilla 
    FirstTwoQubits = ["Up", "Dn"]
    RestofQubits = repeat(["Dn", "Up"], outer = Int(N/2 - 1))
    FinalQubit = ["Dn"]

    IntialState = vcat(FirstTwoQubits, RestofQubits, FinalQubit)

    MPSSys = MPS(s, IntialState)

    MPSSys = Scramble!(MPSSys, ScrambleDepth, N, s)

    # Apply gates and measurements
    for (layer, i) in zip(circuit, 1:length(circuit))

        # Applying Gates
        MPSSys = apply(ITensors.ops(layer, s), MPSSys, cutoff=tol)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 

        if FixedMeasurementPosition == false
            MeasuringSites = []
            for j in 2:N+1
                if rand() < p
                    push!(MeasuringSites,j)
                end
            end
            push!(MeasurementPositions, MeasuringSites)
        else 
            MeasuringSites = MeasurementPositions[i]
        end

        MPSSys = PerformMeasurement!(MPSSys, MeasuringSites, s)
    end
    return MPSSys, VonNeumannEntropy(MPSSys), HalfSystemCharge(MPSSys, s)
end


GateCount = 0
phi0Array = []
TwoSiteHaarRandomArray = []
phi3Array = []
MeasurementPositions = [] 


let 
    # Tolerance
    global tol = 1e-150
    # Measurement probabilities
    global problist = [0.1] #LinRange(0.0, 0.5, 10)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # SE Array
    global BigSEArray = []
    # Array of System Sizes
    global SystemSizes = [8]
    # Variance
    global BigVarArray = []
    # Charge Variance
    global BigChargeVarArray = []


    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 2 * N
        # How many layers of scrambling (2 -> 1 even 1 odd)
        global ScrambleDepth = N
        # Number of trials
        ntrials = 1000
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        smallSEArray = []
        SmallVarArray = []
        SmallChargeVarArray = []
        
        for prob in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = zeros(ntrials)
            chargetrialarray = zeros(ntrials)
            countertrialarray = zeros(ntrials)

            # Create the Circuit for the First Time (Doesn't do anything just creates a vector of connected sites)
            Circuit = CreateCircuitAncilla(CircuitDepth, N+1)

            # Running the first circuit (Determines the gates and the measurement positions) 
            FixedMeasurementPosition = false
            MPSSys, entropy, charge = RunMonitoredCircuit(Circuit, prob, N, ScrambleDepth, FixedMeasurementPosition)

            # Save Cirucit Data 
            #save_object("phi0Array.jdl2", phi0Array)
            #save_object("TwoSiteHaarRandomArray.jdl2", TwoSiteHaarRandomArray)
            #save_object("phi3Array.jdl2", phi3Array)
            #save_object("MeasurementPositions.jdl2", MeasurementPositions)

            # Keep track of entropy and charge 
            entropytrialarray[1] = entropy
            chargetrialarray[1] = charge

            @time @Threads.threads for i in 2:ntrials
            FixedMeasurementPosition = true
            # Load in the Circuit (N+1) as there is an ancilla
            Circuit = CreateFixedAncillaCircuit(CircuitDepth, N+1, Scramble==false, ScrambleDepth)
            # Run the monitored circuit 
            MPSSys, entropy, charge = RunMonitoredCircuit(Circuit, prob, N, ScrambleDepth, FixedMeasurementPosition)
            # Keep track of entropies since multi-threading
            entropytrialarray[i] = entropy
            chargetrialarray[i] = charge
            global GateCount = 0 
            end

            push!(smallSEArray, std(entropytrialarray))
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

print(length(MeasurementPositions))

plot(problist, real.(BigEntropyArray[1]), label="L=8")
plot!(problist, real.(BigEntropyArray[2]), label="L=10")
scatter!(problist, real.(BigEntropyArray), label="")
xlabel!("p")
ylabel!("Average Entropy")


save_object("New810CorrectAncillaSetupETOneSiteWideEntropy.jld2", BigEntropyArray)
save_object("New810CorrectAncillaSetupETOneSiteWideStandardError.jld2", BigSEArray)
save_object("New810CorrectAncillaSetupETOneSiteWideEntropyVariance.jld2", BigVarArray)
save_object("New810CorrectAncillaSetupETOneSiteWideChargeVariance.jld2", BigChargeVarArray)
save_object("New810CorrectAncillaSetupETOneSiteWideProbList.jld2", problist)

title!("Ancilla One Site Measurement, ET, N Scrambling Layers; ntrials=1000")
png("NewCorrectAncilla Setup, One Site Entropy, ET, and Ancilla and N-Scrambling 1000 trials")


plot(problist, real.(BigVarArray[1]), label="L=8")
plot!(problist, real.(BigVarArray[2]), label="L=10")
scatter!(problist, real.(BigVarArray), label="")
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Anciall One Site, Entropy Variance, ET")
png("CorrectAncilla Setup, One Site Entropy Variance, ET, and Ancilla and N-Scrambling 1000 trials")

plot(problist, real.(BigChargeVarArray[1]), label="L=8")
plot!(problist, real.(BigChargeVarArray[2]), label="L=10")
scatter!(problist, real.(BigChargeVarArray), label="")
xlabel!("p")
ylabel!("Variance in Half System Charge")
title!("Ancilla One Site, HalfSys Charge Variance, ET")
png("CorrectAncilla Setup, One Site HalfSys Charge Variance, ET, and Ancilla and N-Scrambling 1000 trials")

a = 1
@time @Threads.threads for i in 2:8
    b = a
    b+=1
    println(b)
end

