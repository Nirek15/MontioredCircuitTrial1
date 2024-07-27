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

    Matrix = [
        exp(1im * phi0)        0                     0               0;
        0              TwoTwoHaarRandom[1]  TwoTwoHaarRandom[3]      0;
        0              TwoTwoHaarRandom[2]  TwoTwoHaarRandom[4]      0;
        0                      0                     0         exp(1im*phi3)
    ]

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
                push!(oddeventemplayer, ("ChargeConservingUnitaryGate2", j, j+1))
            end
            push!(Circuit, oddeventemplayer)
        else
            evenoddtemplayer = []
            for j in 3:2:N-2
                push!(evenoddtemplayer, ("ChargeConservingUnitaryGate2", j, j+1))
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

function Entangle!(MPSSys::MPS, s, TransitionType::String)

    # Entangle Ancilla and First Qubit
    entangleancilla1 = [("H", 1)]
    MPSSys = apply(ITensors.ops(entangleancilla1, s), MPSSys)
    entangleancilla2 = [("CNOT", 1, 2)]
    MPSSys = apply(ITensors.ops(entangleancilla2, s), MPSSys)

    # If Looking at Entanglement Transition
    if TransitionType == "ET"
        entangleancilla3 = [("CNOT", 2, 3)]
        MPSSys = apply(ITensors.ops(entangleancilla3, s), MPSSys)
    end

    return MPSSys
end

function RunMonitoredCircuit(CircuitGateLayers::Vector, N::Int, MeasuringLayers::Vector, ScrambleDepth::Int, TransitionType::String, s)

    # Number of Qubits N

    # Produce MPS state with ancilla 
    FirstTwoQubits = ["Up", "Dn"]
    RestofQubits = repeat(["Dn", "Up"], outer = Int(N/2 - 1))
    FinalQubit = ["Dn"]

    IntialState = vcat(FirstTwoQubits, RestofQubits, FinalQubit)

    MPSSys = MPS(s, IntialState)

    MPSSys = Entangle!(MPSSys, s, TransitionType)

    # Apply Gates and Measurement
    for (GateLayer, i) in zip(CircuitGateLayers, 1:length(CircuitGateLayers))

        MPSSys = apply(GateLayer, MPSSys, cutoff=tol)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 
        if i > ScrambleDepth
            MPSSys = PerformMeasurement!(MPSSys, MeasuringLayers[i], s)
        end
    end

    return MPSSys, VonNeumannEntropy(MPSSys), HalfSystemCharge(MPSSys, s)
end

function RunSetupCircuit(CircuitDepth::Int, ScrambleDepth::Int, N::Int, p::Float64)

    # Total Depth of Circuit
    TotalCircuitLength = CircuitDepth + ScrambleDepth

    # Create Site Indecies
    s = siteinds("S=1/2", N+1)

    # Create Circuit Structure (Brick)
    Circuit = CreateCircuitAncilla(TotalCircuitLength, N+1)

    CircuitGateLayers = []
    for GateLayer in Circuit
        push!(CircuitGateLayers, ITensors.ops(GateLayer, s))
    end

    # Determine Where Measurements Will Occur
    MeasuringLayers = [[] for _=1:TotalCircuitLength]
    for i in ScrambleDepth+1:TotalCircuitLength
        for j in 2:N+1
            if rand() < p
                push!(MeasuringLayers[i], j)
            end
        end
    end


    return CircuitGateLayers, MeasuringLayers, s
end


let 
    # Tolerance
    global tol = 1e-150
    # Measurement probabilities
    global problist = LinRange(0.0, 0.5, 10)
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
    # Number of Circuits 
    global NumCircuits = 10
    # Number of ToleranceTrajectories 
    global NumTrajectories = 1000


    for N in SystemSizes
        
        # Transition Type ("ET" or "CS")
        TransitionType = "CS"

        # Circuit Depth
        CircuitDepth = 2 * N

        # How many layers of scrambling (2 -> 1 even 1 odd)
        global ScrambleDepth = N

        # Arrays for Data Collection
        AvgEntropyArray = []
        SmallSEArray = []
        SmallVarArray = []
        SmallChargeVarArray = []
    
        
        for prob in problist
        
            # Since we are averaging over circuits
            CircuitsForProb = []
            MeasuringForProb = []
            SitesIndexForProb = []

            for i in 1:NumCircuits
                # Create the Circuit for the First Time (Doesn't do anything just creates a vector of connected sites) Creates Scrambling Layer as Well, and Indexes
                CircuitGateLayers, MeasuringLayers, s = RunSetupCircuit(CircuitDepth, ScrambleDepth, N, prob)
                push!(CircuitsForProb, CircuitGateLayers)
                push!(MeasuringForProb, MeasuringLayers)
                push!(SitesIndexForProb, s)
            end

            # Entropies and Charges or a certain measurement rate
            EntropiesForSProb = []
            ChargesForSProb = []


            for (CircuitGateLayers, MeasuringLayers, s) in zip(CircuitsForProb, MeasuringForProb, SitesIndexForProb)
                # Holds all the entropies for a single p for the amount of trials in a certain system (S is specific)
                EntropiesForSCircuitSProb = zeros(NumTrajectories)
                ChargesForSCircuitSProb = zeros(NumTrajectories)
                @time @Threads.threads for i in 1:NumTrajectories
                    # Run the monitored circuit 
                    MPSSys, entropy, charge = RunMonitoredCircuit(CircuitGateLayers, N, MeasuringLayers, ScrambleDepth, TransitionType, s)
                    # Keep track of entropies since multi-threading
                    EntropiesForSCircuitSProb[i] = entropy
                    ChargesForSCircuitSProb[i] = charge
                end 

                push!(EntropiesForSProb, EntropiesForSCircuitSProb)
                push!(ChargesForSProb, ChargesForSCircuitSProb)
            end

            # Flatten
            EntropiesForSProb = vcat(EntropiesForSProb...)
            ChargesForSProb = vcat(ChargesForSProb...)

            # Compute Quantities for each prob
            push!(SmallSEArray, std(EntropiesForSProb))
            push!(AvgEntropyArray, mean(EntropiesForSProb))
            push!(SmallVarArray, var(EntropiesForSProb))
            push!(SmallChargeVarArray, var(ChargesForSProb))
        end
        
        # Printing Entropy
        for avgentropy in AvgEntropyArray
            println("Entropy: ", avgentropy)
        end
        
        # Bring the entropy array to the array for each system size
        push!(BigEntropyArray, AvgEntropyArray)
        push!(BigSEArray, SmallSEArray)
        push!(BigVarArray, SmallVarArray)
        push!(BigChargeVarArray, SmallChargeVarArray)
    end
end


plot(problist, real.(BigEntropyArray[1]), label="L=8")
plot!(problist, real.(BigEntropyArray[2]), label="L=10")
scatter!(problist, real.(BigEntropyArray), label="")
xlabel!("p")
ylabel!("Average Entropy")


save_object("8GateTrajectoryAverageCSOneSiteWideEntropy.jld2", BigEntropyArray)
save_object("8GateTrajectoryAverageCSOneSiteWideStandardError.jld2", BigSEArray)
save_object("8GateTrajectoryAverageCSOneSiteWideEntropyVariance.jld2", BigVarArray)
save_object("8GateTrajectoryAverageCSOneSiteWideChargeVariance.jld2", BigChargeVarArray)
save_object("8GateTrajectoryAverageCSOneSiteWideProbList.jld2", problist)

title!("Ancilla One Site Measurement, CS, N Scrambling Layers; ntrials=1000; Circuits=10")
png("GateTrajectoryAverage, One Site Entropy, CS, and Ancilla and N-Scrambling 1000 trials 10Circuits")


plot(problist, real.(BigVarArray[1]), label="L=8")
plot!(problist, real.(BigVarArray[2]), label="L=10")
scatter!(problist, real.(BigVarArray), label="")
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Ancialla One Site, Entropy Variance, CS")
png("GateTrajectoryAverage Setup, One Site Entropy Variance, CS, and Ancilla and N-Scrambling 1000 trials")

plot(problist, real.(BigChargeVarArray[1]), label="L=8")
plot!(problist, real.(BigChargeVarArray[2]), label="L=10")
scatter!(problist, real.(BigChargeVarArray), label="")
xlabel!("p")
ylabel!("Variance in Half System Charge")
title!("Ancilla One Site, HalfSys Charge Variance, CS")
png("GateTrajectoryAverage Setup, One Site HalfSys Charge Variance, CS, and Ancilla and N-Scrambling 1000 trials 10Circuits")


#=
a = 1
@time @Threads.threads for i in 2:8
    b = a
    b+=1
    println(b)
end



slist = []
GateTest = []
N=8

s = slist[1] #siteinds("S=1/2", N+1)
#push!(slist, s)



FirstTwoQubits = ["Up", "Dn"]
RestofQubits = repeat(["Dn", "Up"], outer = Int(N/2 - 1))
FinalQubit = ["Dn"]

IntialState = vcat(FirstTwoQubits, RestofQubits, FinalQubit)

MPSSys = MPS(s, IntialState)

MPSSys = Entangle!(MPSSys, s)

Circuit = CreateCircuitAncilla(6, N+1)

MeasurementPosTrial = []

tol=1e-150
for i in 1:6

    #hi = ITensors.ops(Circuit[i], s)
    #push!(GateTest, hi)
    applynow = GateTest[i]

    MPSSys = apply(applynow, MPSSys, cutoff=1e-150)

    if i > 3
        #MeasuringSites = []
        #for j in 2:N+1
        #    if rand() < 0.1
        #        push!(MeasuringSites,j)
        #    end
        #end
        #push!(MeasurementPosTrial, MeasuringSites)
        MPSSys = PerformMeasurement!(MPSSys, MeasurementPosTrial[i-3], s)
    end

end

print(VonNeumannEntropy(MPSSys))
=#