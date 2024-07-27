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

function ITensors.op(::OpName"ControlledRy", ::SiteType"S=1/2", s1::Index, s2::Index)

    angle2 = acos(1/sqrt(2)) 

    TempMatrix = zeros((4,4))
    TempMatrix[1,1] = 1
    TempMatrix[2,2] = cos(angle2/2)
    TempMatrix[3,3] = 1
    TempMatrix[4,4] = cos(angle2/2)

    TempMatrix[4,2] = sin(angle2/2)
    TempMatrix[2,4] = -sin(angle2/2)

    return itensor(TempMatrix, s2', s1', s2, s1)
end

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
    #MPSSys = FocusOnCircuit(MPSSys)
    N = length(MPSSys)


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

# Adjusted to allow ancilla
function CreateCircuitAncilla(depth::Int, N::Int)
    Circuit = []
    for i in 1:depth
        oddeventemplayer = []
        for j in 2:2:N-1
            push!(oddeventemplayer, ("ChargeConservingUnitaryGate2", j, j+1))
        end
        push!(Circuit, oddeventemplayer)
        #=
        else
            evenoddtemplayer = []
            for j in 3:2:N-2
                push!(evenoddtemplayer, ("ChargeConservingUnitaryGate2", j, j+1))
            end
            push!(Circuit, evenoddtemplayer)
        end
        =#
    end
    return Circuit
end

function PerformSingleSiteMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)

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
            os += "ProjUp", site
            ProjectUpMPO = MPO(os, s)
            MPSSys = apply(ProjectUpMPO, MPSSys, cutoff=tol)
            addzero()
        else 
            os = OpSum()
            os += "ProjDn", site
            ProjectDownMPO = MPO(os, s)
            MPSSys = apply(ProjectDownMPO, MPSSys, cutoff=tol)
            addone()
        end
        normalize!(MPSSys)
    end

    noprime!(MPSSys)

    return normalize!(MPSSys)
end

function PerformTwoSiteMeasurement!(MPSSys::MPS, MeasurementSites::Vector, s)
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

        print(BornProbabilities)


        if ProjectionDecision == "2Charge"
            Projection = [("TwoSiteProjectUp", Site1, Site2)]
            #Projection = [("ProjectUp", Site1), ("ProjectUp", Site2)]
            println("a")
        elseif ProjectionDecision == "1Charge"
            Projection = [("TwoSiteProjectOneCharge", Site1, Site2)]
            println("b")
        elseif ProjectionDecision == "0Charge"
            Projection = [("TwoSiteProjectDown", Site1, Site2)]
            #Projection = [("ProjectDown", Site1), ("ProjectDown", Site2)]
            println("c")
        end
        MPSSys = apply(ITensors.ops(Projection, s), MPSSys, cutoff=tol)
        normalize!(MPSSys)
    end
    noprime!(MPSSys)
    return normalize!(MPSSys)
end

function Scramble!(MPSSys::MPS, ScrambleDepth::Int, N::Int, s)

    # Entangle Ancilla and First Qubit
    entangleancilla1 = [("H", 1)]
    MPSSys = apply(ITensors.ops(entangleancilla1, s), MPSSys)
    entangleancilla2 = [("CNOT", 1, 2)]
    MPSSys = apply(ITensors.ops(entangleancilla2, s), MPSSys)
    entangleancilla3 =[("CNOT", 2, 3)]
    MPSSys = apply(ITensors.ops(entangleancilla3, s), MPSSys)

    # Create scramble layer gates
    if ScrambleDepth != 0
        ScrambleCircuit = CreateCircuitAncilla(ScrambleDepth, N+1)
        
        # Applying the scrambling
        
        for scramblelayer in ScrambleCircuit
            MPSSys = apply(ITensors.ops(scramblelayer, s), MPSSys)
        end
    end

    return MPSSys
end

function RunMonitoredCircuit(circuit::Vector, p::Float64, N::Int, ScrambleDepth::Int)

    # Number of Qubits N

    # Create Sites for MPS and ancilla
    s = siteinds("S=1/2", N+1)

    # Produce MPS state with ancilla 
    #MPSSys = MPS(s, n -> iseven(n) || n == 1 ? "↑" : "↑")
    teststate = ["Up", "Up", "Dn"]
    MPSSys = MPS(s, teststate)

    testangle = acos(1/sqrt(3)) 

    #=
    gate1 = [("H", 1)]
    MPSSys = apply(ITensors.ops(gate1, s), MPSSys)
    #gate2 = [("Ry", 3, (θ = testangle,))]
    gate2 = [("CNOT", 1, 2)]
    MPSSys = apply(ITensors.ops(gate2, s), MPSSys)
    =#

    
    MPSSys = Scramble!(MPSSys, ScrambleDepth, N, s)

    #=
    # Apply gates and measurements
    for (layer, i) in zip(circuit, 1:length(circuit))

        # Applying Gates
        MPSSys = apply(ITensors.ops(layer, s), MPSSys, cutoff=tol)

        # Possible Measurement
        # For each qubit, according to a given probability of measurement, measure it. 

        MultiSiteMeasurements = []

        #=
        for site in 2:2:N
            if rand() < p
                push!(MultiSiteMeasurements, (site, site+1))
            end
        end
        MPSSys = PerformMultiSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        =#

        
        if isodd(i)
            for site in 2:2:N
                if rand() < p
                    push!(MultiSiteMeasurements, (site, site+1))
                end
            end
            MPSSys = PerformTwoSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        else
            for site in 3:2:N-1
                if rand() < p
                    push!(MultiSiteMeasurements, (site, site+1))
                end
            end
            MPSSys = PerformTwoSiteMeasurement!(MPSSys, MultiSiteMeasurements, s)
        
            #=
            SingleSiteMeasurements = []
            for site in [2, N+1]
                if rand() < p
                    push!(SingleSiteMeasurements, site)
                end
            end
            MPSSys = PerformSingleSiteMeasurement!(MPSSys, SingleSiteMeasurements, s)
            =#
        end
    end
    =#
    MPSSys = PerformTwoSiteMeasurement!(MPSSys, [(2,3)], s)
    return MPSSys, VonNeumannEntropy(MPSSys)
end

zerocount = 0
onecount = 0

let 
    # Tolerance
    global tol = 1e-150
    # Measurement probabilities
    global problist = [0.0] #LinRange(0.0, 0.5, 20)
    # Array to hold each size iteration
    global BigEntropyArray = []
    # SE Array
    global BigSEArray = []
    # Array of System Sizes
    global SystemSizes = [2]
    # How many layers of scrambling (2 -> 1 even 1 odd)
    global ScrambleDepth = 0
    # Variance
    global BigVarArray = []


    for N in SystemSizes
        # Circuit Depth
        CircuitDepth = 6
        # Number of trials
        ntrials = 1
        # Average Entropy Array per System Size
        AvgEntropyArray = []
        smallSEArray = []
        SmallVarArray = []
        
        for prob in problist
            # Holds all the entropies for a single p for the amount of trials in a certain system
            entropytrialarray = zeros(ntrials)
            @time for i in 1:ntrials
                # Create Circuit for N qubitis (N+1) as there is an ancilla
                Circuit = CreateCircuitAncilla(CircuitDepth, N+1)
                # Run the monitored circuit 
                MPSSys, entropy = RunMonitoredCircuit(Circuit, prob, N, ScrambleDepth)
                # Keep track of entropies
                #push!(entropytrialarray, entropy)
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

println(zerocount)
println(onecount)

print(real.(BigEntropyArray[1]))

plot(problist, real.(BigEntropyArray[1]), label="L=6")
plot!(problist, real.(BigEntropyArray[2]), label="L=8")
plot!(problist, real.(BigEntropyArray[3]), label="L=10")
scatter!(problist, real.(BigEntropyArray), label="")
xlabel!("p")
ylabel!("Average Entropy")


save_object("EntagnlementTwoSiteWideEntropies.jld2", BigEntropyArray)
save_object("EntagnlementTwoSiteWideError.jld2", BigSEArray)
save_object("EntagnlementTwoSiteWideVariance.jld2", BigVarArray)
save_object("EntagnlementTwoSiteWideProbList.jld2", problist)

title!("Two Site Measurements, Entanglement, 6 Scramble Layers")
png("Two Site Measurements ,Entanglement Transition, and Ancilla and Scrambling 1000 trials")

Threads.nthreads()


plot(problist, real.(BigVarArray[1]), label="L=6")
plot!(problist, real.(BigVarArray[2]), label="L=8")
plot!(problist, real.(BigVarArray[3]), label="L=10")
scatter!(problist, real.(BigVarArray), label="")
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Two Site Measurement Variance Entanglement Tranition")
png("Two Site Measurement, Entanglement Trasnition and Ancilla and Scrambling, Variance 1000 trials")


OriginalMeasurementVariance = load_object("ChargeSharpeningOriginalSetupVariance.jld2")
SingleSiteMeasurementVariance = load_object("ChargeSharpeningSingleSiteWideVariance.jld2")

plot(problist, OriginalMeasurementVariance[3], label="L= Original Two Site" )
plot!(problist, SingleSiteMeasurementVariance[3], label="L= SingleSite Measurements")
scatter!(problist, OriginalMeasurementVariance[3], label="")
scatter!(problist, SingleSiteMeasurementVariance[3], label="" )
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Charge Sharpening L = 10")
png("Charge Sharpening L=10 Single vs Two Variance")

plot(problist, OriginalMeasurementVariance[2], label="L=Original Two Site" )
plot!(problist, SingleSiteMeasurementVariance[2], label="L=Even Measurements")
scatter!(problist, OriginalMeasurementVariance[2], label="")
scatter!(problist, SingleSiteMeasurementVariance[2], label="" )
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Charge Sharpening L = 8 ")
png("Charge Sharpening L=8 Single vs Two Variance")

plot(problist, OriginalMeasurementVariance[1], label="L=Original Two Site" )
plot!(problist, SingleSiteMeasurementVariance[1], label="L=Even Measurements")
scatter!(problist, OriginalMeasurementVariance[1], label="")
scatter!(problist, SingleSiteMeasurementVariance[1], label="" )
xlabel!("p")
ylabel!("Variance in Entropy")
title!("Charge Sharpening ")
png("Charge Sharpening L=6 Single vs Two Setups")

