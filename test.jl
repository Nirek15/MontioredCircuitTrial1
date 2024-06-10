using Printf
using LinearAlgebra
using StatsBase: mean, sem 
import Pkg; Pkg.add("StatsBase")
import Random
using Distributions
using Random

# define the two measurement projectors
gate(::GateName"Π0") =
  [1 0
   0 0]
gate(::GateName"Π1") =
  [0 0
   0 1]

# compute the Von Neumann entanglement entropy at the center bond 
# of a linear chain of qubits
function entanglemententropy(ψ₀::MPS)
  ψ = normalize!(copy(ψ₀))
  N = length(ψ)
  bond = N ÷ 2
  orthogonalize!(ψ, bond)

  row_inds = (linkind(ψ, bond - 1), siteind(ψ, bond))
  u, s, v = svd(ψ[bond], row_inds)

  S = 0.0
  for n in 1:dim(s, 1)
    λ = s[n, n]^2
    S -= λ * log(λ + 1e-20)
  end
  return S
end

# build a brick-layer circuit of random unitaries
function entangling_layer(N::Int)
  layer_odd  = randomlayer("RandomUnitary",[(j,j+1) for j in 1:2:N-1])
  layer_even = randomlayer("RandomUnitary",[(j,j+1) for j in 2:2:N-1])
  return [layer_odd..., layer_even...]
end

# perform a projective measurement in the computational basis
# at a given site
function projective_measurement!(ψ₀::MPS, site::Int)
  ψ = orthogonalize!(ψ₀, site)
  ϕ = ψ[site]
  # 1-qubit reduced density matrix
  ρ = prime(ϕ, tags="Site") * dag(ϕ)
  # Outcome probabilities
  prob = real.(diag(array(ρ)))
  # Sample
  σ = (rand() < prob[1] ? 0 : 1)
  # Projection
  ψ = runcircuit(ψ, ("Π"*"$(σ)", site))
  normalize!(ψ)
  ψ₀[:] = ψ
  return ψ₀
end

# compute average Von Neumann entropy for an ensemble of random circuits
# for a fixed local measurement probability rate `p`
function monitored_circuits(circuits::Vector{<:Vector}, p::Float64)
  svn = []
  N = nqubits(circuits[1])
  for circuit in circuits
    # initialize state ψ = |000…⟩
    ψ = productstate(N)
    # sweep over layers
    for layer in circuit
      # apply entangling unitary
      ψ = runcircuit(ψ, layer; cutoff = 1e-8)
      # perform measurements
      for j in 1:N
        p > rand() && projective_measurement!(ψ, j)
      end
    end
    push!(svn, entanglemententropy(ψ))
  end
  return svn
end

let
  Random.seed!(1234)
  N = 8       # number of qubits
  depth = 10   # circuit's depth
  ntrials = 50  # number of random trials

  # generate random circuits
  circuits = [[entangling_layer(N) for _ in 1:depth] for _ in 1:ntrials]

  # loop over projective measurement probability (per site)
  for p in 0.0:0.02:0.2
    t = @elapsed svn = monitored_circuits(circuits, p)
    @printf("p = %.2f  S(ρ) = %.5f ± %.1E\t(elapsed = %.2fs)\n", p, mean(svn), sem(svn), t)
  end
end

print([[rand() for _ in 1:3] for _ in 1:5])






# Define a custom sine distribution
struct SinDistribution <: ContinuousUnivariateDistribution
end

# Sampling method (inverse transform sampling)
function Base.rand(d::SinDistribution, rng::AbstractRNG=Random.GLOBAL_RNG)
    return acos(1 - 2 * rand(rng))
end

# Example usage
sin_dist = SinDistribution()

# Sampling
sample_theta = rand(sin_dist)
println("Sampled θ: ", sample_theta)

# Evaluating PDF
θ_value = π / 4
pdf_value = pdf(sin_dist, θ_value)
println("PDF at θ = π/4: ", pdf_value)

# Generate samples
samples = [rand(sin_dist) for _ in 1:10000]

# Plot histogram
histogram(samples, bins=50, normalize=true, label="Sampled Distribution")
plot!(0:0.01:π, θ -> pdf(sin_dist, θ), lw=2, label="0.5 * sin(θ)", xlabel="θ", ylabel="Density")

struct SinDistribution <: ContinuousUnivariateDistribution
end

# Sampling method (inverse transform sampling)
function Base.rand(d::SinDistribution, rng::AbstractRNG=Random.GLOBAL_RNG)
    return acos(1 - 2 * rand(rng))
end

sin_dist = SinDistribution()

# Projective Measurements onto Computational Basis
gate(::GateName"Projector0") = [1 0 ; 0 0]
gate(::GateName"Projector1") = [0 0 ; 0 1]

function gate(::GateName"ChargeConservingUnitary")
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
    return Matrix
end


testStateMPS = productstate(4)
testgatelayer = randomlayer("ChargeConservingUnitary",[(1,2), (3,4)])
testevolutionMPS = runcircuit(testStateMPS, testgatelayer)
print(testevolutionMPS[1])
print(testevolutionMPS[2])
print(testevolutionMPS[3])
print(testevolutionMPS[4])

