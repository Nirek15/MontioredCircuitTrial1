using PastaQ
using ITensors
using Printf
using LinearAlgebra
using StatsBase: mean, sem 
using Random

import PastaQ: gate

# define the two measurement projectors
gate(::GateName"Π0") =[1 0; 0 0]
gate(::GateName"Π1") = [0 0; 0 1]

function addzero()
  global zerocount += 1
end

function addone()
  global onecount += 1
end

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
  if σ == 0
    addzero()
  else
    addone()
  end
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

zerocount = 0
onecount = 0

let
  Random.seed!(1234)
  N = 8      # number of qubits
  depth = 2 * N   # circuit's depth
  ntrials = 50  # number of random trials

  # generate random circuits
  circuits = [[entangling_layer(N) for _ in 1:depth] for _ in 1:ntrials]

  # loop over projective measurement probability (per site)
  for p in 0.0:0.02:0.2
    t = @elapsed svn = monitored_circuits(circuits, p)
    @printf("p = %.2f  S(ρ) = %.5f ± %.1E\t(elapsed = %.2fs)\n", p, mean(svn), sem(svn), t)
  end
end

print(zerocount)
print(onecount)