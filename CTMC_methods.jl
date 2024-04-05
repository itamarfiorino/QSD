using UnicodePlots
using BenchmarkTools
using Printf
using Random


# Generate rng
rng = MersenneTwister();


# Functions
function birth_death(n_states, mu, lambda)
   # Generate Q matrix for birth-death process
   Q = zeros(n_states, n_states)
   for i = 1 : n_states
      if i == n_states
         Q[i, i - 1] = mu
         Q[i, i] = -mu
      elseif i == 1
         Q[i, i + 1] = lambda
         Q[i, i] = -lambda
      else
         Q[i, i - 1] = mu
         Q[i, i] = -(mu + lambda)
         Q[i, i + 1] = lambda
      end
   end
   return Q
end

function birth_death_reflective(n_states, mu, lambda)
   # Generate Q matrix for birth-death process
   Q = zeros(n_states, n_states)
   for i = 1 : n_states
      if i == n_states
         Q[i, i - 1] = 2 * mu
         Q[i, i] = -2 * mu
      elseif i == 1
         Q[i, i + 1] = 2 * lambda
         Q[i, i] = -2 * lambda
      else
         Q[i, i - 1] = mu
         Q[i, i] = -(mu + lambda)
         Q[i, i + 1] = lambda
      end
   end
   return Q
end

function mesh_point(sampling_interval, h, i::Int)
   return sampling_interval[1] + (i-1) * h
end

function mesh_index(sampling_interval, h, x::Float64)::Int
   return (x - sampling_interval[1]) / h + 1
end

function rates_energy_potential(sampling_interval, h, J::Function)::Matrix{Float64}
   dim = Int(((sampling_interval[2] - sampling_interval[1]) / h) + 1)
   Q = zeros(dim, dim)
   for i = 1:dim
      x = mesh_point(sampling_interval, h, i)
      I = J(x) > 0
      if i == 1
         Q[i, i + 1] = (-J(x) * h * (1-I) + 1 / 2) / h^2
         Q[i, i] = -(-J(x) * h * (1-I) + 1 / 2) / h^2
      elseif i == dim
         Q[i, i - 1] = (J(x) * h * I + 1 / 2) / h^2
         Q[i, i] = -(J(x) * h * I + 1 / 2) / h^2
      else
         Q[i, i - 1] = (J(x) * h * I + 1 / 2) / h^2
         Q[i, i] = -(J(x) * h * I + 1 / 2) / h^2 - (-J(x) * h * (1-I) + 1 / 2) / h^2
         Q[i, i + 1] = (-J(x) * h * (1-I) + 1 / 2) / h^2
      end
   end
   return Q
end


function rates_hills(energy_potential)
   Q = zeros(length(energy_potential), length(energy_potential))
   for i = 1:length(energy_potential)
      if i == 1
         Q[i, i + 1] = exp(energy_potential[i] - energy_potential[i+1])
         Q[i, i] = -exp(energy_potential[i] - energy_potential[i+1])
      elseif i == length(energy_potential)
         Q[i, i - 1] = exp(energy_potential[i] - energy_potential[i-1])
         Q[i, i] = -exp(energy_potential[i] - energy_potential[i-1])
      else
         Q[i, i - 1] = exp(energy_potential[i] - energy_potential[i-1])
         Q[i, i] = -exp(energy_potential[i] - energy_potential[i-1]) -exp(energy_potential[i] - energy_potential[i+1])
         Q[i, i + 1] = exp(energy_potential[i] - energy_potential[i+1])
      end
   end
   return Q
end

function Q_hat(Q, absorbing)
   Q_hat = copy(Q)
   for i in absorbing
      Q_hat[i,:] .= 0
   end
   return Q_hat
end


# Validate Q Matrix
function validate(Q)
   for i in 1: n_states
      for j in 1:n_states
         if i == j
            if Q[i, j] >= 0
               return false
            end
         else
            if Q[i,j] <= 0
               return false
            end
         end
      end
      if sum(Q[1,:]) != 0
         return false
      end
   end
   return true
end

# Jumps to corresponding state.
function state(currentState, Q, val)
   buffer = 0
   z = - Q[currentState, currentState]
   for i in 1:size(Q)[1]
      if i == currentState
         continue
      end
      buffer = buffer + Q[currentState, i] / z
      if buffer > val
         return i
      end
   end
end


function launch_particle(Q, n_samples, n_states)
   x = randexp(rng, n_samples)
   r = rand(rng, n_samples)
   y = zeros(Int8, n_samples)
   y[1] = ceil(r[1] * n_states)
   x[1] = x[1] / -Q[y[1], y[1]]
   for i in 2 : n_samples
   #   @printf "State accessed: %i, value: %f\n" y[i-1] r[i]
      y[i] = state(y[i-1], Q, r[i])
      x[i] = x[i-1] + -x[i] / Q[y[i], y[i]]
   end
   pushfirst!(x, 0)
   push!(y, last(y))
   return x, y
end


# mu = 1
# lambda = 3
# size = 10
# length = 20
# Q = birth_death(size, mu, lambda)
# if validate(Q)
#    print("approved")
# end
# x, y = launch_particle(Q, length, size)
# display(x)
# display(y)
# s = stairs(x, y, width = 150, color=:red, style=:post, height=10, title="First Simulation")
# display(s)
