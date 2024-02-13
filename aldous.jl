using UnicodePlots
using Plots
using Printf
using Random


# Generate rng
rng = MersenneTwister();

# Fixed parameters
size = 10
length = 75
lambda = 5 # Mass to the right
mu = 1 # Mass to the left

# Functions
function birth_death(size, mu, lambda)
   # Generate Q matrix for birth-death process
   Q = zeros(size, size)
   for i = 1 : size
      if i == size
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

function birth_death_reflective(size, mu, lambda)
   # Generate Q matrix for birth-death process
   Q = zeros(size, size)
   for i = 1 : size
      if i == size
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


# Validate Q Matrix
function validate(Q)
   for i in 1: size
      for j in 1:size
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
   for i in 1: size
      if i == currentState
         continue
      end
      buffer = buffer + Q[currentState, i] / z
      if buffer > val
         return i
      end
   end
end


# Generate random vector
# Assumptions:
#     Initial distribution is uniform
Q = birth_death(size, mu, lambda)
x = randexp(rng, length)
r = rand(rng, length)
y = zeros(Int8, length)
y[1] = ceil(r[1] * size)
x[1] = x[1] * -Q[y[1], y[1]]
for i in 2 : length
#   @printf "State accessed: %i, value: %f\n" y[i-1] r[i]
   y[i] = state(y[i-1], Q, r[i])
   x[i] = x[i-1] + -x[i] * Q[y[i], y[i]]
end
        
# Display sample path
# pop!(x)
pushfirst!(x, 0)
push!(y, last(y))
# display(x)
# display(y)
s = stairs(x, y, width = 150, color=:red, style=:post, height=10, title="First Simulation")
display(s)

######### Aldous scheme #########
# Procedure:
#  (a) Generate random variables (uniform/exponential)
#  (b) Travel to starting state
#  (c) Determine occupation and update occupation measure/normalizer
#  (d) Travel to following state
#  (e) If state is in the subset of absorbing states, change state to random state via. occupation measure
#  (f) repeat (c) - (e) for fixed (time/jumps)
# Inherits state function

# Define constants
size = 15 # Number of states
length = 50000 # Number of jumps to compute
lambda = 5 # Mass to the right
mu = 5 # Mass to the left
absorbing = [1,15] # Absorbing states


function kill(occupation_measure, time, r)
   # After killing, determines what state to travel to based on the occupation measure / time
   
   # Validate occupation measure
   epsilon = 0.5
   if abs(sum(occupation_measure) - time) > epsilon
      throw(DomainError(time, "Issues with given occupation measure and time"))
   end

   r = r * time
   for i in 1:size
      r -= occupation_measure[i]
      if r <= 0
         return i
      end
   end
   return size
end

function aldous(length, size, Q, absorbing)
   # Initialize random vectors
   continuation = setdiff(1:size, absorbing) # Continuation states
   x = randexp(rng, length)
   r = rand(rng, length)
   y = zeros(Int8, length)
   num_kills = 0
   occupation_measure = zeros(Float32, size)
   
   # Iterate Aldous process
   y[1] = rand(continuation)
   x[1] = 0
   for i in 2:length
      occupation_measure[y[i-1]] += x[i] * -Q[y[i-1], y[i-1]]
      x[i] = x[i-1] + x[i] * -Q[y[i-1], y[i-1]]
      y[i] = state(y[i-1], Q, r[i])
      if y[i] in absorbing
         y[i] = kill(occupation_measure, x[i], rand(rng))
         num_kills += 1
      end
   end
   return x, y, occupation_measure, num_kills
end

Q = birth_death_reflective(size, mu, lambda)
x, y, occupation_measure, num_kills = aldous(length, size, Q, absorbing)

@printf "Times Killed: %i" num_kills
s = stairs(x, y, xlim = (100,400), width = 150, color=:blue, style=:post, height=10, title="Aldous Simulation")
display(s)
p = barplot(1:size, occupation_measure, title="Aldous Occupation Measure")
display(p)
