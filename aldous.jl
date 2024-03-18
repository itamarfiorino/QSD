include("CTMC_methods.jl");


######### Aldous scheme #########
# Procedure:
#  (a) Generate random variables (uniform/exponential)
#  (b) Travel to starting state
#  (c) Determine occupation and update occupation measure/normalizer
#  (d) Travel to following state
#  (e) If state is in the subset of absorbing states, change state to random state via. occupation measure
#  (f) repeat (c) - (e) for fixed (time/jumps)
# Soft killing uses minimum of exponentials (heuristically seperate from movement of particle)
# Inherits methods from CTMC_methods.jl


function kill(occupation_measure, time, r)
   # After killing, determines what state to travel to based on the occupation measure / time
   
   # Validate occupation measure
   epsilon = 0.01
   if abs(sum(occupation_measure) - time) > epsilon
      throw(DomainError(time, "Issues with given occupation measure and time"))
   end

   # Resample via occupation_measure (occupation measure will be zero on absorbed states)
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
   occupation_measure = zeros(Float64, size)
   
   # Iterate Aldous process
   y[1] = rand(continuation)
   x[1] = 0
   for i in 2:length
      # Determine occupation time and evolve state
      occupation_measure[y[i-1]] += x[i] / -Q[y[i-1], y[i-1]]
      x[i] = x[i-1] + x[i] / -Q[y[i-1], y[i-1]]
      y[i] = state(y[i-1], Q, r[i])

      # If absorbed, move to a random location based on occupation time
      if y[i] in absorbing
         y[i] = kill(occupation_measure, x[i], rand(rng))
         num_kills += 1
      end
   end
   return x, y, occupation_measure, num_kills
end

function aldous_soft_killing(length, size, Q, absorbing, soft_killing)
   # Initialize random vectors
   continuation = setdiff(1:size, absorbing) # Continuation states
   x = randexp(rng, length)
   s = randexp(rng, length)
   r = rand(rng, length)
   y = zeros(Int8, length)
   num_kills = 0
   occupation_measure = zeros(Float32, size)
   
   # Iterate Aldous process
   y[1] = rand(continuation)
   x[1] = 0
   for i in 2:length
      # Determine transition and kill time and take minimum to determine occupation time
      kill_time = s[i] / soft_killing[y[i-1]]
      transition_time = x[i] / -Q[y[i-1], y[i-1]]
      elapsed_time = min(kill_time, transition_time)
      occupation_measure[y[i-1]] += elapsed_time
      x[i] = x[i-1] + elapsed_time

      # If killed by absorption or softly, move to location via. occupation measure
      if y[i] in absorbing || kill_time < transition_time
         y[i] = kill(occupation_measure, x[i], rand(rng))
         num_kills += 1
         continue
      end
      y[i] = state(y[i-1], Q, r[i])
   end
   return x, y, occupation_measure, num_kills
end

# Constants
size = 15 # Number of states
length = 50000 # Number of jumps to compute
lambda = .5 # Mass to the right
mu = 1 # Mass to the left
absorbing = [1, 15] # Absorbing states

Q = birth_death_reflective(size, mu, lambda)
# display(Q)
x, y, occupation_measure, num_kills = aldous(length, size, Q, absorbing)

@printf "Times Killed: %i" num_kills
s = stairs(x, y, ylim = (1,15), xlim = (10,30), width = 50, color=:blue, style=:post, height=10, title="Aldous Simulation")
display(s)
p = barplot(1:size, occupation_measure, title="Aldous Occupation Measure")
display(p)
