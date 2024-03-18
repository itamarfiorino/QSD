include("CTMC_methods.jl");

######### Flemming-Viot scheme #########
# Procedure:
#  (a) Initialize particles and sort by last time
#  (b) Move earliest particle to new location
#  (c) If particle exits, move to another particle
#  (d) Determine new update time and reorder by insertion
#  (e) Repeat from (b)
# Soft killing uses sum of exponentials (i.e. soft killing as a "state")
# Particles is a dictionary for speed/memory allocation
# Inherits methods from CTMC_methods.jl

mutable struct Particle
   state::UInt16
   update_time::Float32
end

function initialize(size, absorbing, n, Q)
   # Pre-allocate response using sizehint! for speed
   res = Dict{UInt16, Particle}()
   sizehint!(res, n)

   # Find allowed states, travel, and stay; use arrays for initialization and transfer to dictionary
   s = setdiff(1:size, absorbing)
   r = rand(s, n)
   re = randexp(n) ./ map(x -> -Q[x,x], r)
   particles = [Particle(r[i],re[i]) for i in 1:n]
   sort!(particles, by=p -> p.update_time) # Sort by update_time
   for i in 1:n
      res[i] = particles[i]
   end
   return res
end

function initialize_soft_killing(size, absorbing, n, Q, soft_killing)
   # Pre-allocate response using sizehint! for speed
   res = Dict{UInt16, Particle}()
   sizehint!(res, n)

   # Find allowed states, travel, and stay; use arrays for initialization and transfer to dictionary
   s = setdiff(1:size, absorbing)
   r = rand(s, n)
   re = randexp(n) ./ map(x -> -Q[x,x] + soft_killing[x], r)
   particles = [Particle(r[i],re[i]) for i in 1:n]
   sort!(particles, by=p -> p.update_time)
   for i in 1:n
      res[i] = particles[i]
   end
   return res
end

function swap!(d, a, b)
   t = d[a]
   d[a] = d[b]
   d[b] = t
end

function shift_blind!(particles, n)
   # Shifts the current particle (i.e. the earliest one) blindly until positioned correctly within particles
   t = particles[1].update_time
   i = 2
   while particles[i].update_time < t
      swap!(particles, i-1, i)
      i = i < n ? i + 1 : break
   end
end

function move!(particles, size, absorbing, n, Q)
   # Moves the earliest particle by determining state, resampling if absorbed, and staying for a period of time
   next = state(particles[1].state, Q, rand())
   killed = next in absorbing
   if killed
      next_particle = rand(2:n)
      next = particles[next_particle].state
      # @printf "Moving to particle matching %i at %i" next_particle next
   end
   t = particles[1].update_time + randexp() / -Q[next, next]
   particles[1].update_time = t
   particles[1].state = next
   shift_blind!(particles, n)
   return killed
end

function move_soft_killing!(particles, size, absorbing, n, Q, soft_killing)
   # Find where the particle is
   previous_state = particles[1].state

   # Determine whether the particle is killed and find the next state accordingly
   r = rand()
   q = soft_killing[previous_state] / (soft_killing[previous_state] - Q[previous_state, previous_state]) # Criteria for soft killing
   killed_softly = r < q 
   next = state(previous_state, Q, (r - q) / (1 - q))
   killed = killed_softly || next in absorbing

   # Resample particle if killed
   if killed
      next_particle = rand(2:n)
      next = particles[next_particle].state
      # @printf "Moving to particle matching %i at %i" next_particle next
   end

   # Determine stay time including soft killing
   t = particles[1].update_time + randexp() / (-Q[next, next] + soft_killing[next])
   particles[1].update_time = t
   particles[1].state = next
   shift_blind!(particles, n)
   return killed
end

function rel!(particles, n)
   z = particles[1].update_time
   for i in 1:n
      particles[i].update_time -= z
   end
   return z
end

function batch_soft_killing!(particles, size, absorbing, n, Q)
   # Runs a batch maintaining floating point accuracy for Float64
   # eps(Float64(FP_LIMIT)) is the increment size
   FP_LIMIT = 100000
   TIMEOUT_LIMIT = 100
   l = - mean([Q[i,i] for i in 1:size]) # Does not account for relative frequency (involves QSD)
   batch_size = min(FP_LIMIT * n/l, TIMEOUT_LIMIT)
   kills = 0
   for i in 1:(batch_size)
      kills += move_soft_killing!(particles, size, absorbing, n, Q) ? 1 : 0
   end
   scale = rel!(particles, n)
   @printf "Batch size: %g\n\tScale: %g\n\tPrecision %g" batch_size scale eps(Float64(scale))
   return kills, scale
end

function batch!(particles, size, absorbing, n, Q)
   # Runs a batch maintaining floating point accuracy for Float64
   # eps(Float64(FP_LIMIT)) is the increment size
   FP_LIMIT = 100000
   TIMEOUT_LIMIT = 500000
   l = - mean([Q[i,i] for i in 1:size]) # Does not account for relative frequency (involves QSD)
   batch_size = min(FP_LIMIT * n/l, TIMEOUT_LIMIT)
   kills = 0
   for i in 1:(batch_size)
      kills += move!(particles, size, absorbing, n, Q) ? 1 : 0
   end
   scale = rel!(particles, n)
   @printf "Batch size: %g\n\tScale: %g\n\tPrecision %g\n\tKills: %g" batch_size scale eps(Float64(scale)) kills
   return kills, scale
end
function snapshot(particles, size)
   msr = [a.state for a in values(particles)]
   println(length(msr))
   println(size)
   return histogram(msr, title="Flemming-Viot Empirical Measure", nbins = size)
end

# Constants
n = 10000 # Number of particles
size = 15 # Number of states
mu = 1
lambda = .5
absorbing = [1,15]


t = @time Q = birth_death_reflective(size, mu, lambda);
# particles = initialize(size, absorbing, n, Q);
# kills, elapsed = batch!(particles, size, absorbing, n, Q)
# decay_rate = (1 - 1/n)^kills / elapsed
# @printf "\nEstimate of decay rate: %f" decay_rate
# display(snapshot(particles, size))
