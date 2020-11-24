function polyMap(Θ,z)
  g = 0.
  for i in 1:1:length(Θ)
    g += Θ[i]⋅z^(i-1)
  end
  return g
end

# function noiseMix1()
#   u = rand()
#   z = 0.
#   if u < 0.25
#     z = rand(Normal(0,sqrt(0.0001)))
#   elseif u < 0.5
#     z = rand(Normal(0, sqrt(6*0.0001)))
#   elseif u < 0.75
#     z = rand(Normal(0,sqrt(11*0.0001)))
#   else
#     z = rand(Normal(0,sqrt(16*0.0001)))
#   end
#   return z
# end

function noiseMix3(p1::Float64, lam1::Float64, lam2::Float64)
  if rand() < p1
    z = rand(Normal(0, sqrt(1 / lam1)))
  else
    z = rand(Normal(0, sqrt(1 / lam2)))
  end
  return z
end

function genData(n,Θ,x₀,p1,lam1,lam2,seed)
  srand(seed)
  x = zeros(n)
  x[1] = polyMap(Θ,x₀) + noiseMix3(p1,lam1,lam2)
  for i in 2:1:n
    x[i] = polyMap(Θ,x[i-1]) + noiseMix3(p1,lam1,lam2)
  end
  return x
end

function tgeornd(p,k)
  return rand(Geometric(p)) + k
end

function range_intersection(A::Array{Float64}, B::Array{Float64})
#=
Purpose: Range/interval intersection

 A and B two ranges of closed intervals written
 as vectors [lowerbound1 upperbound1 lowerbound2 upperbound2]
 or as matrix [lowerbound1, lowerbound2, lowerboundn;
               upperbound1, upperbound2, upperboundn]
 A and B have to be sorted in ascending order

 out is the mathematical intersection A n B

 EXAMPLE USAGE:
   >> out=rangeIntersection([1 3 5 9],[2 9])
   	out =  [2 3 5 9]
   >> out=rangeIntersection([40 44 55 58], [42 49 50 52])
   	out =  [42 44]
=#

# Allocate, as we don't know yet the size, we assume the largest case
  out1 = zeros(length(B)+(length(A)-2))
  k = 1

  while isempty(A) .== 0 && isempty(B) .== 0
  # make sure that first is ahead second
    if A[1] .> B[1]
      temp = copy(B)
      B = copy(A)
      A = copy(temp)
    end

    if A[2] .< B[1]
      A = copy(A[3:end])
      continue
    elseif A[2] .== B[1]
      out1[k] = B[1]
      out1[k + 1] = B[1]
      k = k + 2

      A = A[3:end]
      continue
    else
      if A[2] .== B[2]
        out1[k] = B[1]
        out1[k+1] = B[2]
        k = k + 2

        A = A[3:end]
        B = B[3:end]

      elseif A[2] .< B[2]
        out1[k] = B[1]
        out1[k+1] = A[2]
        k = k + 2

        A = A[3:end]
      else
        out1[k] = B[1]
        out1[k+1] = B[2]
        k = k + 2

        B = B[3:end]
      end
     end
    end

  # Remove the tails
  out = copy(out1[1:k-1])
  return out
end

function unifmixrnd(x)
  n = length(x)
  nc = 0.0
  for i in 1:2:n
    nc += x[i + 1] - x[i]
  end

  U = 0.
  u = rand()
  prob = 0.
  for i in 1:2:n
    prob += (x[i + 1] - x[i]) / nc
    if u .< prob
      U = x[i] + (x[i + 1] - x[i]) * rand()
      break
    end
  end
  return U
end

function learnc(ac, bc, c, ss, k)
  z = 0.
  x = rand(Beta(c + 1, ss))
  px = (ac + k - 1.0) / (bc - log.(x))
  px = px / (px + ss)
  r = rand(Uniform(0.0, 1.0))
  if (r .< px)
    z = rand(Gamma(ac + k, 1.0 / (bc - log.(x))))
  else
    z = rand(Gamma(ac + k - 1.0, 1.0 / (bc - log.(x))))
  end
  return z
end
