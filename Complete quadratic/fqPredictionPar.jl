function fqPredictionPar(x::Array{Float64}, T::Int64, gibbsIter::Int64, burnIn::Int64, gEps1::Float64, gEps2::Float64, thLow::Float64,
  thUp::Float64, zLow::Float64, zUp::Float64, thin::Int64, samplerSeed::Int64)


	srand(samplerSeed)

  filename = "/Results/Par"

  savelocation = string(pwd(), filename, "/seed$samplerSeed")
  mkpath(savelocation)

	# initialization
	n = length(x)
  nn = Int((gibbsIter - burnIn) / thin)

  if T .> 0 # forward-prediction
    x = append!(x, zeros(T))
    for i in 1:T
      x[n+i] = rand(Uniform(-1.,1.))
    end
    ss = n + T
    xpred = zeros(nn, T)
  else # simple GSBR
    ss = n
  end


  theta = zeros(6) # polynomial coefficients vector
  sampledTheta = zeros(nn, 6) # matrix to store sample thetas

  x0s = zeros(nn, 2)
  sx01, sx02 = 0.5, 0.5

  lambdas = zeros(nn)
  slam = 0.5
  
  toler = 1e-06


	display("Starting MCMC ...")

	##########################################################################
	# MCMC
	##########################################################################

	for iter in 1:gibbsIter

      ## Sample precision
      ##########################################################################

      temp = (x[1] - henon(theta, sx01, sx02)) ^ 2 + (x[2] - henon(theta, x[1], sx01)) ^ 2
      for i in 3:ss
        temp += (x[i] - henon(theta, x[i-1], x[i-2])) ^ 2
      end
      shapel = gEps1 + 0.5 * ss
      ratel = gEps2 + 0.5 * temp
      # println("shape: $shapel")
      # println("rate: $ratel")
      slam = rand(Gamma(shapel, 1. / ratel))


      ## Sample θ₀ = theta[1]
      ##########################################################################

      muth = (x[1] - (theta[5] * sx01^2 + theta[4] * sx01 * sx02 + theta[6] * sx02^2 + theta[2] * sx01 + theta[3] * sx02)) +
                      (x[2] - (theta[5] * x[1]^2 + theta[4] * x[1] * sx01 + theta[6] * sx01^2 + theta[2] * x[1] + theta[3] * sx01)) 

      tauth = 1. + 1.

      for j = 3:ss
        muth +=  x[j] - (theta[5] * x[j-1]^2 + theta[4] * x[j-1] * x[j-2] + theta[6] * x[j-2]^2 + theta[2] * x[j-1] + theta[3] * x[j-2])
        tauth += 1.
      end
      muth = muth / tauth
      tauth = tauth * slam

      temp = -2. / tauth * log.(rand()) + (theta[1] - muth) ^ 2

      theta[1] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₁ = theta[2]
      # ##########################################################################
      #
      muth = (x[1] * sx01 - (theta[5] * sx01^3 + theta[4] * sx01^2 * sx02 + theta[6] * sx01 * sx02^2 +
                              theta[3] * sx01 * sx02 + theta[1] * sx01)) +
             (x[2] * x[1] - (theta[5] * x[1]^3 + theta[4] * x[1]^2 * sx01 + theta[6] * x[1] * sx01^2 +
                              theta[3] * x[1] * sx01 + theta[1] * x[1]))

      tauth = sx01 ^ 2 + x[1] ^ 2

      for j = 3:ss
        muth += (x[j] * x[j-1] - (theta[5] * x[j-1]^3 + theta[4] * x[j-1]^2 * x[j-2] + theta[6] * x[j-1] * x[j-2]^2 +
                                theta[3] * x[j-1] * x[j-2] + theta[1] * x[j-1]))
        tauth += x[j - 1] ^ 2
      end
      muth = muth / tauth
      tauth = tauth * slam

      temp = -2. / tauth * log.(rand()) + (theta[2] - muth) ^ 2

      theta[2] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₂ = theta[3]
      # ##########################################################################
      #
      muth =  (x[1] * sx02 - (theta[5] * sx01^2 * sx02 + theta[4] * sx01 * sx02^2 + theta[6] * sx02^3 +
                              theta[2] * sx01 * sx02 + theta[1] * sx02)) +
              (x[2] * sx01 - (theta[5] * x[1]^2 * sx01 + theta[4] * x[1] * sx01^2 + theta[6] * sx01^3 +
                              theta[2] * x[1] * sx01 + theta[1] * sx01))

      tauth = sx02 ^ 2 + sx01 ^ 2

      for j = 3:ss
        muth += (x[j] * x[j-2] - (theta[5] * x[j-1]^2 * x[j-2] + theta[4] * x[j-1] * x[j-2]^2 + theta[6] * x[j-2]^3 +
                                theta[2] * x[j-1] * x[j-2] + theta[1] * x[j-2]))
        tauth += x[j - 2] ^ 2
      end

      muth = muth / tauth
      tauth = tauth * slam

      temp = -2. / tauth * log.(rand()) + (theta[3] - muth) ^ 2

      theta[3] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₃ = theta[4]
      # ##########################################################################
      #
      muth = (x[1] * sx01 * sx02 - (theta[5] * sx01^3 * sx02 + theta[6] * sx01 * sx02^3 + theta[2] * sx01^2 * sx02 +
                              theta[3] * sx01 * sx02^2 + theta[1] * sx01 * sx02)) +
             (x[2] * x[1] * sx01 - (theta[5] * x[1]^3 * sx01 + theta[6] * x[1] * sx01^3 + theta[2] * x[1]^2 * sx01 +
                              theta[3] * x[1] * sx01^2 + theta[1] * x[1] * sx01))

      tauth =  sx01 ^ 2 * sx02 ^ 2 + x[1] ^ 2 *sx01 ^ 2

      for j = 3:ss
        muth += (x[j] * x[j-1] * x[j-2] - (theta[5] * x[j-1]^3 * x[j-2] + theta[6] * x[j-1] * x[j-2]^3 + theta[2] * x[j-1]^2 * x[j-2] +
                                theta[3] * x[j-1] * x[j-2]^2 + theta[1] * x[j-1] * x[j-2]))
        tauth += x[j - 1] ^ 2 * x[j - 2] ^ 2
      end

      muth = muth / tauth
      tauth = tauth * slam

      temp = -2. / tauth * log.(rand()) + (theta[4] - muth) ^ 2

      theta[4] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₄ = theta[5]
      # ##########################################################################
      #
      muth = (x[1] * sx01^2 - (theta[4] * sx01^3 * sx02 + theta[6] * sx01^2 * sx02^2 + theta[2] * sx01^3 +
                              theta[3] * sx01^2 * sx02 + theta[1] * sx01^2)) +
             (x[2] * x[1]^2 - (theta[4] * x[1]^3 * sx01 + theta[6] * x[1]^2 * sx01^2 + theta[2] * x[1]^3 +
                              theta[3] * x[1]^2 * sx01 + theta[1] * x[1]^2))

      tauth = sx01 ^ 4 + x[1] ^ 4

      for j = 3:ss
        muth += (x[j] * x[j-1]^2 - (theta[4] * x[j-1]^3 * x[j-2] + theta[6] * x[j-1]^2 * x[j-2]^2 + theta[2] * x[j-1]^3 +
                                theta[3] * x[j-1]^2 * x[j-2] + theta[1] * x[j-1]^2))
        tauth += x[j - 1] ^ 4
      end

      muth = muth / tauth
      tauth = tauth * slam

      temp = -2. / tauth * log.(rand()) + (theta[5] - muth) ^ 2

      theta[5] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))

      #
      # ## Sample θ₅ = theta[6]
      # ##########################################################################
      #
      muth = (x[1] * sx02^2 - (theta[5] * sx01^2 * sx02^2 + theta[4] * sx01 * sx02^3 + theta[2] * sx01 * sx02^2 +
                              theta[3] * sx02^3 + theta[1] * sx02^2)) +
             (x[2] * sx01^2 - (theta[5] * x[1]^2 * sx01^2 + theta[4] * x[1] * sx01^3 + theta[2] * x[1] * sx01^2 +
                              theta[3] * sx01^3 + theta[1] * sx01^2))

      tauth = sx02 ^ 4 + sx01 ^ 4

      for j = 3:ss
        muth += (x[j] * x[j-2]^2 - (theta[5] * x[j-1]^2 * x[j-2]^2 + theta[4] * x[j-1] * x[j-2]^3 + theta[2] * x[j-1] * x[j-2]^2 +
                                theta[3] * x[j-2]^3 + theta[1] * x[j-2]^2))
        tauth += x[j - 2] ^ 4
      end

      muth = muth / tauth
      tauth = tauth * slam

      temp = -2. / tauth * log.(rand()) + (theta[6] - muth) ^ 2

      theta[6] = rand(Uniform(max(thLow, muth - temp ^ 0.5), min(thUp, muth + temp ^ 0.5)))


      ## Sample past unobserved observations
      ##########################################################################

      if T .> 0

        if T .> 1
          for i in 1:1:(T-2)

            a4 = theta[5] ^ 2 + theta[6] ^ 2

            a3 = 2. * (theta[4] * theta[5] * x[n+i-1] + theta[2] * theta[5]) + 2. * (theta[4] * theta[6] * x[n+i+1] + theta[3] * theta[6])

            a2 = (theta[4]^2 * x[n+i-1]^2 + 2. * theta[5] * theta[6] * x[n+i-1]^2 + 2. * theta[2] * theta[4] * x[n+i-1] + 2. * theta[3] * theta[5] * x[n+i-1] -
                                2. * theta[5] * x[n+i+1] + 2. * theta[1] * theta[5] + theta[2]^2) +
                  (theta[4]^2 * x[n+i+1]^2 + 2. * theta[5] * theta[6] * x[n+i+1]^2 + 2 * theta[2] * theta[6] * x[n+i+1] + 2 * theta[3] * theta[4] * x[n+i+1] -
                                2. * theta[6] * x[n+i+2] + 2. * theta[1] * theta[6] + theta[3]^2) +
                   1.

            a1 = 2. * ( (theta[4] * theta[6] * x[n+i-1]^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * x[n+i-1]^2 + (-theta[4] * x[n+i+1] + theta[1] * theta[4] + theta[2] * theta[3]) * x[n+i-1] -
                                      theta[2] * x[n+i+1] + theta[1] * theta[2]) +
                       (theta[4] * theta[5] * x[n+i+1]^3 + (theta[2] * theta[4] + theta[3] * theta[5]) * x[n+i+1]^2 -
                                    theta[4] * x[n+i+1] * x[n+i+2] + (theta[1] * theta[4] + theta[2] * theta[3]) * x[n+i+1] - theta[3] * x[n+i+2] + theta[1] * theta[3]) - 
                      (theta[4] * x[n+i-2] * x[n+i-1] + theta[5] * x[n+i-1]^2 + theta[6] * x[n+i-2]^2 + theta[2] * x[n+i-1] + theta[3] * x[n+i-2] + theta[1])  )

            aux = -2. / slam * log.(rand()) + a1 * x[n+i] + a2 * x[n+i] ^ 2 + a3 * x[n+i] ^ 3 + a4 * x[n+i] ^ 4

            poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
            allRoots = Polynomials.roots(Poly(poly))
            sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
            realRoots = sort(real(allRoots[sel]))
            intervals = rangeIntersection(realRoots, [zLow; zUp])

            x[n+i] = unifmixrnd(intervals)

          end

          # 2nd last future unobserved observation
            a4 = theta[5] ^ 2 

            a3 = 2. * (theta[4] * theta[5] * x[n+T-2] + theta[2] * theta[5]) 

            a2 = (theta[4]^2 * x[n+T-2]^2 + 2. * theta[5] * theta[6] * x[n+T-2]^2 + 2. * theta[2] * theta[4] * x[n+T-2] + 2. * theta[3] * theta[5] * x[n+T-2] -
                                2. * theta[5] * x[n+T] + 2. * theta[1] * theta[5] + theta[2]^2) +
                 1.

            a1 = 2. * ( (theta[4] * theta[6] * x[n+T-2]^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * x[n+T-2]^2 + (-theta[4] * x[n+T] + theta[1] * theta[4] + theta[2] * theta[3]) * x[n+T-2] -
                                      theta[2] * x[n+T] + theta[1] * theta[2]) - 
                       (theta[4] * x[n+T-3] * x[n+T-2] + theta[5] * x[n+T-2]^2 + theta[6] * x[n+T-3]^2 + theta[2] * x[n+T-2] + theta[3] * x[n+T-3] + theta[1])  )

            aux = -2. / slam * log.(rand()) + a1 * x[n+T-1] + a2 * x[n+T-1] ^ 2 + a3 * x[n+T-1] ^ 3 + a4 * x[n+T-1] ^ 4

            poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
            allRoots = Polynomials.roots(Poly(poly))
            sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
            realRoots = sort(real(allRoots[sel]))
            intervals = rangeIntersection(realRoots, [zLow; zUp])

            x[n+T-1] = unifmixrnd(intervals)

        end
        
        # last future unobserved observation

        a2 = slam

        a1 = - 2. * slam * (theta[4] * x[n+T-2] * x[n+T-1] + theta[5] * x[n+T-1]^2 + theta[6] * x[n+T-2]^2 + theta[2] * x[n+T-1] + theta[3] * x[n+T-2] + theta[1]) 

        aux = -2. * log.(rand()) + a1 * x[n+T] + a2 * x[n+T] ^ 2 

        poly = [-aux / a2; a1 / a2; 1.0]
        allRoots = Polynomials.roots(Poly(poly))
        sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
        realRoots = sort(real(allRoots[sel]))
        intervals = rangeIntersection(realRoots, [zLow; zUp])

        x[n+T] = unifmixrnd(intervals)

        # a2 = 1.

        # a1 = - (theta[4] * x[n+T-2] * x[n+T-1] + theta[5] * x[n+T-1]^2 + theta[6] * x[n+T-2]^2 + theta[2] * x[n+T-1] + theta[3] * x[n+T-2] + theta[1]) 

        # aux = -2. / slam * log.(rand()) + a1 * x[n+T] + a2 * x[n+T] ^ 2 

        # poly = [-aux / a2; a1 / a2; 1.0]
        # allRoots = Polynomials.roots(Poly(poly))
        # sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
        # realRoots = sort(real(allRoots[sel]))
        # intervals = rangeIntersection(realRoots, [zLow; zUp])

        # x[n+T] = unifmixrnd(intervals)

        # new_mu = henon(theta, x[n+T-1], x[n+T-1])
        # new_var = 1./slam
        # x[n+T] = rand(Normal(new_mu, sqrt(new_var)))

      end



      ## Sample initial conditions x₀,y₀ = x0,x00
      ##########################################################################

      # x₀

      a4 = theta[5] ^ 2 + theta[6] ^ 2

      a3 = 2. * (theta[4] * theta[5] * sx02 + theta[2] * theta[5]) + 2. * (theta[4] * theta[6] * x[1] + theta[3] * theta[6])

      a2 = (theta[4]^2 * sx02^2 + 2. * theta[5] * theta[6] * sx02^2 + 2. * theta[2] * theta[4] * sx02 + 2. * theta[3] * theta[5] * sx02 -
                            2. * theta[5] * x[1] + 2. * theta[1] * theta[5] + theta[2]^2) +
            (theta[4]^2 * x[1]^2 + 2. * theta[5] * theta[6] * x[1]^2 + 2 * theta[2] * theta[6] * x[1] + 2 * theta[3] * theta[4] * x[1] -
                            2. * theta[6] * x[2] + 2. * theta[1] * theta[6] + theta[3]^2)

      a1 = 2. * ( (theta[4] * theta[6] * sx02^3 + (theta[2] * theta[6] + theta[3] * theta[4]) * sx02^2 + (-theta[4] * x[1] + theta[1] * theta[4] + theta[2] * theta[3]) * sx02 -
                                  theta[2] * x[1] + theta[1] * theta[2]) +
                  (theta[4] * theta[5] * x[1]^3 + (theta[2] * theta[4] + theta[3] * theta[5]) * x[1]^2 -
                                theta[4] * x[1] * x[2] + (theta[1] * theta[4] + theta[2] * theta[3]) * x[1] - theta[3] * x[2] + theta[1] * theta[3]))

      aux = -2. / slam * log.(rand()) + a1 * sx01 + a2 * sx01 ^ 2 + a3 * sx01 ^ 3 + a4 * sx01 ^ 4

      poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
      allRoots = Polynomials.roots(Poly(poly))
      sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
      realRoots = sort(real(allRoots[sel]))
      intervals = rangeIntersection(realRoots, [zLow; zUp])
      sx01 = unifmixrnd(intervals)

      #
      # x00 = y₀

      a4  = theta[6] ^ 2

      a3 = 2. * theta[6] * (theta[4] * sx01 + theta[3])

      a2 = ( -2. * theta[6] * (x[1] - theta[1] - theta[2] * sx01 - theta[5] * sx01^2) + (theta[4] * sx01 + theta[3])^2 )

      a1 = -2. * (x[1] - theta[1] - theta[5] * sx01^2 - theta[2] * sx01) * (theta[4] * sx01 + theta[3])

      aux = -2. / slam * log.(rand()) + a1 * sx02 + a2 * sx02 ^ 2 + a3 * sx02 ^ 3 + a4 * sx02 ^ 4

      poly = [-aux / a4; a1 / a4; a2 / a4; a3 / a4; 1.0]
      allRoots = Polynomials.roots(Poly(poly))
      sel = abs.(imag(allRoots)) .< toler # treat as real the roots with imagine part < ϵ
      # sel = imag(allRoots) .== 0.0 # treat as real the roots with imagine part < ϵ
      realRoots = sort(real(allRoots[sel]))
      intervals = rangeIntersection(realRoots, [zLow; zUp])
      sx02 = unifmixrnd(intervals)


      ## After Burn-In period
      ###############################wwwwwwwww###########################################

      if (iter .> burnIn) & ((iter-burnIn) % thin .== 0)

        ii = Int((iter - burnIn)/thin)


        # Store values
        sampledTheta[ii, :] = theta
        x0s[ii, :] = [sx01 sx02]
        lambdas[ii] = slam

        if T .> 0
          xpred[ii,:] = x[(n+1):(n+T)]
        end


      end

      if iter % 50000 .== 0
         println("MCMC Iterations: $iter")
      end

	end

	display("... MCMC finished !")

	## Write values in .txt files - specific path
	##########################################################################

  if T .> 0 
    writedlm(string(savelocation,"/xpred.txt"), xpred)
  end 
	writedlm(string(savelocation, "/thetas.txt"), sampledTheta)
	writedlm(string(savelocation,"/x0s.txt"), x0s)
  writedlm(string(savelocation,"/noise.txt"), lambdas, '\n')

 return 0

end
