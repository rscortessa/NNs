using LinearAlgebra
using Statistics
using StatsBase
BLAS.set_num_threads(4)
using ITensors, ITensorMPS
include("Models.jl")

using .Models
let
  # Define number of spins and create spin-one indices
  N = parse(Int,ARGS[1])
  W = parse(Int,ARGS[2])
  h = parse(Float64,ARGS[3])
  Nh = parse(Int,ARGS[4])
  NS = parse(Int,ARGS[5])
  NR = parse(Int,ARGS[6])
  NC = 1.0
  sites = siteinds("S=1/2", N*W)
  # Initialize a random MPS

  folder="RUN_QIM_DMRG_L" * ARGS[1] * "W" * ARGS[2]* "NS" * ARGS[5] * "G" * ARGS[3] * "ANG0-90NANG" * ARGS[4] * "NR" * ARGS[6]
  mkpath(folder)


  filename_2 = folder*"/"*"DATAM5L" * ARGS[1] * "W" * ARGS[2]* "G" * ARGS[3] * "NS" * ARGS[5] * "R_QIMMPS" * ".txt" # Output file optimization
  open(filename_2,"a") do file
      write(file,"G","\t","E","\t","E_var","\n")        
  end

   
  for iter in 0:Nh

      
      theta=iter*1.0
      En=zeros(NR)
      var=0
      
      println("L=$N","W=$W","G=$theta","NS=$NS ","R_QIM")
      angle=theta*pi/(2*Nh)
      os=rotated_quantum_ising(N,W,h,angle)
      H = MPO(os, sites)
      
      
      for sample_iter in 1:NR

      	  psi0 = random_mps(sites)
      	  filename = folder*"/"*"DATAM5L" * ARGS[1] * "W" * ARGS[2] * "NS" * ARGS[5] * "MPSG" * string(Int(theta)) * ".txt" * string(sample_iter)  # Output file to store configurations

	  
	  # Run DMRG to find the ground state
      	  nsweeps = 45
      	  maxdim = [64,64,64,128,256,256,256,400,400,512,1024,1024,1024,1024,1024]
      	  cutoff = 1E-14
      	  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)


	  H2 = inner(H,psi,H,psi)
	  var = H2-energy^2
	  En[sample_iter]=energy
	 
      	  println("Final energy = $energy","Final var =$var")
      	  # Call the function to sample and save to the file
      	  sample_mps_to_file(psi, filename, NS)
      	  	nothing
		
      end
      if NR>1
      		E_mean=mean(En)
      		E_var=sqrt(var(En)/sqrt(Float64(NR)))
	else
		E_mean=En[1]
		E_var=var
	end
	

      open(filename_2,"a") do file
              write(file,"$theta","\t","$E_mean","\t","$E_var","\t","\n")
      	  end
  end	
  
end
