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
  NC = 0.2
  sites = siteinds("S=1/2", N*W)
  # Initialize a random MPS
  psi0 = random_mps(sites)
   
  filename_2 = "DATAM5L" * ARGS[1] * "W" * ARGS[2]* "NS" * ARGS[6] * "R_QIMMPS" * ".txt" # Output file optimization

   
  for iter in 0:Nh

      theta=iter

      En=zeros(NR)
      OZ=zeros(NR)
      OZZ=zeros(NR)
      
      println("L=$N","W=$W","G=$theta","NS=$NS ","R_QIM")
      angle=theta*pi/Nh
      os=rotated_quantum_ising(N,W,h,angle)
      H = MPO(os, sites)
      
      
      for sample_iter in 1:NR
      
      	  filename = "DATAM5L" * ARGS[1] * "W" * ARGS[2] * "NS" * ARGS[6] * "MPSG" * string(Int(theta)) * ".txt" * string(sample_iter)  # Output file to store configurations

	  
	  # Run DMRG to find the ground state
      	  nsweeps = 30
      	  maxdim = [64,64,64,128,256,256,256,400,400,512,1024,1024,1024,1024,1024]
      	  cutoff = 1E-11
      	  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

	  psi0=psi

	  H2 = inner(H,psi,H,psi)

	  #AV_OZ=O_Z(psi,N)	  
	  #AV_OZZ=CHAIN_O_Z(psi,N)
	  var = H2-energy^2
	  
	  En[sample_iter]=energy
	  #OZ[sample_iter]=AV_OZ
	  #OZZ[sample_iter]=AV_OZZ
      	  println("Final energy = $energy","Final var =$var")
      	  

      	  # Call the function to sample and save to the file
      	  sample_mps_to_file(psi, filename, NS)
      	  	nothing
		
	  NE=Int( floor(NC*length(psi)))
	  indices_to_modify=rand(1:length(psi0),NE)
	  for K in indices_to_modify
    	      tensor_data = ITensors.data(psi0[K])  # Access raw data
    	      tensor_data .+= randn(eltype(tensor_data), size(tensor_data))
    	      psi0[K] = ITensor(tensor_data, inds(psi0[K])...)  # Reconstruct the ITensor with modified data
	  end

	  
      end
      E_mean=mean(En)
      #E_var=sqrt(var(En)/sqrt(Float64(NR)))
      E_var=var
      OZ_mean=mean(OZ)
      OZZ_mean=mean(OZZ)
      
      open(filename_2,"a") do file
      	      write(file,"G","\t","E","\t","E_var","\t","OZ","\t","OZZ","\n")
              write(file,"$theta","\t","$E_mean","\t","$E_var","\t","$OZ_mean","\t","$OZZ_mean","\n")
      	  end
  end	
  
end
