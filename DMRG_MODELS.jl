using LinearAlgebra
BLAS.set_num_threads(8)
using ITensors, ITensorMPS
include("Models.jl")

using .Models
let
  # Define number of spins and create spin-one indices
  N = parse(Int,ARGS[1])
  W = parse(Int,ARGS[2])
  h = parse(Float64,ARGS[3])
  NS = parse(Int,ARGS[4])
  model = ARGS[5]
  sites = siteinds("S=1/2", N*W)
  println("L=$N","W=$W","G=$h","NS=$NS ",model)
  filename = "DATAM5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * "MPSG" * ARGS[3] * ".txt"  # Output file to store configurations
  filename_2 = "DATAM5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * model * "MPSG.txt"

  os=process_model(model,N,1,h)
  H = MPO(os, sites)

  # Initialize a random MPS
  psi0 = random_mps(sites)

  # Run DMRG to find the ground state
  nsweeps = 45
  maxdim = [64,64,64,128,256,256,256,400,400,512,1024,1024,1024,1024,1024]
  cutoff = 1E-12
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
  H2 = inner(H,psi,H,psi)
  var = H2-energy^2
  println("Final energy = $energy","Final var =$var")
  open(filename_2,"a") do file
      write(file,"$energy","\t","$var","\n")
  end

# Call the function to sample and save to the file
sample_mps_to_file(psi, filename, NS)
  nothing
  
end
