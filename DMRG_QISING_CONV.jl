using ITensors, ITensorMPS

let
  # Define number of spins and create spin-one indices
  N = parse(Int,ARGS[1])
  W = parse(Int,ARGS[2])
  h = parse(Float64,ARGS[3])
  NS = parse(Int,ARGS[4])
  sites = siteinds("S=1/2", N*W)
  println("L=$N","W=$W","G=$h","NS=$NS")
  filename = "M5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * "MPSG" * ARGS[3]* "E.txt"  # Output file to store configurations
  
# Define the Hamiltonian for the 1D Heisenberg model
  os = OpSum()
  if W>1
   for i = 0:W-2
     for j = 0:N-1
      os += -4.0,"Sz",j+1+i*N, "Sz",(j+1)%N+1+i*N
      os += -4.0,"Sz",j+1+i*N, "Sz",j+1+((i+1)%W)*N
      os += -h*0.002, "Sx", j+1+i*W
     end
   end
  else
   for j = 0:N-1
     os += -4.0,"Sz",j+1, "Sz",(j+1)%N+1
     os += -h*0.002, "Sx",j+1
   end
  end
  
  H = MPO(os, sites)

  # Initialize a random MPS
  psi0 = random_mps(sites)

  # Run DMRG to find the ground state
  maxdim = [64, 128, 128, 256, 256, 512, 512, 1024]
  cutoff = 1E-10
  nsweeps = 30
  open(filename, "W") do file

      for jj in 1:nsweeps
 
          energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
          H2 = inner(H,psi0,H,psi0)
          var = H2-energy^2
          println(file,"$energy","\t","$var")
      end	  
  end
end



# Call the function to sample and save to the file
sample_mps_to_file(psi, filename, NS)
  nothing
  
end
