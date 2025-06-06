using ITensors, ITensorMPS

let
  # Define number of spins and create spin-one indices
  N = parse(Int,ARGS[1])
  W = parse(Int,ARGS[2])
  h = parse(Float64,ARGS[3])
  NS = parse(Int,ARGS[4])
  sites = siteinds("S=1/2", N*W)
  println("L=$N","W=$W","G=$h","NS=$NS")
  filename = "DATAM5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * "MPSG" * ARGS[3]* ".txt"  # Output file to store configurations
  filename_2 = "DATAM5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * "QIMMPSG.txt"
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
  nsweeps = 15
  maxdim = [64,64,64,128,256,256,256,400,400,512,1024,1024,1024,1024,1024]
  cutoff = 1E-10
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
  H2 = inner(H,psi,H,psi)
  var = H2-energy^2
  println("Final energy = $energy","Final var =$var")
  open(filename_2,"a") do file
      write(file,"$energy","\t","$var")
  end
  
# Function to sample MPS and store configurations in a file
function sample_mps_to_file(psi::MPS, filename::String, N::Int)
    # Open a file for writing
    open(filename, "w") do file
        for _ in 1:N
            # Sample a configuration from the MPS
            config = sample!(psi)
	    # config.-= 3/2
	    # config.*=2
	    config.=round.(Int,(config.-3/2)*2)
            # Convert the configuration into a string
            # The configuration `config` is a vector of spins at each site
            config_str = join(string.(config), " ")  # Convert each spin to a string
            println(file, config_str)  # Write to file
        end
    end
    println("Sampling complete. Configurations saved to $filename")
end



# Call the function to sample and save to the file
sample_mps_to_file(psi, filename, NS)
  nothing
  
end
