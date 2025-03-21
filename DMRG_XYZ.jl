using ITensors, ITensorMPS

let
  # Define number of spins and create spin-one indices
  N = parse(Int,ARGS[1])
  W = parse(Int,ARGS[2])
  h = parse(Float64,ARGS[3])
  NS = parse(Int,ARGS[4])
  sites = siteinds("S=1/2", N*W)
  println("L=$N","W=$W","G=$h","NS=$NS")
  # Define the Hamiltonian for the 1D Heisenberg model
  os = OpSum()
  if W>1
   for i = 0:W-1
     for j = 0:N-1
      os += -0.04*h,"Sz",j+1+i*W, "Sz",(j+1)%N+1+i*N
      os += -0.04*h,"Sz",j+1+i*W, "Sz",j+1+((i+1)%W)*N
      os += -4.0,"Sx",j+1+i*W, "Sx",(j+1)%N+1+i*N
      os += -4.0,"Sx",j+1+i*W, "Sx",j+1+((i+1)%W)*N
      os += -4.0,"Sy",j+1+i*W, "Sy",(j+1)%N+1+i*N
      os += -4.0,"Sy",j+1+i*W, "Sy",j+1+((i+1)%W)*N

     end
   end
  else
   for j = 0:N-1
     os += -0.04*h,"Sz",j+1, "Sz",(j+1)%N+1
     os += -4.0,"Sx",j+1, "Sx",(j+1)%N+1
     os += -4.0,"Sy",j+1, "Sy",(j+1)%N+1

   end
  end
  
  H = MPO(os, sites)

  # Initialize a random MPS
  psi0 = random_mps(sites)

  # Run DMRG to find the ground state
  nsweeps = 15
  maxdim = [64,128,256,256,512,512,1024]
  cutoff = 1E-10
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

  println("Final energy = $energy")

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

filename = "DATAM5L" * ARGS[1] *"W"* ARGS[2]*"NS" * ARGS[4] * "MPSG" * ARGS[3]* ".txt"  # Output file to store configurations

# Call the function to sample and save to the file
sample_mps_to_file(psi, filename, NS)
  nothing
  
end
